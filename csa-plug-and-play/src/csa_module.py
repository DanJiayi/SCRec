import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAModule(nn.Module):
    """
    Contrastive + Semantic + Alignment Module
    Plug-and-play for TIGER / T5 / GPT style models
    """

    def __init__(
        self,
        hidden_dim,
        text_dim,
        dataset,
        contrastive_alpha=0.5,
        contrastive_tau=0.07,
        manifold_beta=0.2,
        manifold_c=0.2,
        n_codebook=4,
        code_weight_tau=1.0,
        max_items_per_seq=20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        # `text_dim` is the dimension of item-level text embeddings,
        # typically different from `hidden_dim` (e.g., 768 vs T5 d_model).
        # We project it to `hidden_dim` inside `text_mlp`.
        self.text_dim = text_dim
        self.dataset = dataset
        self.max_items_per_seq = max_items_per_seq

        # fusion
        self.text_mlp = nn.Linear(self.text_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # contrastive
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_tau = contrastive_tau

        # manifold
        self.manifold_beta = manifold_beta
        self.manifold_c = manifold_c
        self.proj_phi = nn.Linear(hidden_dim, hidden_dim)
        self.proj_psi = nn.Linear(hidden_dim, hidden_dim)

        # Optional learnable weights for aggregating code-level embeddings
        # into item-level ID embeddings (length = number of semantic codes).
        self.n_codebook = n_codebook
        self.code_weight_tau = code_weight_tau
        if n_codebook is not None:
            # Use text embeddings to dynamically generate per-code weights
            self.code_weight_fc = nn.Linear(self.text_dim, n_codebook)
        else:
            self.code_weight_fc = None

        # Cache for padded text embeddings keyed by device string.
        # Populated lazily in _prepare_text_embeddings().
        self._text_embeddings_device_cache = {}

    # ---------------- helpers ----------------

    @staticmethod
    def _prepare_text_embeddings(item_embedding):
        """
        Pad item-level text embeddings with a zero row at index 0 so that
        padding / user-id positions can safely map to 0.  The result is
        kept on CPU (matching the original behaviour in training.py) and
        will be moved to the target device lazily via
        ``_text_embeddings_device_cache``.

        Args:
            item_embedding: Tensor of shape (n_items, text_dim).

        Returns:
            text_embeddings_with_pad: Tensor of shape (n_items+1, text_dim) on CPU.
        """
        with torch.no_grad():
            if item_embedding.device != torch.device("cpu"):
                text_embeddings_cpu = item_embedding.detach().cpu()
            else:
                text_embeddings_cpu = item_embedding
            pad_row = torch.zeros(
                1, text_embeddings_cpu.size(1), dtype=text_embeddings_cpu.dtype
            )
            text_embeddings_with_pad = torch.cat(
                [pad_row, text_embeddings_cpu], dim=0
            )
        return text_embeddings_with_pad

    # ---------------- hyperbolic ----------------

    @staticmethod
    def hyp_map(x, c):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        scale = c * torch.tanh(norm / (2 * c)) / norm
        return scale * x

    # ---------------- prepare ----------------

    def prepare(self, batch, device, method_config, model, item_embedding):
        """
        Prepare id_embeddings and a CSA-specific batch dict from the raw
        training batch.  Returns (id_embeddings, batch_for_csa) when the
        sequence is long enough to hold all code tokens; otherwise returns
        (None, None) so that the caller can skip CSA gracefully.

        This method encapsulates the logic that was previously inlined in
        training.py, making the CSA module self-contained.
        """
        # Build padded text embeddings (on CPU) and populate the
        # per-device cache so that get_code_weights / forward can look
        # them up by device key.
        text_embeddings_with_pad = self._prepare_text_embeddings(item_embedding)
        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            self._text_embeddings_device_cache[device_key] = (
                text_embeddings_with_pad.to(device)
            )

        # input_sids: [B, seq_len]
        input_sids = batch["input_sids"].to(device)

        # Code tokens start after the (optional) user id.
        item_idx_start = 1 if method_config["include_user_id"] else 0
        seq_len = input_sids.size(1)

        if not (
            self.n_codebook is not None
            and item_idx_start + self.n_codebook * self.max_items_per_seq <= seq_len
        ):
            return None, None

        # Obtain raw code-level token embeddings from the shared
        # embedding matrix: [B, seq_len, D]
        token_embeds = model.shared(input_sids)

        # Slice out the region that corresponds to item semantic IDs
        # and reshape to [B, max_items_per_seq, n_codebook, D].
        code_region = token_embeds[
            :,
            item_idx_start : item_idx_start
            + self.n_codebook * self.max_items_per_seq,
            :,
        ]
        B, _, D = code_region.shape
        code_region = code_region.view(
            B, self.max_items_per_seq, self.n_codebook, D
        )

        # Build a batch view for CSA: we align item_ids with the
        # aggregated item-level embeddings (length = max_items_per_seq).
        batch_for_csa = dict(batch)
        full_input_ids = batch_for_csa["input_ids"].to(device)
        if method_config["include_user_id"]:
            # Skip the user-id position when present.
            item_ids = full_input_ids[:, 1 : 1 + self.max_items_per_seq]
        else:
            item_ids = full_input_ids[:, : self.max_items_per_seq]
        # If for any reason the sequence is shorter, pad with zeros.
        if item_ids.size(1) < self.max_items_per_seq:
            pad_cols = self.max_items_per_seq - item_ids.size(1)
            pad = torch.zeros(
                item_ids.size(0),
                pad_cols,
                dtype=item_ids.dtype,
                device=item_ids.device,
            )
            item_ids = torch.cat([item_ids, pad], dim=1)

        batch_for_csa["input_ids"] = item_ids

        # Dynamic weights over code positions, generated from text
        # embeddings via a single linear layer inside CSAModule.
        # weights: (B, max_items_per_seq, L)
        weights = self.get_code_weights(
            batch_for_csa, device=code_region.device
        )
        if weights is None:
            raise ValueError(
                "CSAModule.n_codebook must be set to use dynamic code weights."
            )
        # Apply position-specific weights to code embeddings
        id_embeddings = (code_region * weights[..., :, None]).sum(
            dim=2
        )  # [B, max_items_per_seq, D]

        return id_embeddings, batch_for_csa

    # ---------------- forward ----------------

    def get_code_weights(self, batch, device):
        """
        Compute dynamic per-code weights from text embeddings.
        Returns a tensor of shape (B, S, L) where:
          B = batch size, S = sequence length, L = n_codebook.
        """
        if self.n_codebook is None or self.code_weight_fc is None:
            return None

        item_ids = batch["input_ids"].to(device)  # (B, S)

        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            raise RuntimeError(
                "_text_embeddings_device_cache not populated. "
                "Call prepare() before get_code_weights()."
            )

        text_emb = self._text_embeddings_device_cache[device_key][item_ids]  # (B, S, text_dim)
        code_logits = self.code_weight_fc(text_emb)  # (B, S, L)
        weights = torch.softmax(code_logits / self.code_weight_tau, dim=-1)
        return weights

    def forward(self, batch, device, method_config, model, item_embedding):

        # ---- Step 1: prepare id_embeddings from the raw batch ----
        id_embeddings, batch_for_csa = self.prepare(
            batch, device, method_config, model, item_embedding
        )
        if id_embeddings is None:
            # Sequence too short for CSA; skip gracefully.
            return None, {"total": 0.0}

        B, S, D = id_embeddings.shape

        # ========= text embedding =========

        item_ids = batch_for_csa["input_ids"]

        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            raise RuntimeError(
                "_text_embeddings_device_cache not populated. "
                "Call prepare() before accessing text embeddings."
            )

        text_emb = self._text_embeddings_device_cache[device_key][item_ids]
        e_sem = self.text_mlp(text_emb)

        # ========= gated fusion =========

        gate_inp = torch.cat([id_embeddings, e_sem], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        fused = g * id_embeddings + (1 - g) * e_sem

        losses = {}

        # ================= InfoNCE =================

        e_cf = F.normalize(id_embeddings.reshape(-1, D), dim=-1)
        e_sem_flat = F.normalize(e_sem.reshape(-1, D), dim=-1)

        logits = torch.matmul(e_cf, e_sem_flat.T) / self.contrastive_tau
        labels = torch.arange(logits.size(0), device=device)

        contrastive_loss = F.cross_entropy(logits, labels)
        losses["contrastive"] = contrastive_loss

        # ================= Hyperbolic =================

        z_hyp = self.hyp_map(self.proj_phi(e_cf), self.manifold_c)
        e_hyp = self.hyp_map(self.proj_psi(e_sem.reshape(-1, D)), self.manifold_c) #fused.reshape(-1, D)

        diff_sq = torch.sum((z_hyp - e_hyp) ** 2, dim=-1)
        z_norm = torch.clamp(self.manifold_c**2 - torch.sum(z_hyp**2, dim=-1), min=1e-6)
        e_norm = torch.clamp(self.manifold_c**2 - torch.sum(e_hyp**2, dim=-1), min=1e-6)

        arg = 1 + 2 * (self.manifold_c**2) * diff_sq / (z_norm * e_norm)
        manifold_dist = torch.acosh(torch.clamp(arg, min=1 + 1e-5))
        manifold_loss = manifold_dist.mean()

        losses["manifold"] = manifold_loss

        total = (
            self.contrastive_alpha * contrastive_loss
            + self.manifold_beta * manifold_loss
        )

        losses["total"] = total

        return fused, losses
