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
        text_embeddings,
        dataset,
        contrastive_alpha=0.5,
        contrastive_tau=0.07,
        manifold_beta=0.2,
        manifold_c=0.2,
        n_codebook=None,
        code_weight_tau=1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        # `text_embeddings` is item-level text embedding, typically of
        # dimension different from `hidden_dim` (e.g., 768 vs T5 d_model).
        # We record its original dimension and project it to `hidden_dim`
        # inside `text_mlp`.
        self.text_embeddings = text_embeddings
        self.text_dim = text_embeddings.shape[1]
        self.dataset = dataset

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
            self.code_weights = nn.Parameter(torch.zeros(n_codebook))
        else:
            self.code_weights = None

        self._text_embeddings_device_cache = {}

    # ---------------- hyperbolic ----------------

    @staticmethod
    def hyp_map(x, c):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        scale = c * torch.tanh(norm / (2 * c)) / norm
        return scale * x

    # ---------------- forward ----------------

    def forward(self, id_embeddings, batch):

        B, S, D = id_embeddings.shape
        device = id_embeddings.device

        # ========= text embedding =========

        item_ids = batch["input_ids"]

        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            self._text_embeddings_device_cache[device_key] = self.text_embeddings.to(device)

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
        e_mix = self.hyp_map(self.proj_psi(fused.reshape(-1, D)), self.manifold_c)

        diff_sq = torch.sum((z_hyp - e_mix) ** 2, dim=-1)

        z_norm = torch.clamp(self.manifold_c**2 - torch.sum(z_hyp**2, dim=-1), min=1e-6)
        e_norm = torch.clamp(self.manifold_c**2 - torch.sum(e_mix**2, dim=-1), min=1e-6)

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
