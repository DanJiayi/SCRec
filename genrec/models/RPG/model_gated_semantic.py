# Copyright (c) Meta Platforms, Inc. and affiliates.
# Gated semantic aggregation: e_cf = sum_l W^(l) v^(l), e_sem = MLP_frozen(h^text), gate, e_mix = g*e_cf + (1-g)*e_sem.

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer

from genrec.models.RPG.model import ResBlock


def load_semantic_embeddings_for_dataset(
    json_path: str,
    id2item: list,
    emb_index: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Load h^text (e.g. emb1) from item_id:[emb1,emb2,emb3] JSON.
    Returns tensor (n_items, text_dim) aligned with id2item; missing items get zeros.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    # Infer dimension from first available embedding
    text_dim = None
    for item_id, emb_list in data.items():
        if isinstance(emb_list, list) and len(emb_list) > emb_index:
            text_dim = len(emb_list[emb_index])
            break
    if text_dim is None:
        raise ValueError(f"No valid embedding at index {emb_index} in {json_path}")
    n_items = len(id2item)
    out = np.zeros((n_items, text_dim), dtype=np.float32)
    for i in range(n_items):
        item_str = id2item[i]
        if item_str == "[PAD]":
            continue
        if item_str in data and isinstance(data[item_str], list) and len(data[item_str]) > emb_index:
            out[i] = np.asarray(data[item_str][emb_index], dtype=np.float32)
    t = torch.from_numpy(out)
    if device is not None:
        t = t.to(device)
    return t


class RPGGatedSemantic(AbstractModel):
    """
    RPG with gated aggregation:
    - e_cf = sum_{l=1}^{L} W^{(l)} v_{c_i}^{(l)}  (per-layer linear then sum)
    - e_sem = MLP(h_i^{text})
    - g_i = sigmoid(W_g [e_cf; e_sem])
    - e_mix = g_i * e_cf + (1 - g_i) * e_sem
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(RPGGatedSemantic, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config["device"])
        n_embd = config["n_embd"]
        n_digit = tokenizer.n_digit

        # Semantic embeddings: h^text from JSON (emb1), then MLP_frozen -> e_sem
        # semantic_emb_path = config.get(
        #     "semantic_emb_path",
        #     "/root/test/preprocess/emb_{category}.json",
        # )
        semantic_emb_path = f"/root/test/preprocess/emb_Beauty.json"
        # if "{category}" in semantic_emb_path and "category" in config:
        #     semantic_emb_path = semantic_emb_path.format(
        #         category=config["category"]
        #     )
        print("semantic_emb_path: ", semantic_emb_path)
        semantic_emb_index = config.get("semantic_emb_index", 0)
        id2item = dataset.id_mapping["id2item"]
        self.register_buffer(
            "_item_text_emb",
            load_semantic_embeddings_for_dataset(
                semantic_emb_path,
                id2item,
                emb_index=semantic_emb_index,
                device=None,
            ),
        )
        text_dim = self._item_text_emb.shape[1]

        # MLP_frozen: h^text -> n_embd (frozen)
        self.mlp_sem = nn.Sequential(
            nn.Linear(text_dim, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )
        # for p in self.mlp_sem.parameters():
        #     p.requires_grad = False

        # Per-layer linear W^{(l)}: n_embd -> n_embd
        self.W_cf = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_digit)])

        # Gate: W_g [e_cf; e_sem] -> n_embd, then sigmoid
        self.W_gate = nn.Linear(2 * n_embd, n_embd)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=n_embd,
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=config["n_inner"],
            activation_function=config["activation_function"],
            resid_pdrop=config["resid_pdrop"],
            embd_pdrop=config["embd_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            initializer_range=config["initializer_range"],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = n_digit
        self.pred_heads = nn.Sequential(
            *[ResBlock(n_embd) for _ in range(self.n_pred_head)]
        )
        self.temperature = config["temperature"]
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config["chunk_size"]
        self.num_beams = config["num_beams"]
        self.n_edges = config["n_edges"]
        self.propagation_steps = config["propagation_steps"]

    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long
        )
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    def _item_embeddings_from_codes_and_semantic(self, input_ids: torch.Tensor):
        """
        input_ids: (batch, seq_len) item indices.
        Returns e_mix: (batch, seq_len, n_embd).
        """
        device = input_ids.device
        n_embd = self.config["n_embd"]
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # (B, S, L)
        input_tokens = self.item_id2tokens[input_ids]
        # (B, S, L, n_embd)
        v_all = self.gpt2.wte(input_tokens)

        # e_cf = sum_l W^(l) v^(l)
        e_cf_list = []
        for l in range(n_digit):
            e_cf_list.append(self.W_cf[l](v_all[:, :, l, :]))
        e_cf = torch.stack(e_cf_list, dim=0).sum(dim=0)  # (B, S, n_embd)

        # e_cf = v_all.mean(dim=-2)  # (B, S, n_embd)

        # e_sem = MLP(h^text)
        # h_text = self._item_text_emb[input_ids]  # (B, S, text_dim)
        # e_sem = self.mlp_sem(h_text)  # (B, S, n_embd)

        # # g = sigmoid(W_g [e_cf; e_sem])
        # concat = torch.cat([e_cf, e_sem], dim=-1)  # (B, S, 2*n_embd)
        # g = torch.sigmoid(self.W_gate(concat))  # (B, S, n_embd)

        # e_mix = g * e_cf + (1 - g) * e_sem
        e_mix = e_cf
        return e_mix

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.gpt2.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_embs = self._item_embeddings_from_codes_and_semantic(
            batch["input_ids"]
        )
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch["attention_mask"],
        )
        final_states = [
            self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head)
        ]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            assert "labels" in batch, "The batch must contain the labels."
            label_mask = batch["labels"].view(-1) != -100
            selected_states = final_states.view(
                -1, self.n_pred_head, self.config["n_embd"]
            )[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(
                selected_states, self.n_pred_head, dim=1
            )
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [
                torch.matmul(
                    selected_states[i].squeeze(dim=1), token_embs[i].T
                )
                / self.temperature
                for i in range(self.n_pred_head)
            ]
            token_labels = self.item_id2tokens[
                batch["labels"].view(-1)[label_mask]
            ]
            losses = [
                self.loss_fct(
                    token_logits[i],
                    token_labels[:, i] - i * self.config["codebook_size"] - 1,
                )
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def build_ii_sim_mat(self):
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)
        item_item_sim = torch.zeros(
            (n_items, n_items), device=self.gpt2.device, dtype=torch.float32
        )
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros(
                    (block_size_i, block_size_j),
                    device=self.gpt2.device,
                    dtype=torch.float32,
                )
                for k in range(n_digit):
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1
                    temp = token_sims_01[k].index_select(0, row_inds)
                    temp = temp.index_select(1, col_inds)
                    sum_block += temp
                avg_block = sum_block / n_digit
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block
        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()
        topk_nodes_sorted = torch.randint(
            1,
            self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device,
        )
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)
        for sid in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(
                batch_size, -1
            )
            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)
                scores = torch.gather(
                    input=token_logits[batch_id]
                    .unsqueeze(0)
                    .expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1),
                ).mean(dim=-1)
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)
        visited_counts = torch.FloatTensor(
            [[len(visited_nodes[batch_id])] for batch_id in range(batch_size)]
        )
        return (
            topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1),
            visited_counts,
        )

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch["seq_lens"] - 1)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, self.n_pred_head, self.config["n_embd"]),
        )
        states = F.normalize(states, dim=-1)
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [
            torch.matmul(states[:, 0, i, :], token_embs[i].T)
            / self.temperature
            for i in range(self.n_pred_head)
        ]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            return self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences,
            )
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(
                -1, self.dataset.n_items, -1
            ),
            dim=-1,
            index=(
                self.item_id2tokens[1:, :] - 1
            ).unsqueeze(0).expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        return preds.unsqueeze(-1)
