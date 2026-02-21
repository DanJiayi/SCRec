"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, latent_size, dropout=0.0, layer_norm_eps=1e-12
    ):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size, eps=layer_norm_eps),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)


class TIGER(T5ForConditionalGeneration):
    """
    A wrapper class for T5ForConditionalGeneration that adds extra functionality while preserving core T5 behavior.
    """

    def __init__(
        self,
        config: T5Config,
        n_semantic_codebook: int,
        max_items_per_seq: int,
        flag_use_learnable_text_embed: bool = False,
        flag_use_output_embedding: bool = False,
        embedding_head_dict: dict = None,
    ):

        self.flag_use_output_embedding = flag_use_output_embedding
        self.embedding_head_dict = embedding_head_dict

        super().__init__(config)

        if flag_use_learnable_text_embed:
            if embedding_head_dict["embed_proj_type"] == "mlp":
                self.emb_proj = MLP(
                    embedding_head_dict["text_embedding_dim"],
                    embedding_head_dict["hidden_sizes"][::-1],
                    config.d_model,
                    dropout=embedding_head_dict["embd_proj_in_dropout_rate"],
                    layer_norm_eps=config.layer_norm_epsilon,
                )
            elif embedding_head_dict["embed_proj_type"] == "linear":
                self.emb_proj = nn.Linear(
                    embedding_head_dict["text_embedding_dim"], config.d_model
                )
            else:
                raise ValueError(
                    f"Invalid embedding projection type: {embedding_head_dict['embed_proj_type']}"
                )
            self.input_embed_dropout = nn.Dropout(
                p=embedding_head_dict["embd_proj_dropout_rate"]
            )
            self.input_embed_layernorm = nn.LayerNorm(
                config.d_model, eps=config.layer_norm_epsilon
            )

        self.n_semantic_codebook = n_semantic_codebook
        self.semantic_pos = nn.Embedding(n_semantic_codebook + 1, config.d_model)
        self.pos_embedding = nn.Embedding(max_items_per_seq, config.d_model)

        if embedding_head_dict["use_new_init"]:
            # this is only applied for dense retrieval
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        initializer_range = factor
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        encoder_outputs = output.encoder_last_hidden_state
        # [batch_size, max_sequence_length, n_embd]

        self.predicted_embedding = None
        if self.flag_use_output_embedding:
            attention_mask = kwargs["attention_mask"]
            item_seq_len = attention_mask.sum(-1)
            predicted_embedding = self.gather_indexes(encoder_outputs, item_seq_len - 1)
            self.predicted_embedding = predicted_embedding

        return output
