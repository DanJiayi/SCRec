

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class BaseModel(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(BaseModel, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_tokens)
            dim=-1,
            index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
        ).mean(dim=-1)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        return preds.unsqueeze(-1)
