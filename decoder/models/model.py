

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from dataloader.dataset import AbstractDataset
from decoder.model import AbstractModel
from decoder.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class BaseModel(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(BaseModel, self).__init__(config, dataset, tokenizer)

        self.rqvae_config = self.config['RQ-VAE']
        self.codebook_size = self.rqvae_config['code_book_size']

        # Core change: directly reference the final lookup table from tokenizer
        self.item_id2tokens = self.tokenizer.item_id2tokens

        gpt2config = GPT2Config(vocab_size=tokenizer.vocab_size, **config)
        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        self.pred_heads = nn.Sequential(*[ResBlock(config['n_embd']) for _ in range(self.n_pred_head)])
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

    # The _map_item_tokens method has been removed because its functionality was moved to Tokenizer

    @property
    def n_parameters(self) -> str:
        # ... (this section remains unchanged)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (f'#Embedding parameters: {emb_params}\n'
                f'#Non-embedding parameters: {total_params - emb_params}\n'
                f'#Total trainable parameters: {total_params}\n')

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # ... (this section remains unchanged; it already handles input correctly)
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch['attention_mask'])
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states

        if return_loss:
            label_mask = batch['labels'].view(-1) != self.tokenizer.ignored_label
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states_chunks = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:self.tokenizer.eos_token]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs_chunks = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [torch.matmul(selected_states_chunks[i].squeeze(dim=1), token_embs_chunks[i].T) / self.temperature for i in range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [self.loss_fct(token_logits[i], token_labels[:, i] - (i * self.codebook_size) - 1) for i in range(self.n_pred_head)]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def generate(self, batch, n_return_sequences=1):
        """
        The generation function is overridden to return Codebook sequences of top-k items instead of their IDs.
        """
        outputs = self.forward(batch, return_loss=False)
        last_step_indices = (batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        states = outputs.final_states.gather(dim=1, index=last_step_indices)
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:self.tokenizer.eos_token]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs_chunks = torch.chunk(token_emb, self.n_pred_head, dim=0)

        logits = [torch.matmul(states[:, 0, i, :], token_embs_chunks[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)

        num_actual_items = self.dataset.n_items - 1
        item_codes_indices = self.item_id2tokens[1:self.dataset.n_items, :] - 1
        
        expanded_logits = token_logits.unsqueeze(1).expand(-1, num_actual_items, -1)
        expanded_indices = item_codes_indices.unsqueeze(0).expand(token_logits.shape[0], -1, -1)

        item_code_logits = torch.gather(input=expanded_logits, dim=2, index=expanded_indices)
        item_scores = item_code_logits.sum(dim=-1)
        
        # Get top-k item IDs (1-based)
        topk_item_ids = item_scores.topk(n_return_sequences, dim=-1).indices + 1
        
        # Core change: use top-k item IDs to fetch their corresponding codebook sequences from the lookup table
        predicted_codebooks = self.item_id2tokens[topk_item_ids]
        
        return predicted_codebooks