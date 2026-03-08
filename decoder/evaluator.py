# genrec/evaluator.py (final revised version - fixes device mismatch errors)

import torch

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compare predicted Codebook sequences with ground-truth Codebook sequences.
        All computations run on the original device (e.g., GPU).

        Args:
            preds (torch.Tensor): Predicted Codebook sequences, shape [B, K, n_codebooks], on GPU.
            labels (torch.Tensor): Ground-truth item IDs, shape [B, 1] or [B], on GPU.
        """
        # --- Core change ---
        # Remove .cpu() calls so tensors stay on the original device (cuda:0)
        preds = preds.detach()
        labels = labels.detach()
        # --- End of change ---

        # Get ground-truth Codebook sequences from tokenizer.
        # self.tokenizer.item_id2tokens is already on GPU, so ground_truth_codebooks is also on GPU.
        ground_truth_codebooks = self.tokenizer.item_id2tokens[labels.squeeze()]

        if ground_truth_codebooks.dim() == 1:
            ground_truth_codebooks = ground_truth_codebooks.unsqueeze(0)
            
        expanded_gt = ground_truth_codebooks.unsqueeze(1).expand_as(preds)

        # Now both tensors are on the same GPU device and can be compared directly
        pos_index = torch.all(preds == expanded_gt, dim=-1)
        
        return pos_index

    def recall_at_k(self, pos_index: torch.Tensor, k: int) -> torch.Tensor:
        return (pos_index[:, :k].sum(dim=1) > 0).float()

    def ndcg_at_k(self, pos_index: torch.Tensor, k: int) -> torch.Tensor:
        pos_index_k = pos_index[:, :k]
        # Ensure computation is done on the same device as pos_index
        ranks = (torch.nonzero(pos_index_k, as_tuple=True)[1].float() + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        
        ndcg = torch.zeros(pos_index_k.shape[0], device=pos_index.device)
        hit_indices = torch.nonzero(pos_index_k, as_tuple=True)[0]
        
        ndcg.scatter_add_(0, hit_indices, dcg)
        return ndcg

    def calculate_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> dict:
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        
        # Move final metric tensors to CPU for later aggregation and printing
        for key, value in results.items():
             results[key] = value.cpu()
             
        return results