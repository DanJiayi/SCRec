# genrec/evaluator.py (最终修正版 - 修复设备不匹配错误)

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
        比较预测的 Codebook 序列和真实的 Codebook 序列。
        所有计算都在原始设备（例如 GPU）上进行。

        Args:
            preds (torch.Tensor): 预测的 Codebook 序列，形状为 [B, K, n_codebooks]，在 GPU 上。
            labels (torch.Tensor): 真实的 item ID，形状为 [B, 1] 或 [B]，在 GPU 上。
        """
        # --- 核心修改 ---
        # 移除 .cpu() 调用，让张量保留在原始设备 (cuda:0) 上
        preds = preds.detach()
        labels = labels.detach()
        # --- 修改结束 ---

        # 从 tokenizer 获取真实的 Codebook 序列。
        # self.tokenizer.item_id2tokens 已经在 GPU 上，所以 ground_truth_codebooks 也在 GPU 上。
        ground_truth_codebooks = self.tokenizer.item_id2tokens[labels.squeeze()]

        if ground_truth_codebooks.dim() == 1:
            ground_truth_codebooks = ground_truth_codebooks.unsqueeze(0)
            
        expanded_gt = ground_truth_codebooks.unsqueeze(1).expand_as(preds)

        # 现在两个张量都在同一个 GPU 设备上，可以正常比较
        pos_index = torch.all(preds == expanded_gt, dim=-1)
        
        return pos_index

    def recall_at_k(self, pos_index: torch.Tensor, k: int) -> torch.Tensor:
        return (pos_index[:, :k].sum(dim=1) > 0).float()

    def ndcg_at_k(self, pos_index: torch.Tensor, k: int) -> torch.Tensor:
        pos_index_k = pos_index[:, :k]
        # 确保计算在 pos_index 所在的设备上进行
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
        
        # 将最终的 metric tensor 移到 CPU，以便进行后续的聚合和打印
        for key, value in results.items():
             results[key] = value.cpu()
             
        return results