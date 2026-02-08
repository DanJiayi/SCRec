# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import numpy as np
import os
import json

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
#from .rqvae_tokenizer import RQVaeTokenizer


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


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # 添加文本嵌入加载
        self.text_embeddings = self._load_text_embeddings()
        
        # 性能优化：缓存设备转换，避免每次调用都转换
        self._text_embeddings_device_cache = {}
        
        print(f"[TEXT_MODAL] 🚀 Performance optimization enabled:")
        print(f"[TEXT_MODAL] 🚀   - Vectorized indexing (replaces double Python loops)")
        print(f"[TEXT_MODAL] 🚀   - Device caching (avoids repeated .to() calls)")
        print(f"[TEXT_MODAL] 🚀   - Expected performance improvement: 10x-100x faster")

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
        # per-digit linear projection (for code aggregation)
        # self.W_cf = nn.ModuleList([
        #     nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        #     for _ in range(self.tokenizer.n_digit)
        # ])

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        # 添加文本模态融合层
        text_dim = self.text_embeddings.shape[1]
        print(f"[TEXT_MODAL] 🔧 Text embedding dimension: {text_dim}")
        print(f"[TEXT_MODAL] 🔧 GPT2 embedding dimension: {config['n_embd']}")
        
        # 修改原因：使用拼接融合避免过度正则化，保持模型性能
        # 原代码：简单的线性融合层
        # self.modality_fusion = nn.Linear(
        #     config['n_embd'] + text_dim, 
        #     config['n_embd']
        # )
        
        # 新代码：带正则化的多层融合网络（已注释掉，效果下降）
        # self.modality_fusion = nn.Sequential(
        #     nn.Dropout(0.1),  # 输入dropout防止过拟合
        #     nn.Linear(config['n_embd'] + text_dim, config['n_embd'] * 2),  # 扩展维度
        #     nn.ReLU(),  # 非线性激活
        #     nn.Dropout(0.1),  # 中间dropout
        #     nn.Linear(config['n_embd'] * 2, config['n_embd']),  # 压缩回目标维度
        #     nn.LayerNorm(config['n_embd']),  # 层归一化稳定训练
        #     nn.Dropout(0.05)  # 输出dropout
        # )
        
        # 最新代码：使用简单的线性融合层，避免过度正则化
        # 直接拼接会导致维度不匹配，GPT-2期望448维输入
        # 所以仍然需要一个线性层来调整维度，但保持简单
        self.modality_fusion = nn.Linear(
            config['n_embd']+ text_dim,  # 输入：448 + 1280 = 1728
            config['n_embd']               # 输出：448
        )

        self.text_mlp = nn.Linear(text_dim, config['n_embd'])
        # gate network: g = sigmoid(W_g [e_cf ; e_sem])
        self.gate = nn.Linear(config['n_embd'] * 2, config['n_embd'])


    def _load_text_embeddings(self) -> torch.Tensor:
        """加载文本嵌入，支持动态维度"""
        try:
            category = self.config.get('category', 'Beauty')
            # 修改原因：使用配置中的cache_dir，支持动态路径配置
            # 原代码：硬编码绝对路径，导致权限问题
            # 新代码：从配置中获取cache_dir，如果没有则使用默认值
            cache_dir = self.config.get('cache_dir', 'cache')
            processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
            
            # 支持动态维度：从配置中获取目标维度，如果没有则使用默认值
            target_dim = self.config.get('text_embedding_dim', None)
            
            # 优先尝试加载指定维度的PCA嵌入文件
            if target_dim:
                # 修改原因：所有PCA文件都从数据集目录中读取，确保数据匹配
                # 原代码：从根目录读取通用PCA文件
                # 新代码：只从数据集目录读取，避免数量不匹配问题
                dataset_pca_file = os.path.join(processed_dir, f'final_pca_embeddings_{target_dim}d.npy')
                
                # 只尝试数据集目录下的文件
                if os.path.exists(dataset_pca_file):
                    embeddings = np.load(dataset_pca_file)
                    print(f"[TEXT_MODAL] ✅ Loaded {target_dim}D PCA text embeddings from dataset dir: {embeddings.shape}")
                    
                    # 检查嵌入数量是否与数据集商品数量匹配
                    if embeddings.shape[0] != self.dataset.n_items:
                        print(f"[TEXT_MODAL] ⚠️ Dataset PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                        raise ValueError(f"[TEXT_MODAL] ❌ Dataset PCA embeddings count mismatch: {embeddings.shape[0]} vs {self.dataset.n_items}")
                    
                    return torch.from_numpy(embeddings).float()
                else:
                    print(f"[TEXT_MODAL] ⚠️ Specified {target_dim}D PCA file not found in dataset dir: {dataset_pca_file}")
                    print(f"[TEXT_MODAL] 🔄 Falling back to default PCA file...")
            
            # 尝试加载默认的PCA处理后的文本嵌入
            pca_emb_file = os.path.join(processed_dir, 'final_pca_embeddings.npy')
            if os.path.exists(pca_emb_file):
                embeddings = np.load(pca_emb_file)
                print(f"[TEXT_MODAL] ✅ Loaded default PCA text embeddings: {embeddings.shape}")
                
                # 检查嵌入数量是否与数据集商品数量匹配
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[TEXT_MODAL] ⚠️ PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    raise ValueError(f"[TEXT_MODAL] ❌ PCA embeddings count mismatch: {embeddings.shape[0]} vs {self.dataset.n_items}")
                
                return torch.from_numpy(embeddings).float()
            
            # 如果没有PCA文件，尝试加载原始文本嵌入
            sent_emb_file = os.path.join(processed_dir, 'text-embedding-3-large.sent_emb')
            if os.path.exists(sent_emb_file):
                # 使用fromfile加载二进制格式的embedding文件
                # text-embedding-3-large模型的维度是3072
                embeddings = np.fromfile(sent_emb_file, dtype=np.float32).reshape(-1, 3072)
                print(f"[TEXT_MODAL] ⚠️ Loaded raw text embeddings: {embeddings.shape}")
                
                # 检查嵌入数量是否与数据集商品数量匹配
                # 注意：数据集可能包含特殊token如[PAD]，嵌入文件只包含真实商品
                expected_embeddings = self.dataset.n_items
                if embeddings.shape[0] != expected_embeddings:
                    print(f"[TEXT_MODAL] ⚠️ Raw text embeddings count ({embeddings.shape[0]}) doesn't match expected ({expected_embeddings})")
                    
                    # 如果嵌入数量更多，智能截取前N个
                    if embeddings.shape[0] > expected_embeddings:
                        print(f"[TEXT_MODAL] 🔄 Truncating embeddings to match expected size: {expected_embeddings}")
                        embeddings = embeddings[:expected_embeddings]
                        print(f"[TEXT_MODAL] ✅ Truncated embeddings shape: {embeddings.shape}")
                        return torch.from_numpy(embeddings).float()
                    else:
                        # 如果嵌入数量少，检查是否是特殊token的问题
                        if embeddings.shape[0] == expected_embeddings - 1:
                            print(f"[TEXT_MODAL] 🔍 Detected potential special token issue")
                            print(f"[TEXT_MODAL] 🔍 Embeddings: {embeddings.shape[0]}, Expected: {expected_embeddings}")
                            print(f"[TEXT_MODAL] 🔍 This usually means the dataset has a [PAD] token at index 0")
                            
                            # 为[PAD] token创建一个占位嵌入（全零）
                            pad_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
                            embeddings = np.vstack([pad_embedding, embeddings])
                            print(f"[TEXT_MODAL] ✅ Added [PAD] embedding, final shape: {embeddings.shape}")
                            return torch.from_numpy(embeddings).float()
                        else:
                            raise ValueError(f"[TEXT_MODAL] ❌ Text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({expected_embeddings}) and cannot be automatically fixed")
                
                return torch.from_numpy(embeddings).float()
            
            # 如果没有，直接报错
            raise FileNotFoundError("[TEXT_MODAL] ❌ No text embeddings found. Please ensure text embeddings are available.")
            
        except Exception as e:
            print(f"[TEXT_MODAL] ❌ Error loading text embeddings: {e}")
            raise RuntimeError(f"[TEXT_MODAL] ❌ Failed to load text embeddings: {e}")

    def _fuse_text_modality(self, id_embeddings, batch):
        """融合ID和文本模态 (性能优化版本)"""
        batch_size = id_embeddings.shape[0]
        seq_len = id_embeddings.shape[1]
        device = id_embeddings.device
        
        # 获取文本嵌入维度
        text_dim = self.text_embeddings.shape[1]
        
        # 获取批次中的商品ID
        item_ids = batch['input_ids']  # shape: (batch_size, seq_len)
        
        # --- 性能瓶颈优化：向量化索引替代双重循环 ---
        # 原代码：双重Python循环，性能极差
        # text_emb = torch.zeros(batch_size, seq_len, text_dim, device=device)
        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         item_id = item_ids[b, s].item()
        #         if 0 <= item_id < self.dataset.n_items:
        #             text_emb[b, s] = self.text_embeddings[item_id]
        
        # 新代码：向量化索引，一次性完成所有查找
        # 使用设备缓存，避免重复的设备转换
        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            self._text_embeddings_device_cache[device_key] = self.text_embeddings.to(device)
        
        text_embeddings_device = self._text_embeddings_device_cache[device_key]
        
        # 直接使用item_ids作为索引，PyTorch自动处理所有批次和序列位置
        # 这行代码替代了整个双重循环，性能提升数量级
        text_emb = text_embeddings_device[item_ids]  # shape: (batch_size, seq_len, text_dim)
        
        # 统计有效嵌入的数量（用于调试，不影响性能）
        # 注意：item_ids中0通常是[PAD] token，需要特殊处理
        valid_mask = (item_ids > 0) & (item_ids < self.dataset.n_items)
        valid_embeddings = valid_mask.sum().item()
        total_positions = batch_size * seq_len
        
        # 性能监控：检查是否有无效的item_id（超出范围）
        invalid_mask = (item_ids < 0) | (item_ids >= self.dataset.n_items)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum().item()
            if not hasattr(self, '_invalid_id_warning_logged'):
                print(f"[TEXT_MODAL] ⚠️ Warning: Found {invalid_count} invalid item IDs (out of range)")
                print(f"[TEXT_MODAL] ⚠️ Item ID range: 0 to {self.dataset.n_items - 1}")
                self._invalid_id_warning_logged = True
        
        # 打印融合统计信息（只在第一个batch时打印）
        if not hasattr(self, '_first_batch_logged'):
            print(f"[TEXT_MODAL] 🔄 Fusion stats: {valid_embeddings}/{total_positions} valid text embeddings")
            print(f"[TEXT_MODAL] 🔄 Text embedding sample: {text_emb[0, 0, :5]}")
            print(f"[TEXT_MODAL] 🔄 Text embedding norm: {torch.norm(text_emb[0, 0]):.4f}")
            print(f"[TEXT_MODAL] 🚀 Using vectorized indexing for performance optimization")
            self._first_batch_logged = True
        
        # 拼接ID和文本模态
        # fused_embeddings = torch.cat([id_embeddings, text_emb], dim=-1)
        
        # 通过融合层
        e_sem = self.text_mlp(text_emb)          # (B,S,d)
        # gate input: [e_cf ; e_sem]
        gate_inp = torch.cat([id_embeddings, e_sem], dim=-1)
        # g = sigmoid(W_g [...])
        g = torch.sigmoid(self.gate(gate_inp))  # (B,S,d)
        # e_mix = g ⊙ e_cf + (1-g) ⊙ e_sem
        fused_embeddings = g * id_embeddings + (1 - g) * e_sem
        # fused_embeddings = self.modality_fusion(fused_embeddings)
        return fused_embeddings
    
    def benchmark_fusion_performance(self, batch_size=32, seq_len=50, num_runs=100):
        """性能基准测试函数，用于验证优化效果"""
        import time
        
        print(f"[BENCHMARK] 🚀 Starting fusion performance benchmark...")
        print(f"[BENCHMARK] 📊 Batch size: {batch_size}, Sequence length: {seq_len}, Runs: {num_runs}")
        
        # 创建测试数据
        device = next(self.parameters()).device
        test_batch = {
            'input_ids': torch.randint(0, min(1000, self.dataset.n_items), (batch_size, seq_len), device=device)
        }
        test_id_embeddings = torch.randn(batch_size, seq_len, self.config['n_embd'], device=device)
        
        # 预热GPU
        print(f"[BENCHMARK] 🔥 Warming up GPU...")
        for _ in range(10):
            _ = self._fuse_text_modality(test_id_embeddings, test_batch)
        
        # 性能测试
        print(f"[BENCHMARK] ⏱️ Running performance test...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = self._fuse_text_modality(test_id_embeddings, test_batch)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        print(f"[BENCHMARK] ✅ Performance test completed!")
        print(f"[BENCHMARK] 📈 Total time: {total_time:.4f}s")
        print(f"[BENCHMARK] 📈 Average time per run: {avg_time:.6f}s")
        print(f"[BENCHMARK] 📈 Throughput: {throughput:.2f} runs/second")
        print(f"[BENCHMARK] 📈 Estimated time for 1000 batches: {avg_time * 1000:.4f}s")
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': throughput
        }

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.
        Supports both FAISS and RQVae codebooks.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        # 修改原因：支持RQVae码本和FAISS码本的动态切换
        if self.config.get('use_rqvae_codebook', False):
            print(f"[RQVae] 🔄 Using RQVae codebook for item token mapping")
            # 使用RQVae码本
            # 修改原因：修复索引越界问题，item_id应该从0开始，最大为n_items-1
            item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
            
            valid_items = 0
            skipped_items = 0
            skipped_details = []  # 记录被跳过的item详细信息
            
            print(f"[RQVae] 📊 开始映射码本中的 {len(self.tokenizer.item2tokens)} 个item到数据集...")
            print(f"[RQVae] 📊 数据集总item数量: {self.dataset.n_items}")
            print(f"[RQVae] 📊 数据集item ID范围: 0 到 {self.dataset.n_items - 1}")
            
            for item in self.tokenizer.item2tokens:
                if item in self.dataset.item2id:
                    item_id = self.dataset.item2id[item]
                    # 确保item_id在有效范围内
                    if item_id < self.dataset.n_items:
                        item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
                        valid_items += 1
                    else:
                        skip_reason = f"Item ID {item_id} exceeds dataset size {self.dataset.n_items}"
                        print(f"[WARNING] {skip_reason}")
                        skipped_details.append(f"Item '{item}' -> ID {item_id}: {skip_reason}")
                        skipped_items += 1
                else:
                    skip_reason = f"Item '{item}' not found in dataset item2id mapping"
                    print(f"[WARNING] {skip_reason}")
                    skipped_details.append(f"Item '{item}': {skip_reason}")
                    skipped_items += 1
            
            # 输出详细的跳过信息
            if skipped_items > 0:
                print(f"\n[RQVae] ⚠️ 详细跳过信息 ({skipped_items} 个item):")
                for detail in skipped_details:
                    print(f"  - {detail}")
                print()
            
            print(f"[RQVae] ✅ 映射完成: 成功 {valid_items} 个, 跳过 {skipped_items} 个")
            print(f"[RQVae] 📊 最终张量形状: {item_id2tokens.shape}")
            
            return item_id2tokens
        else:
            print(f"[FAISS] 🔄 Using FAISS codebook for item token mapping")
            # 使用FAISS码本（原有逻辑）
            # 修改原因：修复索引越界问题，item_id应该从0开始，最大为n_items-1
            item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
            
            valid_items = 0
            skipped_items = 0
            skipped_details = []  # 记录被跳过的item详细信息
            
            print(f"[FAISS] 📊 开始映射码本中的 {len(self.tokenizer.item2tokens)} 个item到数据集...")
            print(f"[FAISS] 📊 数据集总item数量: {self.dataset.n_items}")
            print(f"[FAISS] 📊 数据集item ID范围: 0 到 {self.dataset.n_items - 1}")
            
            for item in self.tokenizer.item2tokens:
                if item in self.dataset.item2id:
                    item_id = self.dataset.item2id[item]
                    # 确保item_id在有效范围内
                    if item_id < self.dataset.n_items:
                        item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
                        valid_items += 1
                    else:
                        skip_reason = f"Item ID {item_id} exceeds dataset size {self.dataset.n_items}"
                        print(f"[WARNING] {skip_reason}")
                        skipped_details.append(f"Item '{item}' -> ID {item_id}: {skip_reason}")
                        skipped_items += 1
                else:
                    skip_reason = f"Item '{item}' not found in dataset item2id mapping"
                    print(f"[WARNING] {skip_reason}")
                    skipped_details.append(f"Item '{item}': {skip_reason}")
                    skipped_items += 1
            
            # 输出详细的跳过信息
            if skipped_items > 0:
                print(f"\n[FAISS] ⚠️ 详细跳过信息 ({skipped_items} 个item):")
                for detail in skipped_details:
                    print(f"  - {detail}")
                print()
            
            print(f"[FAISS] ✅ 映射完成: 成功 {valid_items} 个, 跳过 {skipped_items} 个")
            print(f"[FAISS] 📊 最终张量形状: {item_id2tokens.shape}")
            
            return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # 获取ID模态嵌入
        input_tokens = self.item_id2tokens[batch['input_ids']]

        id_embeddings = self.gpt2.wte(input_tokens).mean(dim=-2)
        # v_all = self.gpt2.wte(input_tokens)  # (B, S, L, d)
        # e_cf_list = []
        # for l in range(self.tokenizer.n_digit):
        #     e_cf_list.append(self.W_cf[l](v_all[:, :, l, :]))
        # id_embeddings = torch.stack(e_cf_list, dim=0).sum(dim=0)  # (B, S, d)

        self.code_weights = nn.Parameter(torch.ones(self.tokenizer.n_digit))
        v_all = self.gpt2.wte(input_tokens)   # (B, S, L, d)
        # weights = torch.softmax(self.code_weights.to(v_all.device), dim=0)
        tau = self.config.get("code_weight_tau", 1.0)
        weights = torch.softmax(self.code_weights.to(v_all.device) / tau, dim=0)  # (L,) -> (1, 1, L) -> (B, S, L)

        id_embeddings = (v_all * weights[None, None, :, None]).sum(dim=-2)
        
        # 融合ID和文本模态
        fused_embeddings = self._fuse_text_modality(id_embeddings, batch)

        # 不使用其他模态
        #fused_embeddings = id_embeddings
        
        # 通过GPT-2
        outputs = self.gpt2(
            inputs_embeds=fused_embeddings,
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

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end]  # sub-block for items j

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / n_digit

                # Write back into the final item_item_sim
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

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

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

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_tokens)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)
