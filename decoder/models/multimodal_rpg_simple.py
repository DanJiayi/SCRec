#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import numpy as np
import os
import json

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


class MultimodalRPGSimple(AbstractModel):
    """
    简化版三模态RPG模型：ID + 文本 + 图片
    使用简化的融合策略，避免复杂的维度处理
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(MultimodalRPGSimple, self).__init__(config, dataset, tokenizer)

        self.rqvae_config = self.config['RQ-VAE']
        self.codebook_size = self.rqvae_config['code_book_size']

        # 核心改动：直接从 tokenizer 引用最终的查找表
        self.item_id2tokens = self.tokenizer.item_id2tokens

        # 只加载文本嵌入（去掉图片模态）
        # self.image_embeddings = self._load_image_embeddings()  # 注释掉图片嵌入加载
        self.text_embeddings = self._load_text_embeddings()

        gpt2config = GPT2Config(vocab_size=tokenizer.vocab_size, **config)
        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        self.pred_heads = nn.Sequential(*[ResBlock(config['n_embd']) for _ in range(self.n_pred_head)])
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        
        # 简化的多模态融合层 - 只融合ID和文本模态（去掉图片模态）
        self.modality_fusion = nn.Linear(
            config['n_embd'] + self.text_embeddings.shape[1], 
            config['n_embd']
        )
        
        # 注释掉模态权重（可学习）- 改为直接拼接融合
        # self.modality_weights = nn.Parameter(torch.ones(3))  # ID, 文本, 图片

    def _load_image_embeddings(self) -> torch.Tensor:
        """加载图片嵌入 - 优先加载512维的图片embedding"""
        try:
            category = self.config.get('category', 'Beauty')
            cache_dir = self.config.get('cache_dir', 'cache')
            img_emb_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed', 'image_embeddings')
            
            # 优先尝试加载256维的CLIP嵌入（与之前训练保持一致）
            clip_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32.npy')
            clip_mapping_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32_mapping.json')
            if os.path.exists(clip_file) and os.path.exists(clip_mapping_file):
                embeddings = np.load(clip_file)
                with open(clip_mapping_file, 'r') as f:
                    import json
                    mapping = json.load(f)
                # 智能检测并修复CLIP映射格式
                # 从您提供的内容看，映射格式是 {"索引": "商品ID"}，如 "18025": "B000EX6ILI"
                # 我们需要创建一个从商品ID到索引的映射，用于在训练时查找
                sample_key = list(mapping.keys())[0]
                sample_value = mapping[sample_key]
                
                # 检查是否为数字字符串
                def is_digit_string(s):
                    return isinstance(s, str) and s.isdigit()
                
                if is_digit_string(sample_key) and not is_digit_string(sample_value):
                    # 格式是 {"索引": "商品ID"}，需要转换为 {"商品ID": "索引"} 用于查找
                    print(f"[MULTIMODAL] 🔄 Detected format: {{'索引': '商品ID'}}, converting to {{'商品ID': '索引'}} for lookup")
                    print(f"[MULTIMODAL] 🔄 Sample: key='{sample_key}' (index), value='{sample_value}' (item_id)")
                    # 创建反向映射：商品ID -> 索引
                    reversed_mapping = {v: int(k) for k, v in mapping.items()}
                    self.clip_mapping = reversed_mapping
                    print(f"[MULTIMODAL] ✅ Created reverse mapping with {len(reversed_mapping)} items")
                elif not is_digit_string(sample_key) and is_digit_string(sample_value):
                    # 格式已经是 {"商品ID": "索引"}，直接使用
                    print(f"[MULTIMODAL] ✅ Detected format: {{'商品ID': '索引'}}, using directly")
                    print(f"[MULTIMODAL] ✅ Sample: key='{sample_key}' (item_id), value='{sample_value}' (index)")
                    self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                else:
                    # 无法确定格式，使用原始映射并打印警告
                    print(f"[MULTIMODAL] ⚠️ Unable to determine mapping format, using original mapping")
                    print(f"[MULTIMODAL] ⚠️ Sample key: {sample_key} (type: {type(sample_key)})")
                    print(f"[MULTIMODAL] ⚠️ Sample value: {sample_value} (type: {type(sample_value)})")
                    # 尝试智能修复：如果key看起来像商品ID（以B开头），value看起来像数字
                    if isinstance(sample_key, str) and sample_key.startswith('B') and isinstance(sample_value, str) and sample_value.isdigit():
                        print(f"[MULTIMODAL] 🔄 Attempting smart fix: treating as {{'商品ID': '索引'}} format")
                        self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                    else:
                        self.clip_mapping = mapping
                
                print(f"[MULTIMODAL] ✅ Loaded 256-dim CLIP image embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] ✅ CLIP mapping contains {len(self.clip_mapping)} items")
                print(f"[MULTIMODAL] ✅ Image embedding sample: {embeddings[0][:5]}")  # 显示前5个值
                return torch.from_numpy(embeddings).float()
            
            # 如果256维文件不存在，尝试加载512维的CLIP嵌入（作为备选）
            # 原实现优先512维，因"需要与之前训练保持一致(256维)"而改为优先256维
            clip_512_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32-512d.npy')
            clip_512_mapping_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32-512d_mapping.json')
            if os.path.exists(clip_512_file) and os.path.exists(clip_512_mapping_file):
                embeddings = np.load(clip_512_file)
                with open(clip_512_mapping_file, 'r') as f:
                    import json
                    mapping = json.load(f)
                # 智能检测并修复CLIP映射格式
                # 从您提供的内容看，映射格式是 {"索引": "商品ID"}，如 "18025": "B000EX6ILI"
                # 我们需要创建一个从商品ID到索引的映射，用于在训练时查找
                sample_key = list(mapping.keys())[0]
                sample_value = mapping[sample_key]
                
                # 检查是否为数字字符串
                def is_digit_string_512(s):
                    return isinstance(s, str) and s.isdigit()
                
                if is_digit_string_512(sample_key) and not is_digit_string_512(sample_value):
                    # 格式是 {"索引": "商品ID"}，需要转换为 {"商品ID": "索引"} 用于查找
                    print(f"[MULTIMODAL] 🔄 Detected format: {{'索引': '商品ID'}}, converting to {{'商品ID': '索引'}} for lookup")
                    print(f"[MULTIMODAL] 🔄 Sample: key='{sample_key}' (index), value='{sample_value}' (item_id)")
                    # 创建反向映射：商品ID -> 索引
                    reversed_mapping = {v: int(k) for k, v in mapping.items()}
                    self.clip_mapping = reversed_mapping
                    print(f"[MULTIMODAL] ✅ Created reverse mapping with {len(reversed_mapping)} items")
                elif not is_digit_string_512(sample_key) and is_digit_string_512(sample_value):
                    # 格式已经是 {"商品ID": "索引"}，直接使用
                    print(f"[MULTIMODAL] ✅ Detected format: {{'商品ID': '索引'}}, using directly")
                    print(f"[MULTIMODAL] ✅ Sample: key='{sample_key}' (item_id), value='{sample_value}' (index)")
                    self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                else:
                    # 无法确定格式，使用原始映射并打印警告
                    print(f"[MULTIMODAL] ⚠️ Unable to determine mapping format, using original mapping")
                    print(f"[MULTIMODAL] ⚠️ Sample key: {sample_key} (type: {type(sample_key)})")
                    print(f"[MULTIMODAL] ⚠️ Sample value: {sample_value} (type: {type(sample_value)})")
                    # 尝试智能修复：如果key看起来像商品ID（以B开头），value看起来像数字
                    if isinstance(sample_key, str) and sample_key.startswith('B') and isinstance(sample_value, str) and sample_value.isdigit():
                        print(f"[MULTIMODAL] 🔄 Attempting smart fix: treating as {{'商品ID': '索引'}} format")
                        self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                    else:
                        self.clip_mapping = mapping
                
                print(f"[MULTIMODAL] ⚠️ Loaded 512-dim CLIP image embeddings (fallback): {embeddings.shape}")
                print(f"[MULTIMODAL] ⚠️ CLIP mapping contains {len(self.clip_mapping)} items")
                return torch.from_numpy(embeddings).float()
            
            # 尝试加载随机向量
            random_file = os.path.join(img_emb_dir, 'image_embeddings_random.npy')
            if os.path.exists(random_file):
                embeddings = np.load(random_file)
                print(f"[MULTIMODAL] ⚠️ Loaded random image embeddings: {embeddings.shape}")
                return torch.from_numpy(embeddings).float()
            
            # 如果都没有，生成随机向量
            print("[MULTIMODAL] ❌ No image embeddings found, generating random vectors")
            n_items = self.dataset.n_items
            img_dim = self.config.get('img_emb_dim', 512)
            random_embeddings = np.random.normal(0, 1, (n_items, img_dim))
            return torch.from_numpy(random_embeddings).float()
            
        except Exception as e:
            print(f"[MULTIMODAL] ❌ Error loading image embeddings: {e}")
            print("[MULTIMODAL] ❌ Using random vectors as fallback")
            n_items = self.dataset.n_items
            img_dim = self.config.get('img_emb_dim', 512)
            random_embeddings = np.random.normal(0, 1, (n_items, img_dim))
            return torch.from_numpy(random_embeddings).float()

    def _load_text_embeddings(self) -> torch.Tensor:
        """加载文本嵌入"""
        try:
            category = self.config.get('category', 'Beauty')
            cache_dir = self.config.get('cache_dir', 'cache')
            processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
            
            # 优先尝试加载PCA处理后的文本嵌入（正确的形状）
            pca_emb_file = os.path.join(processed_dir, 'final_pca_embeddings.npy')
            if os.path.exists(pca_emb_file):
                embeddings = np.load(pca_emb_file)
                print(f"[MULTIMODAL] ✅ Loaded PCA-processed text embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] ✅ Text embedding sample: {embeddings[0][:5]}")  # 显示前5个值
                
                # 检查嵌入数量是否与数据集商品数量匹配
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[MULTIMODAL] ⚠️ PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    print("[MULTIMODAL] ⚠️ Generating random text embeddings for current dataset")
                    n_items = self.dataset.n_items
                    text_dim = self.config.get('sent_emb_pca', 512)
                    random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
                    return torch.from_numpy(random_embeddings).float()
                
                return torch.from_numpy(embeddings).float()
            
            # 如果没有PCA文件，尝试加载原始文本嵌入
            sent_emb_file = os.path.join(processed_dir, 'text-embedding-3-large.sent_emb')
            if os.path.exists(sent_emb_file):
                # 使用fromfile加载二进制格式的embedding文件
                embeddings = np.fromfile(sent_emb_file, dtype=np.float32).reshape(-1, 512)
                print(f"[MULTIMODAL] ⚠️ Loaded raw text embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] ⚠️ Text embedding sample: {embeddings[0][:5]}")  # 显示前5个值
                
                # 检查嵌入数量是否与数据集商品数量匹配
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[MULTIMODAL] ⚠️ Raw text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    print("[MULTIMODAL] ⚠️ Generating random text embeddings for current dataset")
                    n_items = self.dataset.n_items
                    text_dim = self.config.get('sent_emb_pca', 512)
                    random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
                    return torch.from_numpy(random_embeddings).float()
                
                return torch.from_numpy(embeddings).float()
            
            # 如果没有，生成随机向量
            print("[MULTIMODAL] ❌ No text embeddings found, generating random vectors")
            n_items = self.dataset.n_items
            text_dim = self.config.get('sent_emb_pca', 512)
            random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
            return torch.from_numpy(random_embeddings).float()
            
        except Exception as e:
            print(f"[MULTIMODAL] ❌ Error loading text embeddings: {e}")
            print("[MULTIMODAL] ❌ Using random vectors as fallback")
            n_items = self.dataset.n_items
            text_dim = self.config.get('sent_emb_pca', 512)
            random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
            return torch.from_numpy(random_embeddings).float()

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (f'#Embedding parameters: {emb_params}\n'
                f'#Non-embedding parameters: {total_params - emb_params}\n'
                f'#Total trainable parameters: {total_params}\n')

    def _simple_fuse_modalities(self, id_embeddings, batch):
        """真正的多模态融合，使用真实的图片和文本嵌入"""
        
        # 如果使用多模态码本，直接返回ID嵌入，不进行额外融合
        # if self.config.get('use_multimodal_codebook', False):
        #     return id_embeddings
        batch_size = id_embeddings.shape[0]
        seq_len = id_embeddings.shape[1]
        device = id_embeddings.device
        
        # 获取嵌入维度
        text_dim = self.text_embeddings.shape[1]
        # image_dim = self.image_embeddings.shape[1]  # 注释掉图片维度
        
        # 获取批次中的商品ID
        item_ids = batch['input_ids']  # shape: (batch_size, seq_len)
        
        # 从嵌入表中获取对应的文本嵌入
        # 注意：item_ids是1-based，但嵌入表是0-based
        text_emb = torch.zeros(batch_size, seq_len, text_dim, device=device)
        # image_emb = torch.zeros(batch_size, seq_len, image_dim, device=device)  # 注释掉图片嵌入
        
        # 统计有效嵌入的数量
        valid_embeddings = 0
        total_positions = batch_size * seq_len
        
        # 调试信息：检查嵌入表大小
        if not hasattr(self, '_first_batch_logged'):
            print(f"[MULTIMODAL] 🔍 Debug info:")
            print(f"[MULTIMODAL] 🔍 Text embeddings shape: {self.text_embeddings.shape}")
            # print(f"[MULTIMODAL] 🔍 Image embeddings shape: {self.image_embeddings.shape}")  # 注释掉图片嵌入信息
            print(f"[MULTIMODAL] 🔍 Dataset n_items: {self.dataset.n_items}")
            print(f"[MULTIMODAL] 🔍 Item IDs range: {item_ids.min().item()} to {item_ids.max().item()}")
        
        # 为每个批次中的商品获取对应的嵌入
        for b in range(batch_size):
            for s in range(seq_len):
                item_id = item_ids[b, s].item()
                # 修复索引问题：item_id应该是1到n_items，嵌入表索引是0到n_items-1
                if item_id > 0 and item_id <= self.dataset.n_items:
                    # 获取文本嵌入
                    text_emb[b, s] = self.text_embeddings[item_id - 1]  # 转换为0-based索引
                    # 注释掉图片嵌入处理（去掉图片模态）
                    # 获取图片嵌入 - 使用CLIP映射
                    # 现在clip_mapping是 {"商品ID": "索引"} 格式，我们需要根据item_id查找对应的商品ID
                    # 但是item_id是数字，我们需要找到对应的商品ID字符串
                    # if hasattr(self, 'clip_mapping') and self.clip_mapping:
                    #     # 由于我们无法直接从item_id找到商品ID，我们暂时使用默认的索引方式
                    #     # 这是原始设计的限制，我们暂时使用item_id-1作为索引
                    #     if item_id - 1 < self.image_embeddings.shape[0]:
                    #         image_emb[b, s] = self.image_embeddings[item_id - 1]
                    #         valid_embeddings += 1
                    #     else:
                    #         print(f"[MULTIMODAL] ⚠️ Item ID {item_id} out of range for image embeddings shape {self.image_embeddings.shape}")
                    # else:
                    #     # 如果没有CLIP映射，使用默认索引（可能不准确）
                    #     if item_id - 1 < self.image_embeddings.shape[0]:
                    #         image_emb[b, s] = self.image_embeddings[item_id - 1]
                    #         valid_embeddings += 1
        
        # 打印融合统计信息（只在第一个batch时打印）
        if not hasattr(self, '_first_batch_logged'):
            print(f"[MULTIMODAL] 🔄 Fusion stats: {valid_embeddings}/{total_positions} valid embeddings")
            print(f"[MULTIMODAL] 🔄 Text embedding sample: {text_emb[0, 0, :5]}")
            # print(f"[MULTIMODAL] 🔄 Image embedding sample: {image_emb[0, 0, :5]}")  # 注释掉图片嵌入信息
            print(f"[MULTIMODAL] 🔄 Text embedding norm: {torch.norm(text_emb[0, 0]):.4f}")
            # print(f"[MULTIMODAL] 🔄 Image embedding norm: {torch.norm(image_emb[0, 0]):.4f}")  # 注释掉图片嵌入信息
            print(f"[MULTIMODAL] 🔄 只融合ID和文本模态（去掉图片模态）")
            self._first_batch_logged = True
        
        # 直接拼接融合（不加权）
        # 注释掉加权融合代码 - 改为直接拼接融合
        # weights = F.softmax(self.modality_weights, dim=0)
        # weighted_id = weights[0] * id_embeddings
        # weighted_text = weights[1] * text_emb
        # weighted_image = weights[2] * image_emb
        
        # 直接拼接ID和文本模态（去掉图片模态）
        fused_embeddings = torch.cat([id_embeddings, text_emb], dim=-1)
        
        # 通过融合层
        fused_embeddings = self.modality_fusion(fused_embeddings)
        
        return fused_embeddings

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # 获取ID模态嵌入
        input_tokens = self.item_id2tokens[batch['input_ids']]
        id_embeddings = self.gpt2.wte(input_tokens).mean(dim=-2)
        
        # 真正的多模态融合，传入batch参数
        fused_embeddings = self._simple_fuse_modalities(id_embeddings, batch)
        
        # 通过GPT-2
        outputs = self.gpt2(inputs_embeds=fused_embeddings, attention_mask=batch['attention_mask'])
        
        # 生成最终状态
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
        生成函数，返回 top-k 物品的 Codebook 序列
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
        
        # 得到 top-k 物品的 ID (1-based)
        topk_item_ids = item_scores.topk(n_return_sequences, dim=-1).indices + 1
        
        # 使用 top-k item ID 从查找表中获取它们对应的 codebook 序列
        predicted_codebooks = self.item_id2tokens[topk_item_ids]
        
        return predicted_codebooks
