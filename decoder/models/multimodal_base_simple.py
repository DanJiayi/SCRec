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


class MultimodalBaseSimple(BaseModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(MultimodalBaseSimple, self).__init__(config, dataset, tokenizer)

        self.rqvae_config = self.config['RQ-VAE']
        self.codebook_size = self.rqvae_config['code_book_size']

        # Core change: directly reference the final lookup table from tokenizer
        self.item_id2tokens = self.tokenizer.item_id2tokens

        # Load only text embeddings (remove image modality)
        # self.image_embeddings = self._load_image_embeddings()  # Image embedding loading commented out
        self.text_embeddings = self._load_text_embeddings()

        gpt2config = GPT2Config(vocab_size=tokenizer.vocab_size, **config)
        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        self.pred_heads = nn.Sequential(*[ResBlock(config['n_embd']) for _ in range(self.n_pred_head)])
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        
        # Simplified multimodal fusion layer - fuse only ID and text modalities (image modality removed)
        self.modality_fusion = nn.Linear(
            config['n_embd'] + self.text_embeddings.shape[1], 
            config['n_embd']
        )
        
        # Comment out learnable modality weights - switch to direct concatenation fusion
        # self.modality_weights = nn.Parameter(torch.ones(3))  # ID, text, image

    def _load_image_embeddings(self) -> torch.Tensor:
        """Load image embeddings - prioritize loading 512-dim image embeddings"""
        try:
            category = self.config.get('category', 'Beauty')
            cache_dir = self.config.get('cache_dir', 'cache')
            img_emb_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed', 'image_embeddings')
            
            # First try loading 256-dim CLIP embeddings (to stay consistent with previous training)
            clip_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32.npy')
            clip_mapping_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32_mapping.json')
            if os.path.exists(clip_file) and os.path.exists(clip_mapping_file):
                embeddings = np.load(clip_file)
                with open(clip_mapping_file, 'r') as f:
                    import json
                    mapping = json.load(f)
                # Intelligently detect and fix CLIP mapping format
                # Based on the provided content, mapping format is {"index": "item_id"}, e.g., "18025": "B000EX6ILI"
                # We need a mapping from item_id to index for lookup during training
                sample_key = list(mapping.keys())[0]
                sample_value = mapping[sample_key]
                
                # Check whether it is a digit string
                def is_digit_string(s):
                    return isinstance(s, str) and s.isdigit()
                
                if is_digit_string(sample_key) and not is_digit_string(sample_value):
                    # Format is {"index": "item_id"}; convert to {"item_id": "index"} for lookup
                    print(f"[MULTIMODAL] Detected format: {{'index': 'item_id'}}, converting to {{'item_id': 'index'}} for lookup")
                    print(f"[MULTIMODAL] Sample: key='{sample_key}' (index), value='{sample_value}' (item_id)")
                    # Create reverse mapping: item_id -> index
                    reversed_mapping = {v: int(k) for k, v in mapping.items()}
                    self.clip_mapping = reversed_mapping
                    print(f"[MULTIMODAL] Created reverse mapping with {len(reversed_mapping)} items")
                elif not is_digit_string(sample_key) and is_digit_string(sample_value):
                    # Format is already {"item_id": "index"}; use directly
                    print(f"[MULTIMODAL] Detected format: {{'item_id': 'index'}}, using directly")
                    print(f"[MULTIMODAL] Sample: key='{sample_key}' (item_id), value='{sample_value}' (index)")
                    self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                else:
                    # Unable to determine format; use original mapping and print a warning
                    print(f"[MULTIMODAL] Unable to determine mapping format, using original mapping")
                    print(f"[MULTIMODAL] Sample key: {sample_key} (type: {type(sample_key)})")
                    print(f"[MULTIMODAL] Sample value: {sample_value} (type: {type(sample_value)})")
                    # Try a smart fix: if key looks like item_id (starts with B) and value looks numeric
                    if isinstance(sample_key, str) and sample_key.startswith('B') and isinstance(sample_value, str) and sample_value.isdigit():
                        print(f"[MULTIMODAL] Attempting smart fix: treating as {{'item_id': 'index'}} format")
                        self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                    else:
                        self.clip_mapping = mapping
                
                print(f"[MULTIMODAL] Loaded 256-dim CLIP image embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] CLIP mapping contains {len(self.clip_mapping)} items")
                print(f"[MULTIMODAL] Image embedding sample: {embeddings[0][:5]}")  # Show first 5 values
                return torch.from_numpy(embeddings).float()
            
            # If 256-dim file is missing, try loading 512-dim CLIP embeddings (fallback)
            # Original implementation prioritized 512-dim; changed to prioritize 256-dim for training consistency
            clip_512_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32-512d.npy')
            clip_512_mapping_file = os.path.join(img_emb_dir, 'image_embeddings_clip-vit-base-patch32-512d_mapping.json')
            if os.path.exists(clip_512_file) and os.path.exists(clip_512_mapping_file):
                embeddings = np.load(clip_512_file)
                with open(clip_512_mapping_file, 'r') as f:
                    import json
                    mapping = json.load(f)
                # Intelligently detect and fix CLIP mapping format
                # Based on the provided content, mapping format is {"index": "item_id"}, e.g., "18025": "B000EX6ILI"
                # We need a mapping from item_id to index for lookup during training
                sample_key = list(mapping.keys())[0]
                sample_value = mapping[sample_key]
                
                # Check whether it is a digit string
                def is_digit_string_512(s):
                    return isinstance(s, str) and s.isdigit()
                
                if is_digit_string_512(sample_key) and not is_digit_string_512(sample_value):
                    # Format is {"index": "item_id"}; convert to {"item_id": "index"} for lookup
                    print(f"[MULTIMODAL] 🔄 Detected format: {{'index': 'item_id'}}, converting to {{'item_id': 'index'}} for lookup")
                    print(f"[MULTIMODAL] 🔄 Sample: key='{sample_key}' (index), value='{sample_value}' (item_id)")
                    # Create reverse mapping: item_id -> index
                    reversed_mapping = {v: int(k) for k, v in mapping.items()}
                    self.clip_mapping = reversed_mapping
                    print(f"[MULTIMODAL] ✅ Created reverse mapping with {len(reversed_mapping)} items")
                elif not is_digit_string_512(sample_key) and is_digit_string_512(sample_value):
                    # Format is already {"item_id": "index"}; use directly
                    print(f"[MULTIMODAL] ✅ Detected format: {{'item_id': 'index'}}, using directly")
                    print(f"[MULTIMODAL] ✅ Sample: key='{sample_key}' (item_id), value='{sample_value}' (index)")
                    self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                else:
                    # Unable to determine format; use original mapping and print a warning
                    print(f"[MULTIMODAL] ⚠️ Unable to determine mapping format, using original mapping")
                    print(f"[MULTIMODAL] ⚠️ Sample key: {sample_key} (type: {type(sample_key)})")
                    print(f"[MULTIMODAL] ⚠️ Sample value: {sample_value} (type: {type(sample_value)})")
                    # Try a smart fix: if key looks like item_id (starts with B) and value looks numeric
                    if isinstance(sample_key, str) and sample_key.startswith('B') and isinstance(sample_value, str) and sample_value.isdigit():
                        print(f"[MULTIMODAL] 🔄 Attempting smart fix: treating as {{'item_id': 'index'}} format")
                        self.clip_mapping = {k: int(v) for k, v in mapping.items()}
                    else:
                        self.clip_mapping = mapping
                
                print(f"[MULTIMODAL] ⚠️ Loaded 512-dim CLIP image embeddings (fallback): {embeddings.shape}")
                print(f"[MULTIMODAL] ⚠️ CLIP mapping contains {len(self.clip_mapping)} items")
                return torch.from_numpy(embeddings).float()
            
            # Try loading random vectors
            random_file = os.path.join(img_emb_dir, 'image_embeddings_random.npy')
            if os.path.exists(random_file):
                embeddings = np.load(random_file)
                print(f"[MULTIMODAL] ⚠️ Loaded random image embeddings: {embeddings.shape}")
                return torch.from_numpy(embeddings).float()
            
            # If none found, generate random vectors
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
        """Load text embeddings"""
        try:
            category = self.config.get('category', 'Beauty')
            cache_dir = self.config.get('cache_dir', 'cache')
            processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
            
            # Prefer loading PCA-processed text embeddings first (correct shape)
            pca_emb_file = os.path.join(processed_dir, 'final_pca_embeddings.npy')
            if os.path.exists(pca_emb_file):
                embeddings = np.load(pca_emb_file)
                print(f"[MULTIMODAL] ✅ Loaded PCA-processed text embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] ✅ Text embedding sample: {embeddings[0][:5]}")  # Show first 5 values
                
                # Check whether embedding count matches dataset item count
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[MULTIMODAL] ⚠️ PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    print("[MULTIMODAL] ⚠️ Generating random text embeddings for current dataset")
                    n_items = self.dataset.n_items
                    text_dim = self.config.get('sent_emb_pca', 512)
                    random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
                    return torch.from_numpy(random_embeddings).float()
                
                return torch.from_numpy(embeddings).float()
            
            # If no PCA file is available, try loading raw text embeddings
            sent_emb_file = os.path.join(processed_dir, 'text-embedding-3-large.sent_emb')
            if os.path.exists(sent_emb_file):
                # Use fromfile to load the binary embedding file
                embeddings = np.fromfile(sent_emb_file, dtype=np.float32).reshape(-1, 512)
                print(f"[MULTIMODAL] ⚠️ Loaded raw text embeddings: {embeddings.shape}")
                print(f"[MULTIMODAL] ⚠️ Text embedding sample: {embeddings[0][:5]}")  # Show first 5 values
                
                # Check whether embedding count matches dataset item count
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[MULTIMODAL] ⚠️ Raw text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    print("[MULTIMODAL] ⚠️ Generating random text embeddings for current dataset")
                    n_items = self.dataset.n_items
                    text_dim = self.config.get('sent_emb_pca', 512)
                    random_embeddings = np.random.normal(0, 1, (n_items, text_dim))
                    return torch.from_numpy(random_embeddings).float()
                
                return torch.from_numpy(embeddings).float()
            
            # If none found, generate random vectors
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
        """Actual multimodal fusion using real image and text embeddings"""
        
        # If using a multimodal codebook, return ID embeddings directly without additional fusion
        # if self.config.get('use_multimodal_codebook', False):
        #     return id_embeddings
        batch_size = id_embeddings.shape[0]
        seq_len = id_embeddings.shape[1]
        device = id_embeddings.device
        
        # Get embedding dimensions
        text_dim = self.text_embeddings.shape[1]
        # image_dim = self.image_embeddings.shape[1]  # Image dimension commented out
        
        # Get item IDs in the batch
        item_ids = batch['input_ids']  # shape: (batch_size, seq_len)
        
        # Get corresponding text embeddings from embedding table
        # Note: item_ids are 1-based, but embedding table is 0-based
        text_emb = torch.zeros(batch_size, seq_len, text_dim, device=device)
        # image_emb = torch.zeros(batch_size, seq_len, image_dim, device=device)  # Image embeddings commented out
        
        # Count number of valid embeddings
        valid_embeddings = 0
        total_positions = batch_size * seq_len
        
        # Debug info: check embedding table sizes
        if not hasattr(self, '_first_batch_logged'):
            print(f"[MULTIMODAL] 🔍 Debug info:")
            print(f"[MULTIMODAL] 🔍 Text embeddings shape: {self.text_embeddings.shape}")
            # print(f"[MULTIMODAL] 🔍 Image embeddings shape: {self.image_embeddings.shape}")  # Image embedding info commented out
            print(f"[MULTIMODAL] 🔍 Dataset n_items: {self.dataset.n_items}")
            print(f"[MULTIMODAL] 🔍 Item IDs range: {item_ids.min().item()} to {item_ids.max().item()}")
        
        # Get corresponding embeddings for each item in each batch
        for b in range(batch_size):
            for s in range(seq_len):
                item_id = item_ids[b, s].item()
                # Fix index issue: item_id should be 1..n_items, while embedding table index is 0..n_items-1
                if item_id > 0 and item_id <= self.dataset.n_items:
                    # Get text embedding
                    text_emb[b, s] = self.text_embeddings[item_id - 1]  # Convert to 0-based index
                    # Image embedding processing commented out (image modality removed)
                    # Get image embedding - use CLIP mapping
                    # clip_mapping is now in {"item_id": "index"} format, and we need item_id to find the corresponding item string
                    # But item_id is numeric, so we need to find the corresponding item_id string
                    # if hasattr(self, 'clip_mapping') and self.clip_mapping:
                    #     # Since we cannot directly map item_id to item string, temporarily use the default indexing method
                    #     # This is a limitation of the original design, so we temporarily use item_id-1 as index
                    #     if item_id - 1 < self.image_embeddings.shape[0]:
                    #         image_emb[b, s] = self.image_embeddings[item_id - 1]
                    #         valid_embeddings += 1
                    #     else:
                    #         print(f"[MULTIMODAL] ⚠️ Item ID {item_id} out of range for image embeddings shape {self.image_embeddings.shape}")
                    # else:
                    #     # If CLIP mapping is unavailable, use default index (may be inaccurate)
                    #     if item_id - 1 < self.image_embeddings.shape[0]:
                    #         image_emb[b, s] = self.image_embeddings[item_id - 1]
                    #         valid_embeddings += 1
        
        # Print fusion statistics (only for the first batch)
        if not hasattr(self, '_first_batch_logged'):
            print(f"[MULTIMODAL] Fusion stats: {valid_embeddings}/{total_positions} valid embeddings")
            print(f"[MULTIMODAL] Text embedding sample: {text_emb[0, 0, :5]}")
            # print(f"[MULTIMODAL] Image embedding sample: {image_emb[0, 0, :5]}")  # Image embedding info commented out
            print(f"[MULTIMODAL] Text embedding norm: {torch.norm(text_emb[0, 0]):.4f}")
            # print(f"[MULTIMODAL] Image embedding norm: {torch.norm(image_emb[0, 0]):.4f}")  # Image embedding info commented out
            print(f"[MULTIMODAL] Fuse only ID and text modalities (image modality removed)")
            self._first_batch_logged = True
        
        # Direct concatenation fusion (unweighted)
        # Comment out weighted fusion code - switch to direct concatenation fusion
        # weights = F.softmax(self.modality_weights, dim=0)
        # weighted_id = weights[0] * id_embeddings
        # weighted_text = weights[1] * text_emb
        # weighted_image = weights[2] * image_emb
        
        # Directly concatenate ID and text modalities (image modality removed)
        fused_embeddings = torch.cat([id_embeddings, text_emb], dim=-1)
        
        # Pass through fusion layer
        fused_embeddings = self.modality_fusion(fused_embeddings)
        
        return fused_embeddings

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # Get ID-modality embeddings
        input_tokens = self.item_id2tokens[batch['input_ids']]
        id_embeddings = self.gpt2.wte(input_tokens).mean(dim=-2)
        
        # Actual multimodal fusion with batch input
        fused_embeddings = self._simple_fuse_modalities(id_embeddings, batch)
        
        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=fused_embeddings, attention_mask=batch['attention_mask'])
        
        # Generate final states
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
        Generation function that returns top-k item Codebook sequences
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
        
        # Use top-k item IDs to fetch their corresponding codebook sequences from the lookup table
        predicted_codebooks = self.item_id2tokens[topk_item_ids]
        
        return predicted_codebooks
