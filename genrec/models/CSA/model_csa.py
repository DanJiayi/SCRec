

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


class CSA(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(CSA, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # Add text embedding loading
        self.text_embeddings = self._load_text_embeddings()
        
        # Performance optimization: cache device transfers to avoid repeated conversion
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

        # Add text-modality fusion layer
        text_dim = self.text_embeddings.shape[1]
        # print(f"[TEXT_MODAL] 🔧 Text embedding dimension: {text_dim}")
        # print(f"[TEXT_MODAL] 🔧 GPT2 embedding dimension: {config['n_embd']}")
        
        # Reason for change: use concatenation fusion to avoid over-regularization and keep model performance
        # Original code: simple linear fusion layer
        # self.modality_fusion = nn.Linear(
        #     config['n_embd'] + text_dim, 
        #     config['n_embd']
        # )
        
        # New code: regularized multi-layer fusion network (commented out, worse performance)
        # self.modality_fusion = nn.Sequential(
        #     nn.Dropout(0.1),  # Input dropout to prevent overfitting
        #     nn.Linear(config['n_embd'] + text_dim, config['n_embd'] * 2),  # Expand dimensions
        #     nn.ReLU(),  # Non-linear activation
        #     nn.Dropout(0.1),  # Middle dropout
        #     nn.Linear(config['n_embd'] * 2, config['n_embd']),  # Compress back to target dimension
        #     nn.LayerNorm(config['n_embd']),  # Layer normalization for stable training
        #     nn.Dropout(0.05)  # Output dropout
        # )
        
        # Latest code: use a simple linear fusion layer to avoid over-regularization
        # Direct concatenation causes dimension mismatch; GPT-2 expects 448-dim input
        # So a linear layer is still needed for dimension adjustment, while keeping it simple
        self.modality_fusion = nn.Linear(
            config['n_embd']+ text_dim,  # Input: 448 + 1280 = 1728
            config['n_embd']               # Output: 448
        )
        # Use text embeddings to dynamically generate per-digit code weights
        self.code_weight_fc = nn.Linear(text_dim, self.tokenizer.n_digit)
        self.text_mlp = nn.Linear(text_dim, config['n_embd'])
        # gate network: g = sigmoid(W_g [e_cf ; e_sem])
        self.gate = nn.Linear(config['n_embd'] * 2, config['n_embd'])

        self.contrastive_alpha = self.config.get("contrastive_alpha", 0.1)
        self.contrastive_tau = self.config.get("contrastive_tau", 0.1)

        self.q_net = nn.Sequential(
        nn.Linear(self.config['n_embd'], self.config['n_embd']),
        nn.ReLU(),
        nn.Linear(self.config['n_embd'], self.config['n_embd'])
        )
        self.alpha = self.config.get("alpha", 1)
        self.manifold_beta = self.config.get("manifold_beta", 0.2)
        self.manifold_c = self.config.get("manifold_c", 1.0)
        self.proj_phi = nn.Linear(self.config['n_embd'], self.config['n_embd'])
        self.proj_psi = nn.Linear(self.config['n_embd'], self.config['n_embd'])

        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _load_text_embeddings(self) -> torch.Tensor:
        try:
            category = self.config.get('category', 'Beauty')
            # Reason for change: use cache_dir from config to support dynamic path configuration
            # Original code: hard-coded absolute path, causing permission issues
            # New code: read cache_dir from config, with a default fallback
            cache_dir = self.config.get('cache_dir', 'cache')
            processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
            
            # Support dynamic dimension: read target dimension from config, use default if missing
            target_dim = self.config.get('text_embedding_dim', None)
            
            # Prefer loading the PCA embedding file with the specified dimension first
            if target_dim:
                # Reason for change: read all PCA files from dataset directory to ensure data consistency
                # Original code: read generic PCA files from root directory
                # New code: only read from dataset directory to avoid count mismatches
                dataset_pca_file = os.path.join(processed_dir, f'final_pca_embeddings_{target_dim}d.npy')
                
                # Only try files in the dataset directory
                if os.path.exists(dataset_pca_file):
                    embeddings = np.load(dataset_pca_file)
                    print(f"[TEXT_MODAL] Loaded {target_dim}D PCA text embeddings from dataset dir: {embeddings.shape}")
                    
                    # Check whether embedding count matches dataset item count
                    if embeddings.shape[0] != self.dataset.n_items:
                        print(f"[TEXT_MODAL] Dataset PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                        raise ValueError(f"[TEXT_MODAL] Dataset PCA embeddings count mismatch: {embeddings.shape[0]} vs {self.dataset.n_items}")
                    
                    return torch.from_numpy(embeddings).float()
                else:
                    print(f"[TEXT_MODAL] Specified {target_dim}D PCA file not found in dataset dir: {dataset_pca_file}")
                    print(f"[TEXT_MODAL] Falling back to default PCA file...")
            
            # Try loading the default PCA-processed text embeddings
            pca_emb_file = os.path.join(processed_dir, 'final_pca_embeddings.npy')
            if os.path.exists(pca_emb_file):
                embeddings = np.load(pca_emb_file)
                print(f"[TEXT_MODAL] Loaded default PCA text embeddings") #: {embeddings.shape}
                
                # Check whether embedding count matches dataset item count
                if embeddings.shape[0] != self.dataset.n_items:
                    print(f"[TEXT_MODAL] PCA text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({self.dataset.n_items})")
                    raise ValueError(f"[TEXT_MODAL] PCA embeddings count mismatch: {embeddings.shape[0]} vs {self.dataset.n_items}")
                
                return torch.from_numpy(embeddings).float()
            
            # If no PCA file is available, try loading raw text embeddings
            sent_emb_file = os.path.join(processed_dir, 'text-embedding-3-large.sent_emb')
            if os.path.exists(sent_emb_file):
                # Use fromfile to load the binary embedding file
                # The text-embedding-3-large model has dimension 3072
                embeddings = np.fromfile(sent_emb_file, dtype=np.float32).reshape(-1, 3072)
                print(f"[TEXT_MODAL] Loaded raw text embeddings: {embeddings.shape}")
                
                # Check whether embedding count matches dataset item count
                # Note: the dataset may include special tokens such as [PAD], while embedding files only contain real items
                expected_embeddings = self.dataset.n_items
                if embeddings.shape[0] != expected_embeddings:
                    print(f"[TEXT_MODAL] Raw text embeddings count ({embeddings.shape[0]}) doesn't match expected ({expected_embeddings})")
                    
                    # If there are more embeddings, truncate to the first N
                    if embeddings.shape[0] > expected_embeddings:
                        print(f"[TEXT_MODAL] Truncating embeddings to match expected size: {expected_embeddings}")
                        embeddings = embeddings[:expected_embeddings]
                        print(f"[TEXT_MODAL] Truncated embeddings shape: {embeddings.shape}")
                        return torch.from_numpy(embeddings).float()
                    else:
                        # If embeddings are fewer, check whether this is caused by special tokens
                        if embeddings.shape[0] == expected_embeddings - 1:
                            print(f"[TEXT_MODAL] Detected potential special token issue")
                            print(f"[TEXT_MODAL] Embeddings: {embeddings.shape[0]}, Expected: {expected_embeddings}")
                            print(f"[TEXT_MODAL] This usually means the dataset has a [PAD] token at index 0")
                            
                            # Create a placeholder embedding (all zeros) for the [PAD] token
                            pad_embedding = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
                            embeddings = np.vstack([pad_embedding, embeddings])
                            print(f"[TEXT_MODAL] Added [PAD] embedding, final shape: {embeddings.shape}")
                            return torch.from_numpy(embeddings).float()
                        else:
                            raise ValueError(f"[TEXT_MODAL] Text embeddings count ({embeddings.shape[0]}) doesn't match dataset items ({expected_embeddings}) and cannot be automatically fixed")
                
                return torch.from_numpy(embeddings).float()
            
            # If none is found, raise an error directly
            raise FileNotFoundError("[TEXT_MODAL] No text embeddings found. Please ensure text embeddings are available.")
            
        except Exception as e:
            print(f"[TEXT_MODAL] Error loading text embeddings: {e}")
            raise RuntimeError(f"[TEXT_MODAL] Failed to load text embeddings: {e}")

    def _fuse_text_modality(self, id_embeddings, batch):
        batch_size = id_embeddings.shape[0]
        seq_len = id_embeddings.shape[1]
        device = id_embeddings.device
        
        # Get text embedding dimension
        text_dim = self.text_embeddings.shape[1]
        
        # Get item IDs in the batch
        item_ids = batch['input_ids']  # shape: (batch_size, seq_len)
        
        # --- Performance bottleneck optimization: replace double loops with vectorized indexing ---
        # Original code: double Python loops with very poor performance
        # text_emb = torch.zeros(batch_size, seq_len, text_dim, device=device)
        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         item_id = item_ids[b, s].item()
        #         if 0 <= item_id < self.dataset.n_items:
        #             text_emb[b, s] = self.text_embeddings[item_id]
        
        # New code: vectorized indexing, complete all lookups in one step
        # Use device cache to avoid repeated device transfers
        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            self._text_embeddings_device_cache[device_key] = self.text_embeddings.to(device)
        
        text_embeddings_device = self._text_embeddings_device_cache[device_key]
        
        # Use item_ids directly as index; PyTorch handles all batch and sequence positions automatically
        # This line replaces the entire double loop and improves performance by orders of magnitude
        text_emb = text_embeddings_device[item_ids]  # shape: (batch_size, seq_len, text_dim)
        
        # Count valid embeddings (for debugging only; does not affect performance)
        # Note: 0 in item_ids is usually the [PAD] token and needs special handling
        valid_mask = (item_ids > 0) & (item_ids < self.dataset.n_items)
        valid_embeddings = valid_mask.sum().item()
        total_positions = batch_size * seq_len
        
        # Performance monitoring: check whether there are invalid item_ids (out of range)
        invalid_mask = (item_ids < 0) | (item_ids >= self.dataset.n_items)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum().item()
            if not hasattr(self, '_invalid_id_warning_logged'):
                print(f"[TEXT_MODAL] Warning: Found {invalid_count} invalid item IDs (out of range)")
                print(f"[TEXT_MODAL] Item ID range: 0 to {self.dataset.n_items - 1}")
                self._invalid_id_warning_logged = True
        
        # Print fusion statistics (only for the first batch)
        if not hasattr(self, '_first_batch_logged'):
            print(f"[TEXT_MODAL] Fusion stats: {valid_embeddings}/{total_positions} valid text embeddings")
            print(f"[TEXT_MODAL] Text embedding sample: {text_emb[0, 0, :5]}")
            print(f"[TEXT_MODAL] Text embedding norm: {torch.norm(text_emb[0, 0]):.4f}")
            print(f"[TEXT_MODAL] 🚀 Using vectorized indexing for performance optimization")
            self._first_batch_logged = True
        
        # Concatenate ID and text modalities
        # fused_embeddings = torch.cat([id_embeddings, text_emb], dim=-1)
        
        # Pass through the fusion layer
        e_sem = self.text_mlp(text_emb)          # (B,S,d)
        # gate input: [e_cf ; e_sem]
        gate_inp = torch.cat([id_embeddings, e_sem], dim=-1)
        # g = sigmoid(W_g [...])
        g = torch.sigmoid(self.gate(gate_inp))  # (B,S,d)
        # e_mix = g ⊙ e_cf + (1-g) ⊙ e_sem
        fused_embeddings = g * id_embeddings + (1 - g) * e_sem
        # fused_embeddings = self.modality_fusion(fused_embeddings)
        return fused_embeddings, e_sem
    
    def benchmark_fusion_performance(self, batch_size=32, seq_len=50, num_runs=100):
        import time
        
        print(f"[BENCHMARK] Starting fusion performance benchmark...")
        print(f"[BENCHMARK] Batch size: {batch_size}, Sequence length: {seq_len}, Runs: {num_runs}")
        
        # Create test data
        device = next(self.parameters()).device
        test_batch = {
            'input_ids': torch.randint(0, min(1000, self.dataset.n_items), (batch_size, seq_len), device=device)
        }
        test_id_embeddings = torch.randn(batch_size, seq_len, self.config['n_embd'], device=device)
        
        # Warm up GPU
        print(f"[BENCHMARK] Warming up GPU...")
        for _ in range(10):
            _ = self._fuse_text_modality(test_id_embeddings, test_batch)
        
        # Performance test
        print(f"[BENCHMARK] Running performance test...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = self._fuse_text_modality(test_id_embeddings, test_batch)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        print(f"[BENCHMARK] Performance test completed!")
        print(f"[BENCHMARK] Total time: {total_time:.4f}s")
        print(f"[BENCHMARK] Average time per run: {avg_time:.6f}s")
        print(f"[BENCHMARK] Throughput: {throughput:.2f} runs/second")
        print(f"[BENCHMARK] Estimated time for 1000 batches: {avg_time * 1000:.4f}s")
        
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
        # Reason for change: support dynamic switching between RQVae and FAISS codebooks
        if self.config.get('use_rqvae_codebook', False):
            print(f"[RQVae] Using RQVae codebook for item token mapping")
            # Use RQVae codebook
            # Reason for change: fix index out-of-range issue; item_id should start from 0 with max n_items-1
            item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
            
            valid_items = 0
            skipped_items = 0
            skipped_details = []  # Record detailed information for skipped items
            
            print(f"[RQVae] Start mapping {len(self.tokenizer.item2tokens)} codebook items to the dataset...")
            print(f"[RQVae] Total number of dataset items: {self.dataset.n_items}")
            print(f"[RQVae] Dataset item ID range: 0 to {self.dataset.n_items - 1}")
            
            for item in self.tokenizer.item2tokens:
                if item in self.dataset.item2id:
                    item_id = self.dataset.item2id[item]
                    # Ensure item_id is within the valid range
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
            
            # Output detailed skip information
            if skipped_items > 0:
                print(f"\n[RQVae] Detailed skip information ({skipped_items} items):")
                for detail in skipped_details:
                    print(f"  - {detail}")
                print()
            
            print(f"[RQVae] Mapping complete: succeeded {valid_items}, skipped {skipped_items}")
            print(f"[RQVae] Final tensor shape: {item_id2tokens.shape}")
            
            return item_id2tokens
        else:
            print(f"[FAISS] Using FAISS codebook for item token mapping")
            # Use FAISS codebook (original logic)
            # Reason for change: fix index out-of-range issue; item_id should start from 0 with max n_items-1
            item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
            
            valid_items = 0
            skipped_items = 0
            skipped_details = []  # Record detailed information for skipped items
            
            print(f"[FAISS] Start mapping {len(self.tokenizer.item2tokens)} codebook items to the dataset...")
            print(f"[FAISS] Total number of dataset items: {self.dataset.n_items}")
            print(f"[FAISS] Dataset item ID range: 0 to {self.dataset.n_items - 1}")
            
            for item in self.tokenizer.item2tokens:
                if item in self.dataset.item2id:
                    item_id = self.dataset.item2id[item]
                    # Ensure item_id is within the valid range
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
            
            # Output detailed skip information
            if skipped_items > 0:
                print(f"\n[FAISS] Detailed skip information ({skipped_items} items):")
                for detail in skipped_details:
                    print(f"  - {detail}")
                print()
            
            print(f"[FAISS] Mapping complete: succeeded {valid_items}, skipped {skipped_items}")
            print(f"[FAISS] Final tensor shape: {item_id2tokens.shape}")
            
            return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        # Get ID-modality embeddings
        input_tokens = self.item_id2tokens[batch['input_ids']]

        # id_embeddings = self.gpt2.wte(input_tokens).mean(dim=-2)
        # v_all = self.gpt2.wte(input_tokens)  # (B, S, L, d)
        # e_cf_list = []
        # for l in range(self.tokenizer.n_digit):
        #     e_cf_list.append(self.W_cf[l](v_all[:, :, l, :]))
        # id_embeddings = torch.stack(e_cf_list, dim=0).sum(dim=0)  # (B, S, d)

        v_all = self.gpt2.wte(input_tokens)   # (B, S, L, d)

        # Compute text embeddings for dynamic code weights
        device = v_all.device
        device_key = str(device)
        if device_key not in self._text_embeddings_device_cache:
            self._text_embeddings_device_cache[device_key] = self.text_embeddings.to(device)
        text_embeddings_device = self._text_embeddings_device_cache[device_key]
        item_ids = batch['input_ids']
        text_emb_for_weights = text_embeddings_device[item_ids]  # (B, S, text_dim)

        # Generate per-digit code weights from text embeddings
        tau0 = self.config.get("code_weight_tau", 2.0)
        epoch = getattr(self, 'current_epoch', 1)
        if epoch <= 10:
            factor = (epoch - 1) / 9.0
            tau = tau0 + (1.0 - tau0) * factor
        else:
            tau = 1.0
        code_logits = self.code_weight_fc(text_emb_for_weights)          # (B, S, L)
        weights = torch.softmax(code_logits / tau, dim=-1)               # (B, S, L)

        # Apply position-specific code weights to codebook embeddings
        id_embeddings = (v_all * weights[..., :, None]).sum(dim=-2)
        
        # Fuse ID and text modalities
        fused_embeddings, e_sem = self._fuse_text_modality(id_embeddings, batch)

        # Do not use other modalities
        # fused_embeddings = id_embeddings
        
        # Pass through GPT-2
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
            base_loss = torch.mean(torch.stack(losses))
            outputs.loss = base_loss

            # flatten B,S
            B, S, D = id_embeddings.shape
            e_cf_flat = id_embeddings.reshape(-1, D)

            # e_sem_detach = e_sem.detach()
            # e_sem_flat = e_sem_detach.reshape(-1, D)
            # # q(e_cf)
            # q_cf = self.q_net(e_cf_flat)
            # mi_term = F.mse_loss(q_cf, e_sem_flat, reduction="mean")
            # # q_cf = F.normalize(q_cf, dim=-1)
            # # e_sem_flat = F.normalize(e_sem_flat, dim=-1)
            # # # MI surrogate (mean cosine similarity)
            # # mi_term = -torch.mean(torch.sum(q_cf * e_sem_flat, dim=-1))
            
            # outputs.loss = base_loss + self.alpha * mi_term
            # outputs.align_loss = self.alpha * mi_term


            # ================= InfoNCE contrastive loss =================
            e_cf_flat = F.normalize(e_cf_flat, dim=-1)
            e_sem_flat = e_sem.reshape(-1, D)
            e_sem_flat = F.normalize(e_sem_flat, dim=-1)

            # similarity matrix: (N, N)
            logits = torch.matmul(e_cf_flat, e_sem_flat.T) / self.contrastive_tau
            # positives are diagonal
            labels = torch.arange(logits.size(0), device=logits.device)

            contrastive_loss = F.cross_entropy(logits, labels)

            # total loss
            outputs.loss += self.contrastive_alpha * contrastive_loss
            outputs.align_loss = self.contrastive_alpha * contrastive_loss

            # -------------------- Hyperbolic manifold alignment loss --------------------
            # Compute hyperbolic mappings
            z_hyp = self.proj_phi(id_embeddings.view(-1, D))
            z_hyp = self.hyp_map(z_hyp, self.manifold_c)          # φ(w_c)
            e_hyp = self.proj_psi(e_sem.view(-1, D)) # fused_embeddings.view(-1, D)
            e_hyp = self.hyp_map(e_hyp, self.manifold_c)   # ψ(e_hyp)

            diff_sq = torch.sum((z_hyp - e_hyp) ** 2, dim=-1)
            # Squared norms
            z_norm_sq = torch.clamp(self.manifold_c**2 - torch.sum(z_hyp ** 2, dim=-1), min=1e-6)
            e_norm_sq = torch.clamp(self.manifold_c**2 - torch.sum(e_hyp ** 2, dim=-1), min=1e-6)
            # Hyperbolic geodesic distance
            arg = 1 + 2 * (self.manifold_c ** 2) * diff_sq / (z_norm_sq * e_norm_sq)
            manifold_dist = torch.acosh(torch.clamp(arg, min=1 + 1e-5))  # Avoid numerical instability

            manifold_loss = torch.mean(manifold_dist)

            # Loss coefficient (you can add manifold_alpha: 0.05 in config.yaml)
            outputs.loss += self.manifold_beta * manifold_loss
            outputs.manifold_loss = self.manifold_beta * manifold_loss

        return outputs

    @staticmethod
    def hyp_map(x, c):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        scale = c * torch.tanh(norm / (2 * c)) / norm
        return scale * x

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
