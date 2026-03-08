import os
import json
import torch
from tqdm import tqdm
import numpy as np

from decoder.dataset import AbstractDataset
from decoder.tokenizer import AbstractTokenizer

class Tokenizer(AbstractTokenizer):
    """
    Refactored Tokenizer.
    It not only loads RQ-VAE codes, but also creates and maintains the final item_id -> tokens mapping table.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # print("config keys:", list(config.keys()))  # Debug output commented out
        # print("RQ-VAE config:", config.get("RQ-VAE"))  # Debug output commented out
        self.rqvae_config = config["RQ-VAE"]
        self.n_codebooks = self.rqvae_config["num_layers"]
        self.codebook_size = self.rqvae_config["code_book_size"]
        
        super(Tokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        
        # item_name -> tokens dictionary
        self.item2tokens = self._init_tokenizer()
        # item_id -> tokens tensor (core change)
        self.item_id2tokens = self._map_item_tokens_tensor(dataset).to(config['device'])
        
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100
        self.collate_fn = {'train': self.collate_fn_train, 'val': self.collate_fn_eval, 'test': self.collate_fn_eval}

    @property
    def n_digit(self):
        return self.n_codebooks

    @property
    def vocab_size(self):
        return self.n_codebooks * self.codebook_size + 2
        
    @property
    def max_token_seq_len(self):
        return self.config['max_item_seq_len']

    def _init_tokenizer(self) -> dict:
        dataset_name = self.config['dataset']
        category = self.config['category']
        
        # Select which codebook to use based on config
        use_image_codebook = self.config.get('use_image_codebook', False)
        use_multimodal_codebook = self.config.get('use_multimodal_codebook', False)

        # Original multimodal default logic (fixed version) was commented out due to the requirement
        # to force using the avg codebook at the project root:
        # if use_multimodal_codebook:
        #     codes_path = f"cache/{dataset_name}/{category}/codebook/codebook-multimodal-fixed.json"
        #     self.log(f"[Tokenizer] Loading Item Codes from the fixed multimodal-embedding RQ-VAE output: {codes_path}")
        # Changed to support priority and force switches for avg/concat/image (aligned with CLEAN_CODE version)
        if use_multimodal_codebook:
            concat_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal-concat.json"
            avg_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal-avg.json"
            image_path = f"../cache/{dataset_name}/{category}/codebook/codebook-image-only.json"

            force_use_concat = self.config.get('force_use_concat_codebook', False)
            force_use_avg = self.config.get('force_use_avg_codebook', False)
            force_use_image = self.config.get('force_use_image_codebook', False)

            if force_use_avg and os.path.exists(avg_path):
                codes_path = avg_path
                self.log(f"[Tokenizer] Force using average multimodal RQ-VAE to load Item Codes: {codes_path}")
            elif force_use_concat and os.path.exists(concat_path):
                codes_path = concat_path
                self.log(f"[Tokenizer] Force using concatenated multimodal RQ-VAE to load Item Codes: {codes_path}")
            elif force_use_image and os.path.exists(image_path):
                codes_path = image_path
                self.log(f"[Tokenizer] Force using image RQ-VAE to load Item Codes: {codes_path}")
            elif os.path.exists(avg_path):
                codes_path = avg_path
                self.log(f"[Tokenizer] Loading Item Codes from average multimodal RQ-VAE: {codes_path}")
            elif os.path.exists(concat_path):
                codes_path = concat_path
                self.log(f"[Tokenizer] Loading Item Codes from concatenated multimodal RQ-VAE: {codes_path}")
            elif os.path.exists(image_path):
                codes_path = image_path
                self.log(f"[Tokenizer] Loading Item Codes from image RQ-VAE: {codes_path}")
            else:
                codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal.json"
                self.log(f"[Tokenizer] Loading Item Codes from multimodal RQ-VAE: {codes_path}")
        elif use_image_codebook:
            # Use codebook generated from image embeddings
            codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook-image.json"
            self.log(f"[Tokenizer] Loading Item Codes from image-embedding RQ-VAE output: {codes_path}")
        else:
            # Use codebook generated from text embeddings (default)
            codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook.json"
            self.log(f"[Tokenizer] Loading Item Codes from text-embedding RQ-VAE output: {codes_path}")
        
        if not os.path.exists(codes_path):
            if use_multimodal_codebook:
                raise FileNotFoundError(f"Error: Item Code file '{codes_path}' generated from fixed multimodal embeddings does not exist. Please make sure the fix script has been run to generate codebook-multimodal-fixed.json.")
            elif use_image_codebook:
                raise FileNotFoundError(f"Error: Item Code file '{codes_path}' generated from image embeddings does not exist. Please make sure RQ-VAE training based on image embeddings has been run to generate codebook-image.json.")
            else:
                raise FileNotFoundError(f"Error: Item Code file '{codes_path}' generated from text embeddings does not exist. Please make sure RQ-VAE training based on text embeddings has been run to generate codebook.json.")

        with open(codes_path, 'r') as f:
            item_id_str_map = json.load(f)
            
        item2tokens = {}
        
        if use_multimodal_codebook:
            # For multimodal codebooks, keys are Amazon item ID strings
            for amazon_item_id, codes in item_id_str_map.items():
                # Skip special tokens
                if amazon_item_id in ['[PAD]', '[UNK]', '[MASK]', '[SEP]', '[CLS]']:
                    continue
                # self.codebook_size and self.n_codebooks have already been set correctly in __init__
                adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
                item2tokens[amazon_item_id] = tuple(adjusted_tokens)
        else:
            # For other codebooks, keys are integer IDs
            item_id_map = {int(k): v for k, v in item_id_str_map.items()}
            for item_id, codes in item_id_map.items():
                if item_id == 0:
                    continue
                item_name = self.id2item[item_id]
                # self.codebook_size and self.n_codebooks have already been set correctly in __init__
                adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
                item2tokens[item_name] = tuple(adjusted_tokens)

        if use_multimodal_codebook:
            codebook_type = "multimodal embeddings"
        elif use_image_codebook:
            codebook_type = "image embeddings"
        else:
            codebook_type = "text embeddings"
        self.log(f"[Tokenizer] Successfully loaded RQ-VAE codes for {len(item2tokens)} items based on {codebook_type}.")
        return item2tokens
        
    def _map_item_tokens_tensor(self, dataset: AbstractDataset) -> torch.Tensor:
        """ (New method) Create a mapping tensor from item_id to global token ID sequences. """
        # Compute the actual required tensor size
        max_item_id = 0
        for item_name in self.item2tokens.keys():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                max_item_id = max(max_item_id, item_id)
        
        # Ensure tensor size is large enough to hold all item_id values
        tensor_size = max(dataset.n_items, max_item_id + 1)
        tensor = torch.zeros((tensor_size, self.n_digit), dtype=torch.long)
        
        for item_name, tokens in self.item2tokens.items():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                tensor[item_id] = torch.LongTensor(tokens)
        return tensor

    # --- Tokenize and Collate functions remain unchanged; omitted here for brevity ---
    # ... (paste your original _tokenize_first_n_items, _tokenize_later_items, tokenize_function, tokenize, and collate functions here)
    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        pad_lens = self.config['max_item_seq_len'] - seq_lens
        input_ids.extend([self.padding_token] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]
        pad_lens = self.config['max_item_seq_len'] - seq_lens
        input_ids.extend([self.padding_token] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'labels': all_labels, 'seq_lens': all_seq_lens}
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):], pad_labels=False
            )
            return {'input_ids': [input_ids], 'attention_mask': [attention_mask], 'labels': [labels[-1:]], 'seq_lens': [seq_lens]}

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True, batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )
            tokenized_datasets[split] = tokenized_datasets[split].flatten()
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets

    def collate_fn_train(self, a_list_of_examples: list) -> dict:
        batch = {}
        for key in a_list_of_examples[0].keys():
            batch[key] = torch.stack([example[key] for example in a_list_of_examples])
        return batch

    def collate_fn_eval(self, a_list_of_examples: list) -> dict:
        batch = {}
        for key in a_list_of_examples[0].keys():
            if key == 'labels':
                batch[key] = torch.tensor([example[key] for example in a_list_of_examples], dtype=torch.long)
            else:
                batch[key] = torch.stack([example[key] for example in a_list_of_examples])
        return batch