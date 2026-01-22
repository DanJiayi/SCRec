import os
import json
import torch
from tqdm import tqdm
import numpy as np

from decoder.dataset import AbstractDataset
from decoder.tokenizer import AbstractTokenizer

class Tokenizer(AbstractTokenizer):
    """
    改造后的 Tokenizer。
    它不仅加载 RQ-VAE codes，还负责创建和维护最终的 item_id -> tokens 映射表。
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # print("🔑 config keys:", list(config.keys()))  # 注释掉调试输出
        # print("🔑 RQ-VAE config:", config.get("RQ-VAE"))  # 注释掉调试输出
        self.rqvae_config = config["RQ-VAE"]
        self.n_codebooks = self.rqvae_config["num_layers"]
        self.codebook_size = self.rqvae_config["code_book_size"]
        
        super(Tokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        
        # item_name -> tokens 字典
        self.item2tokens = self._init_tokenizer()
        # item_id -> tokens 张量 (核心改动)
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
        
        # 根据配置选择使用哪种codebook
        use_image_codebook = self.config.get('use_image_codebook', False)
        use_multimodal_codebook = self.config.get('use_multimodal_codebook', False)

        # 原多模态默认逻辑（固定加载 fixed 版本）因“需要在根目录下强制使用 avg 码本”被注释：
        # if use_multimodal_codebook:
        #     codes_path = f"cache/{dataset_name}/{category}/codebook/codebook-multimodal-fixed.json"
        #     self.log(f"✅ [Tokenizer] 正在从修复后的多模态嵌入生成的 RQ-VAE 成果加载 Item Codes: {codes_path}")
        # 改为支持 avg/concat/image 的优先级与强制开关（与 CLEAN_CODE 版本对齐）
        if use_multimodal_codebook:
            concat_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal-concat.json"
            avg_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal-avg.json"
            image_path = f"../cache/{dataset_name}/{category}/codebook/codebook-image-only.json"

            force_use_concat = self.config.get('force_use_concat_codebook', False)
            force_use_avg = self.config.get('force_use_avg_codebook', False)
            force_use_image = self.config.get('force_use_image_codebook', False)

            if force_use_avg and os.path.exists(avg_path):
                codes_path = avg_path
                self.log(f"✅ [Tokenizer] 强制使用均值多模态 RQ-VAE 加载 Item Codes: {codes_path}")
            elif force_use_concat and os.path.exists(concat_path):
                codes_path = concat_path
                self.log(f"✅ [Tokenizer] 强制使用拼接多模态 RQ-VAE 加载 Item Codes: {codes_path}")
            elif force_use_image and os.path.exists(image_path):
                codes_path = image_path
                self.log(f"✅ [Tokenizer] 强制使用图片 RQ-VAE 加载 Item Codes: {codes_path}")
            elif os.path.exists(avg_path):
                codes_path = avg_path
                self.log(f"✅ [Tokenizer] 正在从均值多模态 RQ-VAE 加载 Item Codes: {codes_path}")
            elif os.path.exists(concat_path):
                codes_path = concat_path
                self.log(f"✅ [Tokenizer] 正在从拼接多模态 RQ-VAE 加载 Item Codes: {codes_path}")
            elif os.path.exists(image_path):
                codes_path = image_path
                self.log(f"✅ [Tokenizer] 正在从图片 RQ-VAE 加载 Item Codes: {codes_path}")
            else:
                codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook-multimodal.json"
                self.log(f"✅ [Tokenizer] 正在从多模态 RQ-VAE 加载 Item Codes: {codes_path}")
        elif use_image_codebook:
            # 使用基于图片嵌入生成的codebook
            codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook-image.json"
            self.log(f"✅ [Tokenizer] 正在从图片嵌入生成的 RQ-VAE 成果加载 Item Codes: {codes_path}")
        else:
            # 使用基于文本嵌入生成的codebook（默认）
            codes_path = f"../cache/{dataset_name}/{category}/codebook/codebook.json"
            self.log(f"✅ [Tokenizer] 正在从文本嵌入生成的 RQ-VAE 成果加载 Item Codes: {codes_path}")
        
        if not os.path.exists(codes_path):
            if use_multimodal_codebook:
                raise FileNotFoundError(f"错误: 修复后的多模态嵌入生成的 Item Code 文件 '{codes_path}' 不存在。请确保已运行修复脚本生成 codebook-multimodal-fixed.json。")
            elif use_image_codebook:
                raise FileNotFoundError(f"错误: 图片嵌入生成的 Item Code 文件 '{codes_path}' 不存在。请确保已运行基于图片嵌入的 RQ-VAE 训练生成 codebook-image.json。")
            else:
                raise FileNotFoundError(f"错误: 文本嵌入生成的 Item Code 文件 '{codes_path}' 不存在。请确保已运行基于文本嵌入的 RQ-VAE 训练生成 codebook.json。")

        with open(codes_path, 'r') as f:
            item_id_str_map = json.load(f)
            
        item2tokens = {}
        
        if use_multimodal_codebook:
            # 对于多模态码本，键是Amazon商品ID字符串
            for amazon_item_id, codes in item_id_str_map.items():
                # 跳过特殊标记
                if amazon_item_id in ['[PAD]', '[UNK]', '[MASK]', '[SEP]', '[CLS]']:
                    continue
                # 这里使用的 self.codebook_size 和 self.n_codebooks 已经在 __init__ 中正确设置
                adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
                item2tokens[amazon_item_id] = tuple(adjusted_tokens)
        else:
            # 对于其他码本，键是整数ID
            item_id_map = {int(k): v for k, v in item_id_str_map.items()}
            for item_id, codes in item_id_map.items():
                if item_id == 0:
                    continue
                item_name = self.id2item[item_id]
                # 这里使用的 self.codebook_size 和 self.n_codebooks 已经在 __init__ 中正确设置
                adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
                item2tokens[item_name] = tuple(adjusted_tokens)

        if use_multimodal_codebook:
            codebook_type = "多模态嵌入"
        elif use_image_codebook:
            codebook_type = "图片嵌入"
        else:
            codebook_type = "文本嵌入"
        self.log(f"[Tokenizer] 成功加载了 {len(item2tokens)} 个物品的基于{codebook_type}的 RQ-VAE codes。")
        return item2tokens
        
    def _map_item_tokens_tensor(self, dataset: AbstractDataset) -> torch.Tensor:
        """ (新增方法) 创建从 item_id 到全局 token ID 序列的映射张量。"""
        # 计算实际需要的张量大小
        max_item_id = 0
        for item_name in self.item2tokens.keys():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                max_item_id = max(max_item_id, item_id)
        
        # 确保张量大小足够容纳所有 item_id
        tensor_size = max(dataset.n_items, max_item_id + 1)
        tensor = torch.zeros((tensor_size, self.n_digit), dtype=torch.long)
        
        for item_name, tokens in self.item2tokens.items():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                tensor[item_id] = torch.LongTensor(tokens)
        return tensor

    # --- Tokenize 和 Collate 函数保持不变，此处省略以保持简洁 ---
    # ... (将您原来的 _tokenize_first_n_items, _tokenize_later_items, tokenize_function, tokenize, 和 collate 函数粘贴在这里)
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