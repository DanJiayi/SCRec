#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_amazon_data(data_dir: Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载Amazon数据集
    Args:
        data_dir: 数据目录
        split: 数据集分割 (train/val/test)
    Returns:
        user_ids: 用户ID数组
        item_seqs: 物品序列数组
        timestamps: 时间戳数组
    """
    logger.info(f"Loading {split} data from {data_dir}")
    
    # 加载用户-物品交互数据
    user_item_file = data_dir / f"{split}_user_item.npz"
    if user_item_file.exists():
        data = np.load(user_item_file)
        user_ids = data['user_ids']
        item_seqs = data['item_seqs']
        timestamps = data['timestamps']
    else:
        # 如果没有预处理的数据，尝试加载现有的数据格式
        logger.warning(f"Preprocessed data not found: {user_item_file}")
        logger.info("Trying to load existing data format...")
        
        # 尝试加载all_item_seqs.json
        all_seqs_file = data_dir / "all_item_seqs.json"
        if all_seqs_file.exists():
            logger.info(f"Loading data from {all_seqs_file}")
            with open(all_seqs_file, 'r') as f:
                all_seqs_data = json.load(f)
            
            # 提取用户ID和物品序列
            user_ids = list(all_seqs_data.keys())
            item_seqs = list(all_seqs_data.values())
            
            # 创建虚拟时间戳（因为原始数据中没有时间信息）
            timestamps = []
            for seq in item_seqs:
                # 为每个序列创建递增的时间戳
                seq_timestamps = list(range(len(seq)))
                timestamps.append(seq_timestamps)
            
            logger.info(f"Loaded data from all_item_seqs.json")
        else:
            # 尝试加载raw_data.pkl文件
            raw_data_file = data_dir / "raw_data.pkl"
            if raw_data_file.exists():
                with open(raw_data_file, 'rb') as f:
                    raw_data = pickle.load(f)
                
                # 提取数据
                user_ids = raw_data[f'{split}_user_ids']
                item_seqs = raw_data[f'{split}_item_seqs']
                timestamps = raw_data[f'{split}_timestamps']
            else:
                raise FileNotFoundError(f"No data found in {data_dir}")
    
    logger.info(f"Loaded {split} data:")
    logger.info(f"  Users: {len(user_ids)}")
    logger.info(f"  Sequences: {len(item_seqs)}")
    logger.info(f"  Max sequence length: {max(len(seq) for seq in item_seqs)}")
    
    return user_ids, item_seqs, timestamps

def load_text_embeddings(data_dir: Path, split: str = "train") -> np.ndarray:
    """
    加载文本嵌入
    Args:
        data_dir: 数据目录
        split: 数据集分割
    Returns:
        文本嵌入数组
    """
    # 尝试加载PCA降维后的文本嵌入
    pca_file = data_dir / f"final_pca_embeddings_{split}.npy"
    if pca_file.exists():
        logger.info(f"Loading PCA text embeddings from {pca_file}")
        embeddings = np.load(pca_file)
    else:
        # 尝试加载默认PCA嵌入
        default_pca_file = data_dir / "final_pca_embeddings.npy"
        if default_pca_file.exists():
            logger.info(f"Loading default PCA text embeddings from {default_pca_file}")
            embeddings = np.load(default_pca_file)
        else:
            # 尝试加载原始文本嵌入
            sent_emb_file = data_dir / "text-embedding-3-large.sent_emb"
            if sent_emb_file.exists():
                logger.info(f"Loading raw text embeddings from {sent_emb_file}")
                embeddings = np.fromfile(sent_emb_file, dtype=np.float32).reshape(-1, 3072)
            else:
                raise FileNotFoundError(f"No text embeddings found in {data_dir}")
    
    logger.info(f"Text embeddings shape: {embeddings.shape}")
    return embeddings

def create_item_text_mapping(data_dir: Path) -> Dict[int, int]:
    """
    创建物品ID到文本嵌入索引的映射
    Args:
        data_dir: 数据目录
    Returns:
        物品ID到文本嵌入索引的映射字典
    """
    # 尝试加载物品ID映射文件
    id_mapping_file = data_dir / "id_mapping.json"
    if id_mapping_file.exists():
        logger.info(f"Loading item ID mapping from {id_mapping_file}")
        with open(id_mapping_file, 'r') as f:
            id_mapping = json.load(f)
        
        # 创建物品ID到索引的映射
        # 假设id_mapping.json的格式是 {item_id: index} 或 {index: item_id}
        # 我们需要确定哪个是物品ID，哪个是索引
        if len(id_mapping) > 0:
            first_key = list(id_mapping.keys())[0]
            first_value = id_mapping[first_key]
            
            # 检查第一个键和值的类型，确定映射关系
            try:
                # 尝试将键转换为整数，如果成功说明键是物品ID
                int(first_key)
                # 键是物品ID，值是索引
                mapping = {int(k): int(v) for k, v in id_mapping.items()}
                logger.info(f"Loaded mapping: item_id -> index for {len(mapping)} items")
            except ValueError:
                # 键不是物品ID，尝试将值转换为整数
                try:
                    int(first_value)
                    # 键是索引，值是物品ID
                    mapping = {int(v): int(k) for k, v in id_mapping.items()}
                    logger.info(f"Loaded mapping: index -> item_id for {len(mapping)} items")
                except ValueError:
                    # 都不是，创建简单的连续映射
                    logger.warning("Could not determine mapping format, creating sequential mapping")
                    mapping = {}
        else:
            mapping = {}
        
        return mapping
    
    # 如果没有映射文件，尝试从其他文件推断
    logger.warning("Item ID mapping file not found, trying to infer...")
    
    # 检查是否有物品元数据文件
    metadata_file = data_dir / "metadata.sentence.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            item_metadata = json.load(f)
        
        # 假设物品ID是连续的，从0开始
        mapping = {}
        for i, item_id in enumerate(sorted(item_metadata.keys())):
            try:
                mapping[int(item_id)] = i
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Inferred mapping for {len(mapping)} items from metadata")
        return mapping
    
    # 如果都没有，创建一个简单的连续映射
    logger.warning("No mapping information found, creating sequential mapping")
    # 这里需要根据实际的物品数量来调整
    # 暂时返回空字典，后续处理时会处理
    return {}

def prepare_contrastive_data(data_dir: Path, 
                           output_dir: Path,
                           max_seq_len: int = 128,
                           min_seq_len: int = 5) -> None:
    """
    准备对比学习RQ-VAE的训练数据
    Args:
        data_dir: 输入数据目录
        output_dir: 输出数据目录
        max_seq_len: 最大序列长度
        min_seq_len: 最小序列长度
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    user_ids, item_seqs, timestamps = load_amazon_data(data_dir, "train")
    text_embeddings = load_text_embeddings(data_dir, "train")
    
    # 创建物品ID到文本嵌入的映射
    item_text_mapping = create_item_text_mapping(data_dir)
    
    # 如果没有映射，创建一个简单的映射
    if not item_text_mapping:
        logger.info("Creating simple sequential mapping")
        unique_items = set()
        for seq in item_seqs:
            unique_items.update(seq)
        
        max_item_id = max(unique_items)
        item_text_mapping = {i: i for i in range(max_item_id + 1)}
        logger.info(f"Created mapping for {len(item_text_mapping)} items")
    
    # 过滤和处理序列
    filtered_seqs = []
    filtered_text_embs = []
    
    logger.info("Processing sequences...")
    for i, seq in enumerate(tqdm(item_seqs, desc="Processing sequences")):
        # 过滤太短的序列
        if len(seq) < min_seq_len:
            continue
        
        # 截断太长的序列
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        
        # 检查序列中的物品是否都有对应的文本嵌入
        valid_seq = True
        for item_id in seq:
            if item_id not in item_text_mapping:
                valid_seq = False
                break
        
        if not valid_seq:
            continue
        
        # 获取对应的文本嵌入
        seq_text_embs = []
        for item_id in seq:
            text_idx = item_text_mapping[item_id]
            if text_idx < len(text_embeddings):
                seq_text_embs.append(text_embeddings[text_idx])
            else:
                # 如果索引超出范围，使用零向量
                seq_text_embs.append(np.zeros(text_embeddings.shape[1]))
        
        filtered_seqs.append(seq)
        filtered_text_embs.append(np.array(seq_text_embs))
    
    logger.info(f"Filtered sequences: {len(filtered_seqs)}")
    
    # 转换为numpy数组
    item_seqs_array = np.array(filtered_seqs, dtype=object)
    text_embeddings_array = np.array(filtered_text_embs, dtype=object)
    
    # 保存处理后的数据
    train_output_file = output_dir / "train_item_seqs.npy"
    train_text_file = output_dir / "train_text_embeddings.npy"
    
    np.save(train_output_file, item_seqs_array)
    np.save(train_text_file, text_embeddings_array)
    
    logger.info(f"Saved training data:")
    logger.info(f"  Item sequences: {train_output_file}")
    logger.info(f"  Text embeddings: {train_text_file}")
    
    # 创建验证集（如果原始数据中有的话）
    try:
        val_user_ids, val_item_seqs, val_timestamps = load_amazon_data(data_dir, "val")
        val_text_embeddings = load_text_embeddings(data_dir, "val")
        
        # 处理验证集
        val_filtered_seqs = []
        val_filtered_text_embs = []
        
        for seq in tqdm(val_item_seqs, desc="Processing validation sequences"):
            if len(seq) < min_seq_len:
                continue
            
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            
            # 检查有效性
            valid_seq = True
            for item_id in seq:
                if item_id not in item_text_mapping:
                    valid_seq = False
                    break
            
            if not valid_seq:
                continue
            
            # 获取文本嵌入
            seq_text_embs = []
            for item_id in seq:
                text_idx = item_text_mapping.get(item_id, 0)
                if text_idx < len(val_text_embeddings):
                    seq_text_embs.append(val_text_embeddings[text_idx])
                else:
                    seq_text_embs.append(np.zeros(val_text_embeddings.shape[1]))
            
            val_filtered_seqs.append(seq)
            val_filtered_text_embs.append(np.array(seq_text_embs))
        
        # 保存验证集
        val_output_file = output_dir / "val_item_seqs.npy"
        val_text_file = output_dir / "val_text_embeddings.npy"
        
        np.save(val_output_file, np.array(val_filtered_seqs, dtype=object))
        np.save(val_text_file, np.array(val_filtered_text_embs, dtype=object))
        
        logger.info(f"Saved validation data:")
        logger.info(f"  Item sequences: {val_output_file}")
        logger.info(f"  Text embeddings: {val_text_file}")
        
    except Exception as e:
        logger.warning(f"Could not create validation set: {e}")
    
    # 保存映射信息
    mapping_file = output_dir / "item_text_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(item_text_mapping, f, indent=2)
    
    logger.info(f"Saved item-text mapping: {mapping_file}")
    
    # 保存数据集统计信息
    stats = {
        "num_sequences": len(filtered_seqs),
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len,
        "text_embedding_dim": text_embeddings.shape[1],
        "num_unique_items": len(item_text_mapping),
        "avg_seq_len": np.mean([len(seq) for seq in filtered_seqs])
    }
    
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"Data preparation completed! Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare contrastive RQ-VAE training data")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Input data directory containing Amazon dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--max_seq_len", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--min_seq_len", type=int, default=5,
                       help="Minimum sequence length")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    prepare_contrastive_data(
        data_dir=data_dir,
        output_dir=output_dir,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len
    )


if __name__ == "__main__":
    main()
