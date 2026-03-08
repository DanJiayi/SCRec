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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_amazon_data(data_dir: Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Amazon dataset
    Args:
        data_dir: Data directory
        split: Dataset split (train/val/test)
    Returns:
        user_ids: User ID array
        item_seqs: Item sequence array
        timestamps: Timestamp array
    """
    logger.info(f"Loading {split} data from {data_dir}")
    
    # Load user-item interaction data
    user_item_file = data_dir / f"{split}_user_item.npz"
    if user_item_file.exists():
        data = np.load(user_item_file)
        user_ids = data['user_ids']
        item_seqs = data['item_seqs']
        timestamps = data['timestamps']
    else:
        # If preprocessed data is unavailable, try loading existing data formats
        logger.warning(f"Preprocessed data not found: {user_item_file}")
        logger.info("Trying to load existing data format...")
        
        # Try loading all_item_seqs.json
        all_seqs_file = data_dir / "all_item_seqs.json"
        if all_seqs_file.exists():
            logger.info(f"Loading data from {all_seqs_file}")
            with open(all_seqs_file, 'r') as f:
                all_seqs_data = json.load(f)
            
            # Extract user IDs and item sequences
            user_ids = list(all_seqs_data.keys())
            item_seqs = list(all_seqs_data.values())
            
            # Create dummy timestamps (original data has no time information)
            timestamps = []
            for seq in item_seqs:
                # Create increasing timestamps for each sequence
                seq_timestamps = list(range(len(seq)))
                timestamps.append(seq_timestamps)
            
            logger.info(f"Loaded data from all_item_seqs.json")
        else:
            # Try loading raw_data.pkl file
            raw_data_file = data_dir / "raw_data.pkl"
            if raw_data_file.exists():
                with open(raw_data_file, 'rb') as f:
                    raw_data = pickle.load(f)
                
                # Extract data
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
    Load text embeddings
    Args:
        data_dir: Data directory
        split: Dataset split
    Returns:
        Text embedding array
    """
    # Try loading PCA-reduced text embeddings
    pca_file = data_dir / f"final_pca_embeddings_{split}.npy"
    if pca_file.exists():
        logger.info(f"Loading PCA text embeddings from {pca_file}")
        embeddings = np.load(pca_file)
    else:
        # Try loading default PCA embeddings
        default_pca_file = data_dir / "final_pca_embeddings.npy"
        if default_pca_file.exists():
            logger.info(f"Loading default PCA text embeddings from {default_pca_file}")
            embeddings = np.load(default_pca_file)
        else:
            # Try loading raw text embeddings
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
    Create mapping from item IDs to text embedding indices
    Args:
        data_dir: Data directory
    Returns:
        Mapping dictionary from item IDs to text embedding indices
    """
    # Try loading item ID mapping file
    id_mapping_file = data_dir / "id_mapping.json"
    if id_mapping_file.exists():
        logger.info(f"Loading item ID mapping from {id_mapping_file}")
        with open(id_mapping_file, 'r') as f:
            id_mapping = json.load(f)
        
        # Create mapping from item IDs to indices
        # Assume id_mapping.json format is either {item_id: index} or {index: item_id}
        # We need to determine which one is item ID and which one is index
        if len(id_mapping) > 0:
            first_key = list(id_mapping.keys())[0]
            first_value = id_mapping[first_key]
            
            # Check types of first key and value to determine mapping direction
            try:
                # Try converting key to int; if successful, key is item ID
                int(first_key)
                # Key is item ID, value is index
                mapping = {int(k): int(v) for k, v in id_mapping.items()}
                logger.info(f"Loaded mapping: item_id -> index for {len(mapping)} items")
            except ValueError:
                # Key is not item ID, try converting value to int
                try:
                    int(first_value)
                    # Key is index, value is item ID
                    mapping = {int(v): int(k) for k, v in id_mapping.items()}
                    logger.info(f"Loaded mapping: index -> item_id for {len(mapping)} items")
                except ValueError:
                    # Neither fits, create a simple sequential mapping
                    logger.warning("Could not determine mapping format, creating sequential mapping")
                    mapping = {}
        else:
            mapping = {}
        
        return mapping
    
    # If mapping file is missing, try inferring from other files
    logger.warning("Item ID mapping file not found, trying to infer...")
    
    # Check if item metadata file exists
    metadata_file = data_dir / "metadata.sentence.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            item_metadata = json.load(f)
        
        # Assume item IDs are continuous and start from 0
        mapping = {}
        for i, item_id in enumerate(sorted(item_metadata.keys())):
            try:
                mapping[int(item_id)] = i
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Inferred mapping for {len(mapping)} items from metadata")
        return mapping
    
    # If all else fails, create a simple sequential mapping
    logger.warning("No mapping information found, creating sequential mapping")
    # This should be adjusted based on actual number of items
    # Temporarily return an empty dict; it will be handled later
    return {}

def prepare_contrastive_data(data_dir: Path, 
                           output_dir: Path,
                           max_seq_len: int = 128,
                           min_seq_len: int = 5) -> None:
    """
    Prepare training data for contrastive-learning RQ-VAE
    Args:
        data_dir: Input data directory
        output_dir: Output data directory
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    user_ids, item_seqs, timestamps = load_amazon_data(data_dir, "train")
    text_embeddings = load_text_embeddings(data_dir, "train")
    
    # Create mapping from item IDs to text embeddings
    item_text_mapping = create_item_text_mapping(data_dir)
    
    # If mapping is empty, create a simple mapping
    if not item_text_mapping:
        logger.info("Creating simple sequential mapping")
        unique_items = set()
        for seq in item_seqs:
            unique_items.update(seq)
        
        max_item_id = max(unique_items)
        item_text_mapping = {i: i for i in range(max_item_id + 1)}
        logger.info(f"Created mapping for {len(item_text_mapping)} items")
    
    # Filter and process sequences
    filtered_seqs = []
    filtered_text_embs = []
    
    logger.info("Processing sequences...")
    for i, seq in enumerate(tqdm(item_seqs, desc="Processing sequences")):
        # Filter out sequences that are too short
        if len(seq) < min_seq_len:
            continue
        
        # Truncate sequences that are too long
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        
        # Check whether all items in sequence have corresponding text embeddings
        valid_seq = True
        for item_id in seq:
            if item_id not in item_text_mapping:
                valid_seq = False
                break
        
        if not valid_seq:
            continue
        
        # Get corresponding text embeddings
        seq_text_embs = []
        for item_id in seq:
            text_idx = item_text_mapping[item_id]
            if text_idx < len(text_embeddings):
                seq_text_embs.append(text_embeddings[text_idx])
            else:
                # If index is out of range, use zero vector
                seq_text_embs.append(np.zeros(text_embeddings.shape[1]))
        
        filtered_seqs.append(seq)
        filtered_text_embs.append(np.array(seq_text_embs))
    
    logger.info(f"Filtered sequences: {len(filtered_seqs)}")
    
    # Convert to numpy arrays
    item_seqs_array = np.array(filtered_seqs, dtype=object)
    text_embeddings_array = np.array(filtered_text_embs, dtype=object)
    
    # Save processed data
    train_output_file = output_dir / "train_item_seqs.npy"
    train_text_file = output_dir / "train_text_embeddings.npy"
    
    np.save(train_output_file, item_seqs_array)
    np.save(train_text_file, text_embeddings_array)
    
    logger.info(f"Saved training data:")
    logger.info(f"  Item sequences: {train_output_file}")
    logger.info(f"  Text embeddings: {train_text_file}")
    
    # Create validation set (if present in original data)
    try:
        val_user_ids, val_item_seqs, val_timestamps = load_amazon_data(data_dir, "val")
        val_text_embeddings = load_text_embeddings(data_dir, "val")
        
        # Process validation set
        val_filtered_seqs = []
        val_filtered_text_embs = []
        
        for seq in tqdm(val_item_seqs, desc="Processing validation sequences"):
            if len(seq) < min_seq_len:
                continue
            
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            
            # Check validity
            valid_seq = True
            for item_id in seq:
                if item_id not in item_text_mapping:
                    valid_seq = False
                    break
            
            if not valid_seq:
                continue
            
            # Get text embeddings
            seq_text_embs = []
            for item_id in seq:
                text_idx = item_text_mapping.get(item_id, 0)
                if text_idx < len(val_text_embeddings):
                    seq_text_embs.append(val_text_embeddings[text_idx])
                else:
                    seq_text_embs.append(np.zeros(val_text_embeddings.shape[1]))
            
            val_filtered_seqs.append(seq)
            val_filtered_text_embs.append(np.array(seq_text_embs))
        
        # Save validation set
        val_output_file = output_dir / "val_item_seqs.npy"
        val_text_file = output_dir / "val_text_embeddings.npy"
        
        np.save(val_output_file, np.array(val_filtered_seqs, dtype=object))
        np.save(val_text_file, np.array(val_filtered_text_embs, dtype=object))
        
        logger.info(f"Saved validation data:")
        logger.info(f"  Item sequences: {val_output_file}")
        logger.info(f"  Text embeddings: {val_text_file}")
        
    except Exception as e:
        logger.warning(f"Could not create validation set: {e}")
    
    # Save mapping information
    mapping_file = output_dir / "item_text_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(item_text_mapping, f, indent=2)
    
    logger.info(f"Saved item-text mapping: {mapping_file}")
    
    # Save dataset statistics
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
