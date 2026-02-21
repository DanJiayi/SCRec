#!/usr/bin/env python3
"""
Convert embeddings.json (format: item_id: [emb1, emb2, emb3]) to liger RQ-VAE format.
Uses emb3(fused embedding) by default. Outputs _embeddings_new.pt.

Uses only embeddings.json. Runs liger preprocessing if id2meta/id2item are missing,
skips if they already exist.

IMPORTANT: When using run.py with +dataset.embedding_suffix=_new, pass
+dataset.embedding_file_model=sentence-t5-base so run.py looks for this file.
Then --output must match: {processed_path}/{dataset_name}_{embedding_file_model}_embeddings_new.pt
Example: ./ID_generation/preprocessing/processed/Beauty_sentence-t5-base_embeddings_new.pt

Usage:
  python scripts/convert_embeddings_to_liger.py \
    --embeddings /path/to/embeddings.json \
    --output ./ID_generation/preprocessing/processed/Beauty_sentence-t5-base_embeddings_new.pt \
    [--dataset Beauty] [--data-type Amazon] [--raw-path ...] [--processed-path ...] [--features ...] [--prompt-format amazon]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Add project root for importing ID_generation
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_preprocessing_run(config):
    """
    Run liger preprocessing if id2meta or id2item do not exist.
    Returns (id2meta_path, id2item_path).
    """
    features_used = "_".join(config["features_needed"])
    id2meta_path = os.path.join(
        config["processed_data_path"],
        f"{config['dataset_name']}_{features_used}_{config['prompt_format']}_id2meta.json",
    )
    id2item_path = os.path.join(
        config["processed_data_path"],
        f"{config['dataset_name']}_id2item.json",
    )
    if os.path.exists(id2meta_path) and os.path.exists(id2item_path):
        print(f"Found existing id2meta and id2item, skipping preprocessing.")
        return id2meta_path, id2item_path

    print("id2meta or id2item not found, running liger preprocessing...")
    from ID_generation.preprocessing.data_process import preprocessing

    prep_config = {
        "name": config["dataset_name"],
        "type": config["data_type"],
        "raw_data_path": config["raw_data_path"],
        "processed_data_path": config["processed_data_path"],
        "features_needed": config["features_needed"],
        "prompt_format": config["prompt_format"],
    }
    preprocessing(prep_config)
    return id2meta_path, id2item_path


def convert(embeddings_path, id2meta_path, id2item_path, output_path, emb_index=2):
    """
    Convert embeddings.json to liger format using id2meta + id2item for correct ordering.
    emb_index: 0=emb1, 1=emb2, 2=emb3 (default).
    """
    emb_data = load_json(embeddings_path)
    id2meta = load_json(id2meta_path)
    id2item = load_json(id2item_path)

    sorted_ids = sorted(id2meta.keys(), key=lambda x: int(x))
    emb_list = []
    missing = []
    for iid in sorted_ids:
        asin = id2item.get(iid) or id2item.get(str(iid))
        if asin is None:
            missing.append(iid)
            continue
        if asin not in emb_data:
            missing.append(f"{iid}({asin})")
            continue
        vec = emb_data[asin]
        if isinstance(vec, list) and len(vec) > emb_index:
            emb_list.append(vec[emb_index])
        else:
            missing.append(f"{iid}({asin})")

    if missing:
        print(f"Warning: {len(missing)} items missing or invalid: {missing[:5]}...")
    emb_array = np.array(emb_list, dtype=np.float32)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_array)
    tensor = torch.tensor(emb_scaled, dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(tensor, output_path)
    print(f"Saved {tensor.shape} to {output_path}")
    return tensor


def main():
    ap = argparse.ArgumentParser(
        description="Convert embeddings.json to liger RQ-VAE format. Uses only embeddings.json; runs preprocessing if needed."
    )
    ap.add_argument("--embeddings", required=True, help="Path to embeddings.json")
    ap.add_argument("--output", required=True, help="Output path for _embeddings_new.pt")
    ap.add_argument("--emb-index", type=int, default=2, help="Which embedding to use: 0/1/2 for emb1/emb2/emb3")
    ap.add_argument("--dataset", default="Beauty", help="Dataset name (e.g. Beauty, Toys_and_Games)")
    ap.add_argument(
        "--data-type",
        default="Amazon",
        choices=["Amazon", "steam"],
        help="Dataset type",
    )
    ap.add_argument(
        "--raw-path",
        default=None,
        help="Liger raw data path; default: <project>/ID_generation/preprocessing/raw_data/",
    )
    ap.add_argument(
        "--processed-path",
        default=None,
        help="Liger processed data path; default: <project>/ID_generation/preprocessing/processed/",
    )
    ap.add_argument(
        "--features",
        nargs="+",
        default=["title", "price", "brand", "categories"],
        help="Features for id2meta filename",
    )
    ap.add_argument(
        "--prompt-format",
        default="amazon",
        help="Prompt format for id2meta filename",
    )
    args = ap.parse_args()

    raw_path = args.raw_path or os.path.join(
        _PROJECT_ROOT, "ID_generation", "preprocessing", "raw_data"
    )
    processed_path = args.processed_path or os.path.join(
        _PROJECT_ROOT, "ID_generation", "preprocessing", "processed"
    )
    config = {
        "dataset_name": args.dataset,
        "data_type": args.data_type,
        "raw_data_path": raw_path,
        "processed_data_path": processed_path,
        "features_needed": args.features,
        "prompt_format": args.prompt_format,
    }

    id2meta_path, id2item_path = ensure_preprocessing_run(config)
    convert(args.embeddings, id2meta_path, id2item_path, args.output, args.emb_index)


if __name__ == "__main__":
    main()
