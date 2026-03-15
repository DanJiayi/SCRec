import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from quantization.utils import set_weight_decay, calc_cos_sim
from quantization.rqvae.rqvae import RQVAE
from train_rqvae_from_emb import train_rqvae, generate_codebook


def load_mmg_embeddings(processed_dir, emb_index=0):
    """
    Load embeddings.json (take the emb_index-th), sasrec_embeddings.json, and id_mapping.json
    from processed_dir, align by id2item, concatenate, and return (item_ids, concat_embedding_np).
    item_ids are original ids (consistent with codebook-text for downstream use), ordered same as matrix rows.
    """
    path_emb = os.path.join(processed_dir, "embeddings.json")
    path_sasrec = os.path.join(processed_dir, "sasrec_embeddings.json")
    path_id_mapping = os.path.join(processed_dir, "id_mapping.json")

    for p in (path_emb, path_sasrec, path_id_mapping):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(path_emb, "r", encoding="utf-8") as f:
        text_emb = json.load(f)
    with open(path_sasrec, "r", encoding="utf-8") as f:
        sasrec_emb = json.load(f)
    with open(path_id_mapping, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    id2item = id_mapping["id2item"]
    itemnum = len(id2item) - 1  # 0 is [PAD]

    def get_text_emb(original_id):
        v = text_emb.get(original_id) or text_emb.get(str(original_id))
        if v is None or not isinstance(v, list) or len(v) <= emb_index:
            return None
        return v[emb_index]

    item_ids = []
    concat_list = []

    for i in range(1, itemnum + 1):
        original_id = id2item[i]
        sasrec = sasrec_emb.get(str(i))
        text_emb_vec = get_text_emb(original_id)
        if sasrec is None or text_emb_vec is None:
            continue
        text_emb_vec = np.asarray(text_emb_vec, dtype=np.float32)
        sasrec = np.asarray(sasrec, dtype=np.float32)
        concat_list.append(np.concatenate([text_emb_vec, sasrec], axis=-1))
        item_ids.append(original_id)

    if not concat_list:
        raise ValueError("No items exist in both embeddings.json and sasrec_embeddings.json, cannot concatenate")

    item_embedding_np = np.stack(concat_list, axis=0)
    return item_ids, item_embedding_np


def main():
    parser = argparse.ArgumentParser(
        description="Train RQ-VAE on concatenated text emb1 + SASRec emb, output codebook-mmg.json"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quantization/rqvae_config.yaml",
        help="RQ-VAE config file path (relative to SCRec root)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Dataset category, e.g. Beauty; if not set, use config dataset.name",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Processed directory; if not set, use cache/AmazonReviews2014/{category}/processed",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Saved model filename; if not set, use rqvae-{category}-mmg.pth",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    category = args.category or config["dataset"]["name"]
    processed_dir = args.processed_dir or os.path.join(
        SCRIPT_DIR, "cache", "AmazonReviews2014", category, "processed"
    )
    cache_dir = os.path.join(SCRIPT_DIR, "cache", "AmazonReviews2014", category)
    codebook_dir = os.path.join(cache_dir, "codebook")
    codebook_path = os.path.join(codebook_dir, "codebook-mmg.json")

    log_dir = os.path.join(SCRIPT_DIR, "logs", "rqvae", category)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_mmg_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",
    )
    root_logger = logging.getLogger()
    if not root_logger.handlers or not any(h.stream.name == "<stderr>" for h in root_logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(ch)

    device_name = config.get("training", {}).get("device", "cuda:0")
    device = torch.device(device_name)

    logging.info("[MMG] Loading text emb1 + SASRec embeddings and aligning by id_mapping...")
    item_ids, item_embedding_np = load_mmg_embeddings(processed_dir, emb_index=0)
    input_size = item_embedding_np.shape[1]
    logging.info(f"[MMG] Concatenated embedding shape: {item_embedding_np.shape}, input_size={input_size}")

    model_config = config["RQ-VAE"]
    rqvae = RQVAE(
        input_size,
        model_config["hidden_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["code_book_size"],
        model_config["dropout"],
        latent_loss_weight=model_config["beta"],
    )

    logging.info("[MMG] Training RQ-VAE on concatenated (text_emb1 + sasrec) embeddings...")
    train_rqvae(rqvae, item_embedding_np, device, config)

    rqvae.to(device)
    rqvae.eval()
    item_embedding_tensor = torch.from_numpy(item_embedding_np)

    save_dir = os.path.join(SCRIPT_DIR, "ckpt", category, "rqvae")
    os.makedirs(save_dir, exist_ok=True)
    model_name = args.model_name or f"rqvae-{category}-mmg.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(rqvae.state_dict(), save_path)
    logging.info(f"[MMG] Model saved to {save_path}")

    os.makedirs(codebook_dir, exist_ok=True)
    generate_codebook(rqvae, item_embedding_tensor, item_ids, config, device, codebook_path)
    logging.info(f"[MMG] Codebook saved to {codebook_path} (format same as codebook-text.json)")


if __name__ == "__main__":
    main()
