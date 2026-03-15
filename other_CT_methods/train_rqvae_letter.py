
import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from quantization.utils import set_weight_decay, calc_cos_sim
from quantization.rqvae.rqvae import RQVAE
from train_rqvae_from_emb import generate_codebook


def load_emb1_and_sasrec(processed_dir, emb_index=0):
    """
    Load emb1 and SASRec, align by id_mapping, return (item_ids, emb1_np, sasrec_np).
    """
    path_emb = os.path.join(processed_dir, "embeddings.json")
    path_sasrec = os.path.join(processed_dir, "sasrec_embeddings.json")
    path_id_mapping = os.path.join(processed_dir, "id_mapping.json")
    for p in (path_emb, path_sasrec, path_id_mapping):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")

    with open(path_emb, "r", encoding="utf-8") as f:
        text_emb = json.load(f)
    with open(path_sasrec, "r", encoding="utf-8") as f:
        sasrec_emb = json.load(f)
    with open(path_id_mapping, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    id2item = id_mapping["id2item"]
    itemnum = len(id2item) - 1

    def get_text_emb(original_id):
        v = text_emb.get(original_id) or text_emb.get(str(original_id))
        if v is None or not isinstance(v, list) or len(v) <= emb_index:
            return None
        return v[emb_index]

    item_ids, emb1_list, sasrec_list = [], [], []
    for i in range(1, itemnum + 1):
        original_id = id2item[i]
        sasrec = sasrec_emb.get(str(i))
        emb1 = get_text_emb(original_id)
        if sasrec is None or emb1 is None:
            continue
        item_ids.append(original_id)
        emb1_list.append(np.asarray(emb1, dtype=np.float32))
        sasrec_list.append(np.asarray(sasrec, dtype=np.float32))

    if not emb1_list:
        raise ValueError("No items exist in both emb1 and sasrec")
    return item_ids, np.stack(emb1_list), np.stack(sasrec_list)


def infonce_loss(recon_x, h_proj, tau=0.07):
    """InfoNCE: -log(exp(<z_i,h_i>/tau) / sum_j exp(<z_i,h_j>/tau))"""
    B = recon_x.size(0)
    logits = (recon_x @ h_proj.T) / tau
    labels = torch.arange(B, device=recon_x.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def train_epoch_letter(model, projection, dataloader, optimizer, config, device, cf_weight, tau):
    model.train()
    beta = config["RQ-VAE"]["beta"]
    total_loss = total_rec = total_commit = total_cf = 0.0

    for batch in dataloader:
        x_batch = batch[0].to(device)
        h_batch = batch[1].to(device)
        optimizer.zero_grad()

        recon_x, commitment_loss, _ = model(x_batch)
        L_recon = F.mse_loss(recon_x, x_batch, reduction="mean")
        h_proj = projection(h_batch)
        L_cf = infonce_loss(recon_x, h_proj, tau)
        loss = L_recon + beta * commitment_loss + cf_weight * L_cf

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rec += L_recon.item()
        total_commit += commitment_loss.item()
        total_cf += L_cf.item()

    n = len(dataloader)
    return total_loss / n, total_rec / n, total_commit / n, total_cf / n


def train_epoch_letter_eval(model, projection, dataloader, config, device, cf_weight, tau):
    model.eval()
    beta = config["RQ-VAE"]["beta"]
    total_loss = total_rec = total_commit = total_cf = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0].to(device)
            h_batch = batch[1].to(device)
            recon_x, commitment_loss, _ = model(x_batch)
            L_recon = F.mse_loss(recon_x, x_batch, reduction="mean")
            h_proj = projection(h_batch)
            L_cf = infonce_loss(recon_x, h_proj, tau)
            loss = L_recon + beta * commitment_loss + cf_weight * L_cf
            total_loss += loss.item()
            total_rec += L_recon.item()
            total_commit += commitment_loss.item()
            total_cf += L_cf.item()

    n = len(dataloader)
    return total_loss / n, total_rec / n, total_commit / n, total_cf / n


def main():
    parser = argparse.ArgumentParser(
        description="Train RQ-VAE with emb1 + InfoNCE contrastive loss, output codebook-letter.json"
    )
    parser.add_argument("--config", type=str, default="quantization/rqvae_config.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--cf_weight", type=float, default=0.02, help="InfoNCE loss weight(set the recommended values reported in the Letter paper as the default)")
    parser.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(SCRIPT_DIR, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    category = args.category or config["dataset"]["name"]
    processed_dir = args.processed_dir or os.path.join(
        SCRIPT_DIR, "cache", "AmazonReviews2014", category, "processed"
    )
    codebook_dir = os.path.join(SCRIPT_DIR, "cache", "AmazonReviews2014", category, "codebook")
    codebook_path = os.path.join(codebook_dir, "codebook-letter.json")

    log_dir = os.path.join(SCRIPT_DIR, "logs", "rqvae", category)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_letter_{ts}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        filename=log_file, filemode="a")
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(ch)

    device = torch.device(config.get("training", {}).get("device", "cuda:0"))

    logging.info("[LETTER] Loading emb1 and SASRec...")
    item_ids, emb1_np, sasrec_np = load_emb1_and_sasrec(processed_dir, emb_index=0)
    input_size = emb1_np.shape[1]
    sasrec_dim = sasrec_np.shape[1]

    rqvae_config = config["RQ-VAE"]
    model = RQVAE(
        input_size,
        rqvae_config["hidden_dim"],
        rqvae_config["latent_dim"],
        rqvae_config["num_layers"],
        rqvae_config["code_book_size"],
        rqvae_config["dropout"],
        latent_loss_weight=rqvae_config["beta"],
    )
    projection = nn.Linear(sasrec_dim, input_size).to(device)

    emb1_train, emb1_val, sasrec_train, sasrec_val = train_test_split(
        emb1_np, sasrec_np, test_size=0.05, random_state=42
    )
    train_ds = TensorDataset(torch.from_numpy(emb1_train), torch.from_numpy(sasrec_train))
    val_ds = TensorDataset(torch.from_numpy(emb1_val), torch.from_numpy(sasrec_val))
    batch_size = rqvae_config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=rqvae_config["lr"],
        weight_decay=rqvae_config.get("weight_decay", 0.0),
    )
    if optim.param_groups[0].get("weight_decay"):
        set_weight_decay(optim, rqvae_config.get("weight_decay", 0.0))

    num_epochs = rqvae_config["epochs"]
    n_eval = 100

    logging.info("[LETTER] Training RQ-VAE with InfoNCE contrastive loss...")
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc="RQ-VAE+InfoNCE"):
        train_loss, tr, tc, tcf = train_epoch_letter(
            model, projection, train_loader, optim, config, device, args.cf_weight, args.tau
        )
        if (epoch + 1) % n_eval == 0:
            val_loss, vr, vc, vcf = train_epoch_letter_eval(
                model, projection, val_loader, config, device, args.cf_weight, args.tau
            )
            val_x = torch.from_numpy(emb1_val).to(device)
            cos_arr = calc_cos_sim(model, val_x, config)
            logging.info(f"[LETTER] Epoch {epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_cos@L1={cos_arr[0]:.4f}")

    print("[LETTER] Training complete.")

    model.eval()
    save_dir = os.path.join(SCRIPT_DIR, "ckpt", category, "rqvae")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.model_name or f"rqvae-{category}-letter.pth")
    torch.save({"model": model.state_dict(), "projection": projection.state_dict()}, save_path)
    logging.info(f"[LETTER] Model saved: {save_path}")

    item_embedding = torch.from_numpy(emb1_np)
    generate_codebook(model, item_embedding, item_ids, config, device, codebook_path)
    logging.info(f"[LETTER] Codebook saved: {codebook_path}")


if __name__ == "__main__":
    main()
