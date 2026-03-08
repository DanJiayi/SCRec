import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
import argparse
import logging
from datetime import datetime

# Relative imports
from quantization.utils import set_weight_decay, calc_cos_sim
from quantization.rqvae.rqvae import RQVAE


def load_item_embeddings_from_json(json_path, emb_index=2):
    with open(json_path, "r") as f:
        data = json.load(f)

    item_ids = []
    embeddings = []

    for item_id, emb_list in data.items():
        if not isinstance(emb_list, list) or len(emb_list) <= emb_index:
            raise ValueError(f"Item {item_id} does not have emb index {emb_index}: {type(emb_list)}")
        emb = emb_list[emb_index]
        item_ids.append(item_id)
        embeddings.append(emb)

    item_embedding = np.asarray(embeddings, dtype=np.float32)
    if item_embedding.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {item_embedding.shape}")

    return item_ids, item_embedding


def train_epoch(model, dataloader, optimizer, config, flag_eval=False):
    model.eval() if flag_eval else model.train()
    beta = config["RQ-VAE"]["beta"]
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0

    for i_batch, batch in enumerate(dataloader):
        x_batch = batch[0]
        if not flag_eval:
            optimizer.zero_grad()

        with torch.set_grad_enabled(not flag_eval):
            recon_x, commitment_loss, indices = model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
            loss = reconstruction_mse_loss + beta * commitment_loss

        if not flag_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_rec_loss += reconstruction_mse_loss.item()
        total_commit_loss += commitment_loss.item()

    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_commit_loss / len(dataloader)


def train_rqvae(model, x, device, config):
    model.to(device)
    rqvae_config = config["RQ-VAE"]
    batch_size = rqvae_config["batch_size"]
    num_epochs = rqvae_config["epochs"]
    lr = rqvae_config["lr"]

    optimizer = getattr(torch.optim, rqvae_config["optimizer"])(model.parameters(), lr=lr)
    if "weight_decay" in optimizer.param_groups[0]:
        set_weight_decay(optimizer, rqvae_config["weight_decay"])

    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    trainset, validationset = torch.Tensor(trainset).to(device), torch.Tensor(validationset).to(device)
    train_dataset = TensorDataset(trainset)
    val_dataset = TensorDataset(validationset)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_eval_interval = 100
    for epoch in tqdm(range(num_epochs), desc="Training RQ-VAE"):
        train_loss, train_rec_loss, train_commit_loss = train_epoch(model, dataloader, optimizer, config)

        # if (epoch + 1) % 10 == 0:
        #     logging.info(
        #         f"[TRAINING] Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | "
        #         f"Recon Loss: {train_rec_loss:.4f} | Commit Loss: {train_commit_loss:.4f}"
        #     )

        if (epoch + 1) % n_eval_interval == 0:
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_dataloader, None, config, flag_eval=True)
            cos_sim_array = calc_cos_sim(model, validationset, config)
            # logging.info(f"[VALIDATION] Eval @ Epoch {epoch+1}")
            # logging.info(f"[VALIDATION] Validation Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")
            # for i in range(config["RQ-VAE"]["num_layers"]):
            #     logging.info(f"[VALIDATION] Eval Cosine Sim @L{i+1}: {cos_sim_array[i]:.4f}")

    print("[TRAINING] Training complete.")


def generate_codebook(model, item_embedding, item_ids, config, device, codebook_path):
    logging.info("[CODEBOOK] Generating Codebook")

    model.to(device)
    model.eval()

    all_codes_list = []
    eval_dataset = TensorDataset(item_embedding)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["RQ-VAE"]["batch_size"], shuffle=False)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating Codes for all items"):
            x_batch = batch[0].to(device)
            codes = model.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)

    all_codes_np = np.vstack(all_codes_list)
    # logging.info(f"[CODEBOOK] Successfully generated all codes with shape: {all_codes_np.shape}")

    if len(item_ids) != all_codes_np.shape[0]:
        # logging.warning(
        #     f"[CODEBOOK] Warning: Item count mismatch. Codes: {all_codes_np.shape[0]}, Items: {len(item_ids)}"
        # )
        min_count = min(len(item_ids), all_codes_np.shape[0])
        item_ids = item_ids[:min_count]
        all_codes_np = all_codes_np[:min_count]

    item_to_codes = {
        item_ids[item_id]: codes.tolist()
        for item_id, codes in enumerate(all_codes_np)
    }

    os.makedirs(os.path.dirname(codebook_path), exist_ok=True)
    with open(codebook_path, "w") as f:
        json.dump(item_to_codes, f)

    logging.info(f"[CODEBOOK] Codebook successfully saved to: {codebook_path}")
    return codebook_path


def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE from emb JSON and generate item codes")
    parser.add_argument("--config", type=str, default="quantization/rqvae_config.yaml", help="Path to the config file")
    # parser.add_argument(
    #     "--input_json",
    #     type=str,
    #     default="cache/AmazonReviews2014/Beauty/processed/embeddings.json",
    #     help="Path to JSON with item_id -> [emb1, emb2, emb3]",
    # )
    parser.add_argument("--emb_index", type=int, default=2, help="Embedding index to use (emb3 -> 2)")
    parser.add_argument("--codebook_name", type=str, default="codebook-text.json", help="Output codebook filename")
    parser.add_argument("--model_name", type=str, default=None, help="Override model filename")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]

    log_dir = os.path.join("logs", "rqvae", dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="a",
    )
    root_logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    device_name = config.get("training", {}).get("device", "cuda:0")
    device = torch.device(device_name)

    input_json = f"cache/AmazonReviews2014/{dataset_name}/processed/embeddings.json"
    # logging.info(f"[DATA] Loading embeddings from: {input_json} (index {args.emb_index})")
    item_ids, item_embedding_np = load_item_embeddings_from_json(input_json, args.emb_index)
    item_embedding = torch.from_numpy(item_embedding_np)

    model_config = config["RQ-VAE"]
    input_size = item_embedding.shape[1]
    rqvae = RQVAE(
        input_size,
        model_config["hidden_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["code_book_size"],
        model_config["dropout"],
        latent_loss_weight=model_config["beta"],
    )

    logging.info(f"[TRAINING] Starting RQ-VAE model training for dataset: {dataset_name}...")
    train_rqvae(rqvae, item_embedding_np, device, config)

    # logging.info("[TRAINING] Training complete, starting final collision detection")
    rqvae.to(device)
    rqvae.eval()

    all_codes_list = []
    eval_dataset = TensorDataset(item_embedding)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["RQ-VAE"]["batch_size"], shuffle=False)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating codes for all items"):
            x_batch = batch[0].to(device)
            codes = rqvae.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)

    all_codes_np = np.vstack(all_codes_list)
    all_codes_str = ["-".join(map(str, row)) for row in all_codes_np]

    total_items = len(all_codes_str)
    unique_items = len(set(all_codes_str))
    num_duplicates = total_items - unique_items
    collision_rate = num_duplicates / total_items if total_items > 0 else 0

    # logging.info("[COLLISION] Final Collision Detection Results")
    # logging.info(f"[COLLISION] Total Items: {total_items}")
    # logging.info(f"[COLLISION] Unique Codes: {unique_items}")
    # logging.info(f"[COLLISION] Duplicated Items: {num_duplicates}")
    # logging.info(f"[COLLISION] Final Collision Rate: {collision_rate:.4%}")

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"rqvae-{dataset_name}-emb{args.emb_index + 1}.pth"

    save_dir = os.path.join("ckpt", dataset_name, "rqvae")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(rqvae.state_dict(), save_path)
    logging.info(f"[TRAINING] Training complete! Final model saved to: {save_path}")

    codebook_dir = os.path.join("cache", "AmazonReviews2014", dataset_name, "codebook")
    codebook_path = os.path.join(codebook_dir, args.codebook_name)
    generate_codebook(rqvae, item_embedding, item_ids, config, device, codebook_path)
    logging.info("[TRAINING] Full process finished! Model and Codebook generated.")
    logging.info(f"[TRAINING] Model Path: {save_path}")
    logging.info(f"[TRAINING] Codebook Path: {codebook_path}")


if __name__ == "__main__":
    main()
