import os
import pickle
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
from quantization.utils import set_weight_decay, process_embeddings, calc_cos_sim
from quantization.rqvae.rqvae import RQVAE
from dataloader.amazon_data_processor import AmazonDataProcessor


def check_and_prepare_data(config, dataset_name):
    # 使用相对路径，相对于当前工作目录
    cache_dir = config.get('cache_dir', '../cache')
    category = dataset_name
    
    processed_dir = os.path.join(cache_dir, 'AmazonReviews2014', category, 'processed')
    
    required_files = [
        os.path.join(processed_dir, 'all_item_seqs.json'),
        os.path.join(processed_dir, 'id_mapping.json'),
        os.path.join(processed_dir, 'metadata.sentence.json')
    ]
    
    sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
    sent_emb_path = os.path.join(
        processed_dir,
        f'{os.path.basename(sent_emb_model)}.sent_emb'
    )
    required_files.append(sent_emb_path)
    
    pca_emb_path = None
    if config.get('sent_emb_pca', 0) > 0:
        pca_emb_path = os.path.join(processed_dir, 'final_pca_embeddings.npy')
        required_files.append(pca_emb_path)
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logging.info(f"[TRAINING] Missing files detected, starting data processing pipeline...")
        logging.info(f"[TRAINING] Missing files: {missing_files}")
        
        try:
            logging.info("[TRAINING] Start Data processing pipeline")
            processor = AmazonDataProcessor(
                category=category,
                cache_dir=cache_dir,
                config=config
            )
            processor.run_full_pipeline()
            logging.info("[TRAINING] Data processing pipeline completed")
            
            still_missing = [f for f in required_files if not os.path.exists(f)]
            if still_missing:
                logging.error(f"[ERROR] Files still missing after processing: {still_missing}")
                return False
                
        except Exception as e:
            logging.error(f"[ERROR] Error during data processing: {e}")
            return False
    else:
        logging.info("[TRAINING] All required data files exist, skipping data processing")
        if pca_emb_path:
            logging.info(f"[TRAINING] Found PCA embedding file: {pca_emb_path}")
    
    return True


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

    # x 是输入的 item 表征

    model.to(device)
    # 从RQ-VAE配置中获取训练参数
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

        if (epoch + 1) % 10 == 0:
             logging.info(f"[TRAINING] Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Recon Loss: {train_rec_loss:.4f} | Commit Loss: {train_commit_loss:.4f}")

        if (epoch + 1) % n_eval_interval == 0:
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_dataloader, None, config, flag_eval=True)
            cos_sim_array = calc_cos_sim(model, validationset, config)
            logging.info(f"[VALIDATION] Eval @ Epoch {epoch+1}")
            logging.info(f"[VALIDATION] Validation Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")
            for i in range(config["RQ-VAE"]["num_layers"]):
                logging.info(f"[VALIDATION] Eval Cosine Sim @L{i+1}: {cos_sim_array[i]:.4f}")

    print("[TRAINING] Training complete.")


def generate_codebook(model, item_embedding, dataset_name, config, device, codebook_name="codebook.json"):
    logging.info("[CODEBOOK] Generating Codebook")

    model.to(device)
    model.eval()

    model_config = config["RQ-VAE"]

    all_codes_list = []
    eval_dataset = TensorDataset(item_embedding)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["RQ-VAE"]["batch_size"], shuffle=False)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating Codes for all items"):
            x_batch = batch[0].to(device)
            codes = model.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)

    all_codes_np = np.vstack(all_codes_list)
    logging.info(f"[CODEBOOK] Successfully generated all codes with shape: {all_codes_np.shape}")

    # 修改原因：使用实际的item ID作为码本键，而不是简单的数字字符串
    # 需要从数据集加载实际的item ID列表
    from genrec.datasets import AmazonReviews2014
    
    # 创建正确的配置字典
    from accelerate import Accelerator
    
    # 创建Accelerator对象
    accelerator = Accelerator()
    
    dataset_config = {
        'category': dataset_name,
        'cache_dir': 'cache',
        'accelerator': accelerator,  # 使用Accelerator对象而不是字符串
        'metadata': 'sentence'  # 添加metadata字段
    }
    
    dataset = AmazonReviews2014(dataset_config)
    
    # 获取实际的item ID列表（按顺序）
    # 使用dataset.n_items属性而不是len(dataset)
    n_items = dataset.n_items
    logging.info(f"[CODEBOOK] Dataset has {n_items} items")
    
    # 从item2id映射中获取实际的item ID列表
    actual_item_ids = list(dataset.item2id.keys())
    # 移除[PAD]标记
    if '[PAD]' in actual_item_ids:
        actual_item_ids.remove('[PAD]')
    
    logging.info(f"[CODEBOOK] Found {len(actual_item_ids)} unique items")
    
    # 确保item数量匹配
    if len(actual_item_ids) != all_codes_np.shape[0]:
        logging.warning(f"[CODEBOOK] Warning: Item count mismatch. Codes: {all_codes_np.shape[0]}, Items: {len(actual_item_ids)}")
        # 使用较小的数量
        min_count = min(len(actual_item_ids), all_codes_np.shape[0])
        actual_item_ids = actual_item_ids[:min_count]
        all_codes_np = all_codes_np[:min_count]
    
    item_to_codes = {
        actual_item_ids[item_id]: codes.tolist()
        for item_id, codes in enumerate(all_codes_np)
    }

    codebook_dir = os.path.join("cache", "AmazonReviews2014", dataset_name, "codebook")
    os.makedirs(codebook_dir, exist_ok=True)
    codebook_path = os.path.join(codebook_dir, codebook_name)

    with open(codebook_path, 'w') as f:
        json.dump(item_to_codes, f)

    logging.info(f"[CODEBOOK] Codebook successfully saved to: {codebook_path}")
    return codebook_path


def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE using RPG data")
    parser.add_argument('--config', type=str, default='quantization/rqvae_config.yaml', help='Path to the configuration file')
    parser.add_argument('--use_image', action='store_true', help='Use image embeddings instead of text embeddings for codebook generation')
    parser.add_argument('--use_multimodal', action='store_true', help='Use multimodal embeddings (text + image) for codebook generation')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset']['name']

    log_dir = os.path.join('logs', 'rqvae', dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='a'
    )
    root_logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 从配置文件读取device
    device_name = config.get('training', {}).get('device', 'cuda:0')
    device = torch.device(device_name)
    
    logging.info("[TRAINING] Step 0: Check and prepare data")
    data_config = config.get('data_processing', {})
    if not check_and_prepare_data(data_config, dataset_name):
        logging.error("[ERROR] Data preparation failed, exiting program")
        return
    
    # 根据参数选择使用哪种嵌入
    if args.use_multimodal:
        logging.info("[TRAINING] Using multimodal embeddings (text + image) for codebook generation")
        from quantization.utils import process_multimodal_embeddings
        item_embedding = process_multimodal_embeddings(
            config=config, device=device, id2meta_file=None, embedding_save_path=None
        )
    elif args.use_image:
        logging.info("[TRAINING] Using image embeddings for codebook generation")
        from quantization.utils import process_image_embeddings
        item_embedding = process_image_embeddings(
            config=config, device=device, id2meta_file=None, embedding_save_path=None
        )
    else: # 默认是纯文本
        logging.info("[TRAINING] Using text embeddings for codebook generation")
        item_embedding = process_embeddings(
            config=config, device=device, id2meta_file=None, embedding_save_path=None
        )
    
    if item_embedding is None:
        logging.error("[ERROR] Failed to load or process item embeddings. Exiting.")
        return

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
    train_rqvae(rqvae, item_embedding.cpu().numpy(), device, config)

    logging.info("[TRAINING] Training complete, starting final collision detection")
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

    logging.info("[COLLISION] Final Collision Detection Results")
    logging.info(f"[COLLISION] Total Items: {total_items}")
    logging.info(f"[COLLISION] Unique Codes: {unique_items}")
    logging.info(f"[COLLISION] Duplicated Items: {num_duplicates}")
    logging.info(f"[COLLISION] Final Collision Rate: {collision_rate:.4%}")

    model_config = config["RQ-VAE"]
    
    # 根据使用的嵌入类型命名模型
    if args.use_multimodal:
        model_name = f"rqvae-{dataset_name}-multimodal.pth"
        codebook_name = "codebook-multimodal.json"
    elif args.use_image:
        model_name = f"rqvae-{dataset_name}-image.pth"
        codebook_name = "codebook-image.json"
    else:
        model_name = f"rqvae-{dataset_name}-text.pth"
        codebook_name = "codebook-text.json"
    
    save_dir = os.path.join("ckpt", dataset_name, "rqvae")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(rqvae.state_dict(), save_path)
    logging.info(f"[TRAINING] Training complete! Final model saved to: {save_path}")

    # 基于训练好的模型 生成 codebook
    codebook_path = generate_codebook(rqvae, item_embedding, dataset_name, config, device, codebook_name)
    logging.info(f"[TRAINING] Full process finished! Model and Codebook generated.")
    logging.info(f"[TRAINING] Model Path: {save_path}")
    logging.info(f"[TRAINING] Codebook Path: {codebook_path}")


if __name__ == '__main__':
    main()