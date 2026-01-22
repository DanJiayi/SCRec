import json
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import sys

def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        data = data[:, 0, :]
    ids = model.get_codes(data).cpu().numpy()
    max_item_calculate = 1000
    cos_sim_array = np.zeros(config["RQ-VAE"]["num_layers"])

    for n_prefix in range(1, config["RQ-VAE"]["num_layers"] + 1):
        unique_prefix = np.unique(ids[:, :n_prefix], axis=0)
        this_level_cos_sim_within_cluster = []

        for this_level_prefix in unique_prefix:
            mask = (ids[:, :n_prefix] == this_level_prefix).all(axis=1)
            this_cluster = data[mask].cpu()
            this_cluster_num = this_cluster.shape[0]

            if this_cluster_num > 1:
                indice = torch.randperm(this_cluster_num)[:max_item_calculate]
                cos_sim = F.cosine_similarity(
                    this_cluster[indice, :, None],
                    this_cluster.t()[None, :, indice]
                )
                cos_sim_sum = torch.tril(cos_sim, diagonal=-1).sum()
                normalization_factor = (this_cluster_num - 1) * this_cluster_num / 2
                this_level_cos_sim_within_cluster.append(
                    cos_sim_sum.item() / normalization_factor
                )

        if this_level_cos_sim_within_cluster:
            cos_sim_array[n_prefix - 1] = np.mean(this_level_cos_sim_within_cluster)

    return cos_sim_array


def process_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    final_output_path = os.path.join("cache", type, category, "processed", "final_pca_embeddings.npy")

    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Embedding file not found: {final_output_path}")

    np_array = np.load(final_output_path)
    tensor = torch.from_numpy(np_array).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Loaded embeddings from '{final_output_path}', shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def process_image_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    """
    处理图片嵌入，用于基于图片生成codebook
    """
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    
    # 图片嵌入文件路径
    img_emb_dir = os.path.join("cache", type, category, "processed", "image_embeddings")
    img_emb_file = os.path.join(img_emb_dir, "image_embeddings_clip-vit-base-patch32.npy")
    img_emb_mapping_file = os.path.join(img_emb_dir, "image_embeddings_clip-vit-base-patch32_mapping.json")
    
    # 检查图片嵌入文件是否存在
    if not os.path.exists(img_emb_file):
        raise FileNotFoundError(f"Image embedding file not found: {img_emb_file}")
    
    if not os.path.exists(img_emb_mapping_file):
        raise FileNotFoundError(f"Image embedding mapping file not found: {img_emb_mapping_file}")
    
    # 加载图片嵌入
    img_embeddings = np.load(img_emb_file)
    
    # 加载映射文件
    with open(img_emb_mapping_file, 'r') as f:
        img_mapping = json.load(f)
    
    print(f"[QUANTIZATION] Loaded image embeddings from '{img_emb_file}', shape={img_embeddings.shape}")
    print(f"[QUANTIZATION] Image mapping contains {len(img_mapping)} items")
    
    # 检查是否需要PCA降维
    img_emb_pca = config.get("data_processing", {}).get("img_emb_pca", 512)
    if img_emb_pca > 0 and img_emb_pca < img_embeddings.shape[1]:
        print(f"[QUANTIZATION] Applying PCA to image embeddings: {img_embeddings.shape[1]} -> {img_emb_pca}")
        pca = PCA(n_components=img_emb_pca, whiten=True)
        img_embeddings = pca.fit_transform(img_embeddings)
        
        # 保存PCA后的图片嵌入
        pca_img_emb_path = os.path.join(img_emb_dir, "final_pca_image_embeddings.npy")
        np.save(pca_img_emb_path, img_embeddings)
        print(f"[QUANTIZATION] PCA image embeddings saved to: {pca_img_emb_path}")
    
    # 转换为tensor
    tensor = torch.from_numpy(img_embeddings).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Image embeddings tensor shape: {tensor.shape}, dtype={tensor.dtype}")
    
    return tensor


def process_multimodal_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    """
    处理图片和文本的拼接向量，用于构建多模态codebook
    """
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    
    print(f"[QUANTIZATION] Processing multimodal embeddings (text + image) for {category}")
    
    # 1. 加载文本嵌入
    text_emb_file = os.path.join("cache", type, category, "processed", "text-embedding-3-large.sent_emb")
    if not os.path.exists(text_emb_file):
        raise FileNotFoundError(f"Text embedding file not found: {text_emb_file}")
    
    text_embeddings = np.fromfile(text_emb_file, dtype=np.float32).reshape(-1, 512)
    print(f"[QUANTIZATION] Loaded text embeddings: {text_embeddings.shape}")
    
    # 2. 加载图片嵌入
    img_emb_dir = os.path.join("cache", type, category, "processed", "image_embeddings")
    img_emb_file = os.path.join(img_emb_dir, "image_embeddings_clip-vit-base-patch32.npy")
    img_emb_mapping_file = os.path.join(img_emb_dir, "image_embeddings_clip-vit-base-patch32_mapping.json")
    
    if not os.path.exists(img_emb_file):
        raise FileNotFoundError(f"Image embedding file not found: {img_emb_file}")
    
    if not os.path.exists(img_emb_mapping_file):
        raise FileNotFoundError(f"Image embedding mapping file not found: {img_emb_mapping_file}")
    
    image_embeddings = np.load(img_emb_file)
    with open(img_emb_mapping_file, 'r') as f:
        image_mapping = json.load(f)
    
    print(f"[QUANTIZATION] Loaded image embeddings: {image_embeddings.shape}")
    print(f"[QUANTIZATION] Image mapping contains {len(image_mapping)} items")
    
    # 3. 加载数据集映射
    id_mapping_file = os.path.join("cache", type, category, "processed", "id_mapping.json")
    if not os.path.exists(id_mapping_file):
        raise FileNotFoundError(f"ID mapping file not found: {id_mapping_file}")
    
    with open(id_mapping_file, 'r') as f:
        id_mapping = json.load(f)
    
    n_items = len(id_mapping['id2item'])
    print(f"[QUANTIZATION] Dataset contains {n_items} items")
    
    # 4. 创建多模态嵌入矩阵
    # 获取配置中的维度设置
    text_dim = config.get("data_processing", {}).get("sent_emb_pca", 512)
    image_dim = config.get("data_processing", {}).get("img_emb_pca", 256)
    
    # 如果文本嵌入维度不匹配，进行PCA
    if text_embeddings.shape[1] != text_dim:
        print(f"[QUANTIZATION] Applying PCA to text embeddings: {text_embeddings.shape[1]} -> {text_dim}")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=text_dim, whiten=True)
        text_embeddings = pca.fit_transform(text_embeddings)
    
    # 如果图片嵌入维度不匹配，进行PCA
    if image_embeddings.shape[1] != image_dim:
        print(f"[QUANTIZATION] Applying PCA to image embeddings: {image_embeddings.shape[1]} -> {image_dim}")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=image_dim, whiten=True)
        image_embeddings = pca.fit_transform(image_embeddings)
    
    # 5. 创建拼接的多模态嵌入
    multimodal_embeddings = np.zeros((n_items, text_dim + image_dim))
    
    # 统计信息
    text_available = 0
    image_available = 0
    both_available = 0
    
    for item_id in range(1, n_items + 1):  # 1-based item IDs
        item_id_str = str(item_id)
        
        # 获取文本嵌入
        if item_id - 1 < text_embeddings.shape[0]:
            text_emb = text_embeddings[item_id - 1]
            text_available += 1
        else:
            # 如果没有文本嵌入，使用零向量
            text_emb = np.zeros(text_dim)
        
        # 获取图片嵌入
        if item_id_str in image_mapping:
            image_idx = image_mapping[item_id_str]
            if image_idx < image_embeddings.shape[0]:
                image_emb = image_embeddings[image_idx]
                image_available += 1
            else:
                image_emb = np.zeros(image_dim)
        else:
            # 如果没有图片嵌入，使用零向量
            image_emb = np.zeros(image_dim)
        
        # 拼接文本和图片嵌入
        multimodal_embeddings[item_id - 1] = np.concatenate([text_emb, image_emb])
        
        if item_id - 1 < text_embeddings.shape[0] and item_id_str in image_mapping:
            both_available += 1
    
    print(f"[QUANTIZATION] Multimodal embedding statistics:")
    print(f"   Text available: {text_available}/{n_items} ({text_available/n_items:.2%})")
    print(f"   Image available: {image_available}/{n_items} ({image_available/n_items:.2%})")
    print(f"   Both available: {both_available}/{n_items} ({both_available/n_items:.2%})")
    print(f"   Final multimodal embeddings shape: {multimodal_embeddings.shape}")
    
    # 6. 保存多模态嵌入（可选）
    if embedding_save_path is None:
        embedding_save_path = os.path.join("cache", type, category, "processed", "multimodal_embeddings.npy")
    
    np.save(embedding_save_path, multimodal_embeddings)
    print(f"[QUANTIZATION] Multimodal embeddings saved to: {embedding_save_path}")
    
    # 7. 转换为tensor并返回
    tensor = torch.from_numpy(multimodal_embeddings).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Multimodal embeddings tensor shape: {tensor.shape}, dtype={tensor.dtype}")
    
    return tensor


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay