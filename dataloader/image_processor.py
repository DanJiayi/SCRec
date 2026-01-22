#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available, will use random vectors for image embeddings")

class ImageProcessor:
    """
    图片处理模块，负责生成图片嵌入
    当图片不存在时，生成随机向量作为占位符
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.category = config.get('category', 'Beauty')
        self.cache_dir = config.get('cache_dir', 'cache')
        self.images_dir = os.path.join(self.cache_dir, 'AmazonReviews2014', self.category, 'images')
        self.img_emb_dir = os.path.join(self.cache_dir, 'AmazonReviews2014', self.category, 'processed', 'image_embeddings')
        
        # 图片处理配置
        self.img_emb_model = config.get('img_emb_model', 'clip-vit-base-patch32')
        self.img_emb_dim = config.get('img_emb_dim', 512)
        self.img_emb_batch_size = config.get('img_emb_batch_size', 32)
        self.img_emb_pca = config.get('img_emb_pca', 256)
        self.image_size = config.get('image_size', 224)
        self.max_images_per_item = config.get('max_images_per_item', 5)
        
        # 创建目录
        os.makedirs(self.img_emb_dir, exist_ok=True)
        
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
        # 初始化CLIP模型（如果可用）
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE and self.img_emb_model.startswith('clip'):
            self._init_clip_model()
    
    def _init_clip_model(self):
        """初始化CLIP模型"""
        try:
            model_name = f"openai/{self.img_emb_model}"
            # 首先尝试从本地缓存加载
            local_cache_path = f"model_cache/{model_name}"
            if os.path.exists(local_cache_path):
                self.clip_model = CLIPModel.from_pretrained(local_cache_path)
                self.clip_processor = CLIPProcessor.from_pretrained(local_cache_path)
                self.logger.info(f"CLIP model {model_name} loaded successfully from local cache")
            else:
                # 如果本地没有，尝试从Hugging Face加载
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                self.logger.info(f"CLIP model {model_name} loaded successfully from Hugging Face")
        except Exception as e:
            self.logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def get_image_paths(self, item_id: str) -> List[str]:
        """获取商品的所有图片路径"""
        item_dir = os.path.join(self.images_dir, item_id)
        if not os.path.exists(item_dir):
            return []
        
        image_paths = []
        for filename in os.listdir(item_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(item_dir, filename))
        
        # 限制每个商品的图片数量
        return image_paths[:self.max_images_per_item]
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """处理单张图片，返回嵌入向量"""
        try:
            if self.clip_model is not None and self.clip_processor is not None:
                # 使用CLIP处理图片
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.image_size, self.image_size))
                
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                
                return image_features.cpu().numpy().flatten()
            else:
                # 生成随机向量作为占位符
                return np.random.normal(0, 1, self.img_emb_dim)
                
        except Exception as e:
            self.logger.warning(f"Failed to process image {image_path}: {e}")
            # 返回随机向量作为占位符
            return np.random.normal(0, 1, self.img_emb_dim)
    
    def process_item_images(self, item_id: str) -> np.ndarray:
        """处理商品的所有图片，返回平均嵌入向量"""
        image_paths = self.get_image_paths(item_id)
        
        if not image_paths:
            # 没有图片，返回随机向量
            return np.random.normal(0, 1, self.img_emb_dim)
        
        embeddings = []
        for image_path in image_paths:
            emb = self.process_single_image(image_path)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            # 所有图片处理失败，返回随机向量
            return np.random.normal(0, 1, self.img_emb_dim)
        
        # 返回平均嵌入向量
        return np.mean(embeddings, axis=0)
    
    def generate_image_embeddings(self, item_ids: List[str]) -> Dict[str, np.ndarray]:
        """为所有商品生成图片嵌入"""
        self.logger.info(f"Generating image embeddings for {len(item_ids)} items...")
        
        image_embeddings = {}
        
        for item_id in tqdm(item_ids, desc="Processing images"):
            embedding = self.process_item_images(item_id)
            image_embeddings[item_id] = embedding
        
        return image_embeddings
    
    def save_image_embeddings(self, image_embeddings: Dict[str, np.ndarray], filename: str = None):
        """保存图片嵌入到文件"""
        if filename is None:
            filename = f"image_embeddings_{self.img_emb_model}.npy"
        
        filepath = os.path.join(self.img_emb_dir, filename)
        
        # 转换为numpy数组格式
        item_ids = list(image_embeddings.keys())
        embeddings = np.array([image_embeddings[item_id] for item_id in item_ids])
        
        # 保存嵌入向量
        np.save(filepath, embeddings)
        
        # 保存item_id映射
        mapping_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_{self.img_emb_model}_mapping.json")
        with open(mapping_filepath, 'w') as f:
            json.dump({str(i): item_id for i, item_id in enumerate(item_ids)}, f)
        
        self.logger.info(f"Image embeddings saved to {filepath}")
        self.logger.info(f"Embedding shape: {embeddings.shape}")
        
        return filepath, mapping_filepath
    
    def load_image_embeddings(self, filename: str = None) -> Tuple[np.ndarray, Dict[int, str]]:
        """加载图片嵌入"""
        if filename is None:
            filename = f"image_embeddings_{self.img_emb_model}.npy"
        
        filepath = os.path.join(self.img_emb_dir, filename)
        mapping_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_{self.img_emb_model}_mapping.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image embeddings file not found: {filepath}")
        
        embeddings = np.load(filepath)
        
        # 加载映射
        with open(mapping_filepath, 'r') as f:
            id_mapping = json.load(f)
            id_mapping = {int(k): v for k, v in id_mapping.items()}
        
        return embeddings, id_mapping
    
    def apply_pca(self, embeddings: np.ndarray, n_components: int = None) -> np.ndarray:
        """对图片嵌入应用PCA降维"""
        if n_components is None:
            n_components = self.img_emb_pca
        
        if n_components <= 0 or n_components >= embeddings.shape[1]:
            return embeddings
        
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
            self.logger.info(f"PCA applied: {embeddings.shape[1]} -> {reduced_embeddings.shape[1]}")
            return reduced_embeddings
        except ImportError:
            self.logger.warning("sklearn not available, skipping PCA")
            return embeddings
    
    def run_full_pipeline(self, item_ids: List[str]) -> str:
        """运行完整的图片处理流程"""
        self.logger.info("Starting image processing pipeline...")
        
        # 生成图片嵌入
        image_embeddings = self.generate_image_embeddings(item_ids)
        
        # 应用PCA降维
        if self.img_emb_pca > 0:
            embeddings_array = np.array(list(image_embeddings.values()))
            reduced_embeddings = self.apply_pca(embeddings_array)
            
            # 更新嵌入字典
            item_id_list = list(image_embeddings.keys())
            image_embeddings = {item_id: reduced_embeddings[i] for i, item_id in enumerate(item_id_list)}
        
        # 保存嵌入
        filepath, mapping_filepath = self.save_image_embeddings(image_embeddings)
        
        self.logger.info("Image processing pipeline completed successfully!")
        return filepath
