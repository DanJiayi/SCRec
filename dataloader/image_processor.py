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
    Image processing module responsible for generating image embeddings.
    When images are missing, random vectors are generated as placeholders.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.category = config.get('category', 'Beauty')
        self.cache_dir = config.get('cache_dir', 'cache')
        self.images_dir = os.path.join(self.cache_dir, 'AmazonReviews2014', self.category, 'images')
        self.img_emb_dir = os.path.join(self.cache_dir, 'AmazonReviews2014', self.category, 'processed', 'image_embeddings')
        
        # Image processing configuration
        self.img_emb_model = config.get('img_emb_model', 'clip-vit-base-patch32')
        self.img_emb_dim = config.get('img_emb_dim', 512)
        self.img_emb_batch_size = config.get('img_emb_batch_size', 32)
        self.img_emb_pca = config.get('img_emb_pca', 256)
        self.image_size = config.get('image_size', 224)
        self.max_images_per_item = config.get('max_images_per_item', 5)
        
        # Create directories
        os.makedirs(self.img_emb_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize CLIP model (if available)
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE and self.img_emb_model.startswith('clip'):
            self._init_clip_model()
    
    def _init_clip_model(self):
        """Initialize CLIP model"""
        try:
            model_name = f"openai/{self.img_emb_model}"
            # First try loading from local cache
            local_cache_path = f"model_cache/{model_name}"
            if os.path.exists(local_cache_path):
                self.clip_model = CLIPModel.from_pretrained(local_cache_path)
                self.clip_processor = CLIPProcessor.from_pretrained(local_cache_path)
                self.logger.info(f"CLIP model {model_name} loaded successfully from local cache")
            else:
                # If not available locally, try loading from Hugging Face
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                self.logger.info(f"CLIP model {model_name} loaded successfully from Hugging Face")
        except Exception as e:
            self.logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def get_image_paths(self, item_id: str) -> List[str]:
        """Get all image paths for an item"""
        item_dir = os.path.join(self.images_dir, item_id)
        if not os.path.exists(item_dir):
            return []
        
        image_paths = []
        for filename in os.listdir(item_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(item_dir, filename))
        
        # Limit number of images per item
        return image_paths[:self.max_images_per_item]
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """Process a single image and return its embedding vector"""
        try:
            if self.clip_model is not None and self.clip_processor is not None:
                # Process image with CLIP
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.image_size, self.image_size))
                
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                
                return image_features.cpu().numpy().flatten()
            else:
                # Generate random vector as placeholder
                return np.random.normal(0, 1, self.img_emb_dim)
                
        except Exception as e:
            self.logger.warning(f"Failed to process image {image_path}: {e}")
            # Return random vector as placeholder
            return np.random.normal(0, 1, self.img_emb_dim)
    
    def process_item_images(self, item_id: str) -> np.ndarray:
        """Process all images of an item and return the average embedding vector"""
        image_paths = self.get_image_paths(item_id)
        
        if not image_paths:
            # No images available, return random vector
            return np.random.normal(0, 1, self.img_emb_dim)
        
        embeddings = []
        for image_path in image_paths:
            emb = self.process_single_image(image_path)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            # All image processing failed, return random vector
            return np.random.normal(0, 1, self.img_emb_dim)
        
        # Return average embedding vector
        return np.mean(embeddings, axis=0)
    
    def generate_image_embeddings(self, item_ids: List[str]) -> Dict[str, np.ndarray]:
        """Generate image embeddings for all items"""
        self.logger.info(f"Generating image embeddings for {len(item_ids)} items...")
        
        image_embeddings = {}
        
        for item_id in tqdm(item_ids, desc="Processing images"):
            embedding = self.process_item_images(item_id)
            image_embeddings[item_id] = embedding
        
        return image_embeddings
    
    def save_image_embeddings(self, image_embeddings: Dict[str, np.ndarray], filename: str = None):
        """Save image embeddings to file"""
        if filename is None:
            filename = f"image_embeddings_{self.img_emb_model}.npy"
        
        filepath = os.path.join(self.img_emb_dir, filename)
        
        # Convert to numpy array format
        item_ids = list(image_embeddings.keys())
        embeddings = np.array([image_embeddings[item_id] for item_id in item_ids])
        
        # Save embedding vectors
        np.save(filepath, embeddings)
        
        # Save item_id mapping
        mapping_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_{self.img_emb_model}_mapping.json")
        with open(mapping_filepath, 'w') as f:
            json.dump({str(i): item_id for i, item_id in enumerate(item_ids)}, f)
        
        self.logger.info(f"Image embeddings saved to {filepath}")
        self.logger.info(f"Embedding shape: {embeddings.shape}")
        
        return filepath, mapping_filepath
    
    def load_image_embeddings(self, filename: str = None) -> Tuple[np.ndarray, Dict[int, str]]:
        """Load image embeddings"""
        if filename is None:
            filename = f"image_embeddings_{self.img_emb_model}.npy"
        
        filepath = os.path.join(self.img_emb_dir, filename)
        mapping_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_{self.img_emb_model}_mapping.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image embeddings file not found: {filepath}")
        
        embeddings = np.load(filepath)
        
        # Load mapping
        with open(mapping_filepath, 'r') as f:
            id_mapping = json.load(f)
            id_mapping = {int(k): v for k, v in id_mapping.items()}
        
        return embeddings, id_mapping
    
    def apply_pca(self, embeddings: np.ndarray, n_components: int = None) -> np.ndarray:
        """Apply PCA dimensionality reduction to image embeddings"""
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
        """Run the full image processing pipeline"""
        self.logger.info("Starting image processing pipeline...")
        
        # Generate image embeddings
        image_embeddings = self.generate_image_embeddings(item_ids)
        
        # Apply PCA dimensionality reduction
        if self.img_emb_pca > 0:
            embeddings_array = np.array(list(image_embeddings.values()))
            reduced_embeddings = self.apply_pca(embeddings_array)
            
            # Update embedding dictionary
            item_id_list = list(image_embeddings.keys())
            image_embeddings = {item_id: reduced_embeddings[i] for i, item_id in enumerate(item_id_list)}
        
        # Save embeddings
        filepath, mapping_filepath = self.save_image_embeddings(image_embeddings)
        
        self.logger.info("Image processing pipeline completed successfully!")
        return filepath
