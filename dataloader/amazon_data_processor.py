#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import requests
from urllib.parse import urlparse
import yaml
import numpy as np

# Import image processing module
from .image_processor import ImageProcessor


class AmazonDataProcessor:
    def __init__(self, category: str, cache_dir: str = "cache", config_path: str = None, config: dict = None):
        self.category = category
        self.cache_dir = os.path.join(cache_dir, 'AmazonReviews2014', category)
        self.raw_dir = os.path.join(self.cache_dir, 'raw')
        self.processed_dir = os.path.join(self.cache_dir, 'processed')
        # Added: image-related directories
        self.images_dir = os.path.join(self.cache_dir, 'images')
        self.img_emb_dir = os.path.join(self.processed_dir, 'image_embeddings')
        
        self.default_config = {
            'metadata': 'sentence',
            'sent_emb_model': 'text-embedding-3-large',
            'sent_emb_dim': 3072,
            'sent_emb_batch_size': 100,
            'sent_emb_pca': 1280,  
            'n_codebook': 32,
            'codebook_size': 256,
            'faiss_omp_num_threads': 16,
            'opq_use_gpu': False,
            'opq_gpu_id': 0,
            'openai_api_key': None,
            # Added: image processing configuration
            'img_emb_model': 'clip-vit-base-patch32',  # Image embedding model
            'img_emb_dim': 1280,  # Image embedding dimension
            'img_emb_batch_size': 32,  # Image batch size
            'img_emb_pca': 256,  # Image PCA dimension
            'download_images': True,  # Whether to download images
            'image_size': 224,  # Image size
            'max_images_per_item': 5,  # Maximum number of images per item
        }
        
        self.config = self._load_config(config_path, config)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        # Added: create image-related directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.img_emb_dir, exist_ok=True)
        
        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {},
            'item2id': {},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = {}
        
    def _check_available_category(self):
        available_categories = [
            'Books', 'Electronics', 'Movies_and_TV', 'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Kindle_Store',
            'Sports_and_Outdoors', 'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care', 'Toys_and_Games', 'Video_Games',
            'Tools_and_Home_Improvement', 'Beauty', 'Apps_for_Android',
            'Office_Products', 'Pet_Supplies', 'Automotive',
            'Grocery_and_Gourmet_Food', 'Patio_Lawn_and_Garden', 'Baby',
            'Digital_Music', 'Musical_Instruments', 'Amazon_Instant_Video'
        ]
        assert self.category in available_categories, f'Category "{self.category}" not available. Available categories: {available_categories}'
    
    def download_file(self, url: str, local_path: str):
        if os.path.exists(local_path):
            print(f"File already exists: {local_path}")
            return
            
        print(f"Downloading: {url}")
        
        # Method 1: increase timeout and retry mechanism with larger chunk size
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Try multiple download methods
        download_success = False
        
        # Method 1: use requests with larger chunk size
        try:
            print("Trying method 1: requests with larger chunk size...")
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(local_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):  # Increase chunk size to 64KB
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            download_success = True
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: use wget (if available)
        if not download_success:
            try:
                print("Trying method 2: wget...")
                import subprocess
                result = subprocess.run(['wget', '-O', local_path, url], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("Download completed with wget")
                    download_success = True
                else:
                    print(f"wget failed: {result.stderr}")
            except Exception as e:
                print(f"Method 2 failed: {e}")
        
        # Method 3: use curl (if available)
        if not download_success:
            try:
                print("Trying method 3: curl...")
                import subprocess
                result = subprocess.run(['curl', '-L', '-o', local_path, url], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("Download completed with curl")
                    download_success = True
                else:
                    print(f"curl failed: {result.stderr}")
            except Exception as e:
                print(f"Method 3 failed: {e}")
        
        if not download_success:
            raise Exception("All download methods failed")
        
        # Verify integrity of downloaded file
        try:
            if local_path.endswith('.gz'):
                import gzip
                with gzip.open(local_path, 'r') as f:
                    f.read(1024)  # Try reading a small chunk to verify file integrity
                print("File integrity verified")
        except Exception as e:
            print(f"File integrity check failed: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise Exception("Downloaded file is corrupted")
    
    def _download_raw(self, data_type: str = 'reviews') -> str:
        # Try multiple mirror sources
        mirrors = [
            f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{data_type}_{self.category}{"_5" if data_type == "reviews" else ""}.json.gz',
            # Add more mirror sources here if available
        ]
        
        base_name = os.path.basename(mirrors[0])
        local_filepath = os.path.join(self.raw_dir, base_name)
        
        if not os.path.exists(local_filepath):
            # Try downloading from multiple mirrors
            for i, url in enumerate(mirrors):
                try:
                    print(f"Trying mirror {i+1}/{len(mirrors)}: {url}")
                    self.download_file(url, local_filepath)
                    break  # Exit loop if download succeeds
                except Exception as e:
                    print(f"Mirror {i+1} failed: {e}")
                    if i == len(mirrors) - 1:  # If this is the last mirror
                        raise Exception(f"All mirrors failed for {base_name}")
                    continue
        
        return local_filepath
    
    def _parse_gz(self, path: str):
        with gzip.open(path, 'r') as g:
            for line in g:
                line = line.replace(b'true', b'True').replace(b'false', b'False')
                yield eval(line)
    
    def _load_reviews(self, path: str) -> List[Tuple]:
        print('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews
    
    def _get_item_seqs(self, reviews: List[Tuple]) -> Dict:
        item_seqs = defaultdict(list)
        for user, item, time in reviews:
            item_seqs[user].append((item, time))
        
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [item for item, _ in item_time]
        return item_seqs
    
    def _remap_ids(self, item_seqs: Dict) -> Tuple[Dict, Dict]:
        print('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            
            iids = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        
        return self.all_item_seqs, self.id_mapping
    
    def _process_reviews(self, input_path: str) -> Tuple[Dict, Dict]:
        seq_file = os.path.join(self.processed_dir, 'all_item_seqs.json')
        id_mapping_file = os.path.join(self.processed_dir, 'id_mapping.json')
        
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            print('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping
        
        print('[DATASET] Processing reviews...')
        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)
        
        print('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        
        return all_item_seqs, id_mapping
    
    def _load_metadata(self, path: str, item2id: Dict) -> Dict:
        print('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data
    
    def clean_text(self, raw_text: str) -> str:
        import re
        import html
        
        if isinstance(raw_text, list):
            raw_text = ' '.join(str(item) for item in raw_text)
        
        text = str(raw_text)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _sent_process(self, raw) -> str:
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw) + '.'
        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += self.clean_text(str(v))[:-1] + ', '
            sentence = sentence[:-2] + '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += self.clean_text(str(v1))
        else:
            sentence = self.clean_text(str(raw))
        return sentence + ' '
    
    def _extract_meta_sentences(self, metadata: Dict) -> Dict:
        print('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ''
            keys = set(meta.keys())
            features_needed = ['title', 'price', 'brand', 'feature', 'categories', 'description']
            for feature in features_needed:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        return item2meta
    
    def _process_meta(self, input_path: str) -> Optional[Dict]:
        process_mode = self.config['metadata']
        meta_file = os.path.join(self.processed_dir, f'metadata.{process_mode}.json')
        
        if os.path.exists(meta_file):
            print('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)
        
        print(f'[DATASET] Processing metadata, mode: {process_mode}')
        
        if process_mode == 'none':
            return None
        
        item2meta = self._load_metadata(path=input_path, item2id=self.id_mapping['item2id'])
        
        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        elif process_mode == 'multimodal':
            # Multimodal mode: keep raw metadata, but extract text for text embeddings
            print('[MULTIMODAL] Processing metadata for multimodal (text + images)')
            # Generate text descriptions for text embeddings
            item2meta_text = self._extract_meta_sentences(metadata=item2meta)
            # Save text version for embedding generation
            text_meta_file = os.path.join(self.processed_dir, 'metadata.sentence.json')
            with open(text_meta_file, 'w') as f:
                json.dump(item2meta_text, f)
            print('[MULTIMODAL] Text metadata saved for embeddings')
            # Keep raw metadata for multimodal processing
            pass
        
        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        
        return item2meta
    
    def _encode_sent_emb(self, output_path: str) -> np.ndarray:
        print('[TOKENIZER] Encoding sentence embeddings...')
        
        # For multimodal mode, use the text version of metadata
        if self.config['metadata'] == 'multimodal':
            text_meta_file = os.path.join(self.processed_dir, 'metadata.sentence.json')
            if os.path.exists(text_meta_file):
                with open(text_meta_file, 'r') as f:
                    text_metadata = json.load(f)
                print('[TOKENIZER] Using text metadata for embeddings')
            else:
                print('[TOKENIZER] Text metadata not found, falling back to raw metadata')
                text_metadata = self.item2meta
        else:
            text_metadata = self.item2meta
        
        meta_sentences = []
        # Create a zero-vector placeholder for PAD token
        pad_embedding = np.zeros(self.config['sent_emb_dim'], dtype=np.float32)
        
        for i in range(1, len(self.id_mapping['id2item'])):
            item = self.id_mapping['id2item'][i]
            meta_sentences.append(text_metadata[item])
        
        if 'sentence-transformers' in self.config['sent_emb_model']:
            try:
                from sentence_transformers import SentenceTransformer
                device = self.config.get('device', 'cpu')
                sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(device)
                
                sent_embs = sent_emb_model.encode(
                    meta_sentences,
                    convert_to_numpy=True,
                    batch_size=self.config['sent_emb_batch_size'],
                    show_progress_bar=True,
                    device=device
                )
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            if not self.config['openai_api_key']:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            
            try:
                from openai import OpenAI
                
                client_kwargs = {'api_key': self.config['openai_api_key']}
                if 'openai_base_url' in self.config and self.config['openai_base_url']:
                    client_kwargs['base_url'] = self.config['openai_base_url']
                
                client = OpenAI(**client_kwargs)
                
                sent_embs = []
                max_retries = 3  # Increase retry count
                for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                    batch = meta_sentences[i:i + self.config['sent_emb_batch_size']]
                    retry_count = 0
                    success = False
                    
                    while retry_count < max_retries and not success:
                        try:
                            responses = client.embeddings.create(
                                input=batch,
                                model=self.config['sent_emb_model']
                            )
                            
                            for response in responses.data:
                                sent_embs.append(response.embedding)
                            success = True
                        except Exception as e:
                            retry_count += 1
                            print(f'Encoding failed {i} - {i + self.config["sent_emb_batch_size"]} (attempt {retry_count}/{max_retries}): {e}')
                            
                            if retry_count < max_retries:
                                # Handle text-length issues
                                new_batch = []
                                for sent in batch:
                                    if len(sent) > 8000:
                                        new_batch.append(sent[:8000])
                                    else:
                                        new_batch.append(sent)
                                
                                print(f'[TOKENIZER] Retrying batch {i} - {i + self.config["sent_emb_batch_size"]} with shorter texts...')
                                import time
                                time.sleep(5 * retry_count)  # Incremental waiting time
                                
                                batch = new_batch  # Use processed batch
                            else:
                                print(f'All retries failed for batch {i} - {i + self.config["sent_emb_batch_size"]}')
                                raise e
                    
                sent_embs = np.array(sent_embs, dtype=np.float32)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError(f"Unsupported embedding model: {self.config['sent_emb_model']}")
        
        sent_embs.tofile(output_path)
        print(f'[TOKENIZER] Sentence embeddings saved to: {output_path}')
        return sent_embs
    
    def _get_items_for_training(self) -> np.ndarray:
        mask = np.ones(len(self.id_mapping['id2item']) - 1, dtype=bool)
        print(f'[TOKENIZER] Training items count: {mask.sum()} / {len(self.id_mapping["id2item"]) - 1}')
        return mask
    
    def _get_codebook_bits(self, n_codebook: int) -> int:
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)
    

    def generate_embeddings(self):
        if self.config['metadata'] != 'sentence' and self.config['metadata'] != 'multimodal':
            print('[TOKENIZER] Skipping embedding generation, metadata is not in sentence or multimodal mode')
            return
        
        # Check whether text embedding file already exists
        sent_emb_path = os.path.join(
            self.processed_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        
        if os.path.exists(sent_emb_path):
            print(f'[TOKENIZER] Text embeddings already exist: {sent_emb_path}')
            
            # Check whether PCA embeddings need to be regenerated
            pca_emb_path = os.path.join(self.processed_dir, f'final_pca_embeddings_{self.config["sent_emb_pca"]}d.npy')
            default_pca_path = os.path.join(self.processed_dir, 'final_pca_embeddings.npy')
            need_regenerate = False
            
            if self.config['sent_emb_pca'] > 0:
                # Prefer checking dimension-specific filename first
                if os.path.exists(pca_emb_path):
                    try:
                        # import numpy as np
                        existing_pca = np.load(pca_emb_path)
                        if existing_pca.shape[1] == self.config['sent_emb_pca']:
                            print(f'[TOKENIZER] PCA embeddings already exist with correct dimension: {existing_pca.shape}')
                            print('[TOKENIZER] Skipping embedding generation')
                            return
                        else:
                            print(f'[TOKENIZER] Existing PCA embeddings dimension mismatch: {existing_pca.shape[1]} vs {self.config["sent_emb_pca"]}')
                            need_regenerate = True
                    except Exception as e:
                        print(f'[TOKENIZER] Error checking existing PCA embeddings: {e}')
                        need_regenerate = True
                # If dimension-specific file does not exist, check default file
                elif os.path.exists(default_pca_path):
                    try:
                        # import numpy as np
                        existing_pca = np.load(default_pca_path)
                        if existing_pca.shape[1] == self.config['sent_emb_pca']:
                            print(f'[TOKENIZER] Default PCA embeddings have correct dimension: {existing_pca.shape}')
                            print('[TOKENIZER] But dimension-specific file is missing, will generate it')
                            need_regenerate = True
                        else:
                            print(f'[TOKENIZER] Default PCA embeddings dimension mismatch: {existing_pca.shape[1]} vs {self.config["sent_emb_pca"]}')
                            need_regenerate = True
                    except Exception as e:
                        print(f'[TOKENIZER] Error checking default PCA embeddings: {e}')
                        need_regenerate = True
                else:
                    print('[TOKENIZER] PCA embeddings not found, will generate new ones')
                    need_regenerate = True
            else:
                print('[TOKENIZER] Skipping text embedding generation')
                return
            
            if not need_regenerate:
                return
        
        sem_ids_path = os.path.join(
            self.processed_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}_OPQ{self.config["n_codebook"]},IVF1,PQ{self.config["n_codebook"]}x{self._get_codebook_bits(self.config["codebook_size"])}.sem_ids'
        )
        
        if os.path.exists(sem_ids_path):
            print(f'[TOKENIZER] Semantic IDs already exist: {sem_ids_path}')
            return
        

        # If text embeddings already exist and only PCA regeneration is needed, load directly
        if os.path.exists(sent_emb_path) and need_regenerate:
            print('[TOKENIZER] Loading existing text embeddings for PCA regeneration...')
            # import numpy as np
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32)
            sent_embs = sent_embs.reshape(-1, self.config['sent_emb_dim'])
            print(f'[TOKENIZER] Loaded text embeddings shape: {sent_embs.shape}')
        else:
            print('[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(sent_emb_path)
        
        if self.config['sent_emb_pca'] > 0:
            print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
                
                # Add zero-vector embedding for PAD token
                pad_embedding_pca = np.zeros(self.config['sent_emb_pca'], dtype=np.float32)
                sent_embs = np.vstack([pad_embedding_pca, sent_embs])
                
                # Generate dimension-specific filename
                pca_emb_path = os.path.join(self.processed_dir, f'final_pca_embeddings_{self.config["sent_emb_pca"]}d.npy')
                np.save(pca_emb_path, sent_embs)
                print(f'[TOKENIZER] PCA embeddings saved to: {pca_emb_path}')
                
                # Also save a default filename for compatibility
                default_pca_path = os.path.join(self.processed_dir, 'final_pca_embeddings.npy')
                np.save(default_pca_path, sent_embs)
                print(f'[TOKENIZER] Default PCA embeddings also saved to: {default_pca_path}')
            except ImportError:
                raise ImportError("Please install scikit-learn: pip install scikit-learn")
        
        print(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
        print(f'[TOKENIZER] Total items (including PAD): {sent_embs.shape[0]}')
        print(f'[TOKENIZER] Actual items (excluding PAD): {sent_embs.shape[0] - 1}')
    
    def _extract_image_urls(self, metadata: Dict) -> Dict:
        """从元数据中提取图片URL"""
        print('[MULTIMODAL] Extracting image URLs from metadata...')
        item2images = {}
        
        for item_id, meta in tqdm(metadata.items(), desc='Extracting image URLs'):
            image_urls = []
            
            # Check different image fields
            if 'image' in meta:
                if isinstance(meta['image'], list):
                    image_urls.extend(meta['image'])
                elif isinstance(meta['image'], str):
                    image_urls.append(meta['image'])
            
            if 'images' in meta:
                if isinstance(meta['images'], list):
                    image_urls.extend(meta['images'])
                elif isinstance(meta['images'], str):
                    image_urls.append(meta['images'])
            
            # Limit number of images per item
            max_images = self.config.get('max_images_per_item', 5)
            image_urls = image_urls[:max_images]
            
            if image_urls:
                item2images[item_id] = image_urls
        
        print(f'[MULTIMODAL] Found {len(item2images)} items with images')
        return item2images
    
    def _download_images(self, item2images: Dict) -> Dict:
        """下载图片"""
        if not self.config.get('download_images', True):
            print('[MULTIMODAL] Image downloading disabled')
            return {}
        
        print('[MULTIMODAL] Downloading images...')
        item2image_paths = {}
        
        for item_id, image_urls in tqdm(item2images.items(), desc='Downloading images'):
            item_image_dir = os.path.join(self.images_dir, item_id)
            os.makedirs(item_image_dir, exist_ok=True)
            
            item_paths = []
            for i, url in enumerate(image_urls):
                try:
                    # Generate image filename
                    file_ext = '.jpg'  # Default extension
                    if '.' in url.split('/')[-1]:
                        file_ext = '.' + url.split('/')[-1].split('.')[-1]
                    
                    image_filename = f'image_{i}{file_ext}'
                    image_path = os.path.join(item_image_dir, image_filename)
                    
                    # If file already exists, skip download
                    if os.path.exists(image_path):
                        item_paths.append(image_path)
                        continue
                    
                    # Download image
                    response = requests.get(url, timeout=30, stream=True)
                    response.raise_for_status()
                    
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    item_paths.append(image_path)
                    
                except Exception as e:
                    print(f'[MULTIMODAL] Failed to download image {url}: {e}')
                    continue
            
            if item_paths:
                item2image_paths[item_id] = item_paths
        
        print(f'[MULTIMODAL] Downloaded images for {len(item2image_paths)} items')
        return item2image_paths
    
    def _encode_image_emb(self, item2image_paths: Dict) -> np.ndarray:
        """生成图片嵌入"""
        if not item2image_paths:
            print('[MULTIMODAL] No images to encode')
            return None
        
        print('[MULTIMODAL] Encoding image embeddings...')
        
        try:
            from PIL import Image
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            print('[MULTIMODAL] Please install required packages: pip install pillow torch transformers')
            return None
        
        # Load CLIP model
        model_name = self.config.get('img_emb_model', 'openai/clip-vit-base-patch32')
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
        except Exception as e:
            print(f'[MULTIMODAL] Failed to load CLIP model: {e}')
            return None
        
        # Prepare image path list and corresponding item IDs
        all_image_paths = []
        item_ids = []
        
        for item_id, image_paths in item2image_paths.items():
            for image_path in image_paths:
                if os.path.exists(image_path):
                    all_image_paths.append(image_path)
                    item_ids.append(item_id)
        
        if not all_image_paths:
            print('[MULTIMODAL] No valid images found')
            return None
        
        # Batch-process images
        batch_size = self.config.get('img_emb_batch_size', 32)
        image_embeddings = []
        processed_item_ids = []
        
        for i in tqdm(range(0, len(all_image_paths), batch_size), desc='Encoding images'):
            batch_paths = all_image_paths[i:i + batch_size]
            batch_item_ids = item_ids[i:i + batch_size]
            
            try:
                # Load and preprocess images
                images = []
                valid_indices = []
                
                for j, image_path in enumerate(batch_paths):
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize((224, 224))  # Resize
                        images.append(image)
                        valid_indices.append(j)
                    except Exception as e:
                        print(f'[MULTIMODAL] Failed to load image {image_path}: {e}')
                        continue
                
                if not images:
                    continue
                
                # Process images
                inputs = processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features.cpu().numpy()
                
                # Append to results
                for j, idx in enumerate(valid_indices):
                    image_embeddings.append(image_features[j])
                    processed_item_ids.append(batch_item_ids[idx])
                
            except Exception as e:
                print(f'[MULTIMODAL] Failed to process batch {i}: {e}')
                continue
        
        if not image_embeddings:
            print('[MULTIMODAL] No image embeddings generated')
            return None
        
        # Convert to numpy array
        image_embeddings = np.array(image_embeddings, dtype=np.float32)
        
        # Save raw embeddings
        img_emb_path = os.path.join(self.img_emb_dir, 'raw_image_embeddings.npy')
        np.save(img_emb_path, image_embeddings)
        
        # Save item ID mapping
        img_id_mapping_path = os.path.join(self.img_emb_dir, 'image_item_ids.json')
        with open(img_id_mapping_path, 'w') as f:
            json.dump(processed_item_ids, f)
        
        print(f'[MULTIMODAL] Image embeddings shape: {image_embeddings.shape}')
        print(f'[MULTIMODAL] Image embeddings saved to: {img_emb_path}')
        
        return image_embeddings
    
    def process_multimodal_data(self):
        """处理多模态数据（文本+图片）"""
        if self.config['metadata'] != 'multimodal':
            print('[MULTIMODAL] Skipping multimodal processing, metadata mode is not multimodal')
            return
        
        print('\n=== Step 4: Process multimodal data (text + images) ===')
        
        # Check whether CLIP embedding files already exist
        clip_emb_file = os.path.join(self.img_emb_dir, 'image_embeddings_clip-vit-base-patch32.npy')
        clip_mapping_file = os.path.join(self.img_emb_dir, 'image_embeddings_clip-vit-base-patch32_mapping.json')
        
        if os.path.exists(clip_emb_file) and os.path.exists(clip_mapping_file):
            print('[MULTIMODAL] CLIP image embeddings already exist, skipping multimodal processing')
            print(f'[MULTIMODAL] CLIP embeddings: {clip_emb_file}')
            print(f'[MULTIMODAL] CLIP mapping: {clip_mapping_file}')
            return
        
        # Check whether metadata exists
        if not self.item2meta:
            print('[MULTIMODAL] No metadata available for multimodal processing')
            return
        
        # Extract image URLs
        item2images = self._extract_image_urls(self.item2meta)
        
        # Download images
        item2image_paths = self._download_images(item2images)
        
        # Generate image embeddings
        if item2image_paths:
            image_embeddings = self._encode_image_emb(item2image_paths)
            
            # Apply PCA dimensionality reduction
            if image_embeddings is not None and self.config.get('img_emb_pca', 0) > 0:
                print(f'[MULTIMODAL] Applying PCA to image embeddings...')
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['img_emb_pca'], whiten=True)
                    image_embeddings = pca.fit_transform(image_embeddings)
                    
                    pca_img_emb_path = os.path.join(self.img_emb_dir, 'pca_image_embeddings.npy')
                    np.save(pca_img_emb_path, image_embeddings)
                    print(f'[MULTIMODAL] PCA image embeddings saved to: {pca_img_emb_path}')
                except ImportError:
                    print('[MULTIMODAL] Please install scikit-learn: pip install scikit-learn')
        
        print('[MULTIMODAL] Multimodal data processing completed')
    
    def run_full_pipeline(self):
        print(f"Starting Amazon Reviews 2014 dataset processing - Category: {self.category}")
        
        self._check_available_category()
        
        print("\n=== Step 1: Download raw data ===")
        reviews_path = self._download_raw('reviews')
        meta_path = self._download_raw('meta')
        
        print("\n=== Step 2: Process reviews ===")
        self.all_item_seqs, self.id_mapping = self._process_reviews(reviews_path)
        
        print("\n=== Step 3: Process metadata ===")
        self.item2meta = self._process_meta(meta_path)
        
        if self.item2meta:
            if self.config['metadata'] == 'multimodal':
                print("\n=== Step 4: Process multimodal data (text + images) ===")
                self.process_multimodal_data()
                print("\n=== Step 5: Generate text embeddings and semantic IDs ===")
                self.generate_embeddings()
                print("\n=== Step 6: Process image embeddings ===")
                self.process_image_embeddings()
            else:
                print("\n=== Step 4: Generate embeddings and semantic IDs ===")
                self.generate_embeddings()
        
        print(f"\n=== Processing completed ===")
        print(f"Data saved in: {self.cache_dir}")
        print(f"Raw data: {self.raw_dir}")
        print(f"Processed data: {self.processed_dir}")
        
        # Comment out file-structure output to avoid printing a large number of image filenames
        # print("\nGenerated files:")
        # for root, dirs, files in os.walk(self.cache_dir):
        #     level = root.replace(self.cache_dir, '').count(os.sep)
        #     indent = ' ' * 2 * level
        #     print(f"{indent}{os.path.basename(root)}/")
        #     subindent = ' ' * 2 * (level + 1)
        #     for file in files:
        #         print(f"{subindent}{file}")
    
    def process_image_embeddings(self):
        """处理图片嵌入"""
        print("[IMAGE] Starting image embedding processing...")
        
        # Check whether CLIP embedding files already exist
        clip_emb_file = os.path.join(self.img_emb_dir, 'image_embeddings_clip-vit-base-patch32.npy')
        clip_mapping_file = os.path.join(self.img_emb_dir, 'image_embeddings_clip-vit-base-patch32_mapping.json')
        
        if os.path.exists(clip_emb_file) and os.path.exists(clip_mapping_file):
            print("[IMAGE] CLIP image embeddings already exist, skipping image processing")
            print(f"[IMAGE] CLIP embeddings: {clip_emb_file}")
            print(f"[IMAGE] CLIP mapping: {clip_mapping_file}")
            return
        
        # Initialize image processor
        image_processor = ImageProcessor(self.config)
        
        # Get all item IDs
        item_ids = list(self.id_mapping['item2id'].keys())
        item_ids = [item_id for item_id in item_ids if item_id != '[PAD]']
        
        print(f"[IMAGE] Processing {len(item_ids)} items...")
        
        # Run image processing pipeline
        try:
            image_emb_filepath = image_processor.run_full_pipeline(item_ids)
            print(f"[IMAGE] Image embeddings saved to: {image_emb_filepath}")
        except Exception as e:
            print(f"[IMAGE] Error processing image embeddings: {e}")
            print("[IMAGE] Will continue with random vectors as placeholders")
            
            # Generate random vectors as placeholders
            random_embeddings = {}
            for item_id in item_ids:
                random_embeddings[item_id] = np.random.normal(0, 1, self.config.get('img_emb_dim', 1280))
            
            # Save random vectors
            embeddings_array = np.array([random_embeddings[item_id] for item_id in item_ids])
            random_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_random.npy")
            np.save(random_filepath, embeddings_array)
            
            # Save mapping
            mapping_filepath = os.path.join(self.img_emb_dir, f"image_embeddings_random_mapping.json")
            with open(mapping_filepath, 'w') as f:
                json.dump({str(i): item_id for i, item_id in enumerate(item_ids)}, f)
            
            print(f"[IMAGE] Random embeddings saved to: {random_filepath}")
        
        print("[IMAGE] Image embedding processing completed!")

    def _load_config(self, config_path: str = None, config_dict: dict = None) -> dict:
        final_config = self.default_config.copy()
        
        category_config_path = os.path.join(
            os.path.dirname(__file__), 
            'AmazonReviews2014', 
            'config.yaml'
        )
        if os.path.exists(category_config_path):
            try:
                with open(category_config_path, 'r', encoding='utf-8') as f:
                    category_config = yaml.safe_load(f)
                    if category_config:
                        print(f"[CONFIG] Loading from category config: {category_config_path}")
                        final_config.update(category_config)
            except Exception as e:
                print(f"[CONFIG] Warning: Cannot load category config: {e}")
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    external_config = yaml.safe_load(f)
                    if external_config:
                        print(f"[CONFIG] Loading from external config: {config_path}")
                        final_config.update(external_config)
            except Exception as e:
                print(f"[CONFIG] Warning: Cannot load external config: {e}")
        
        if config_dict:
            print("[CONFIG] Loading from provided config dict")
            final_config.update(config_dict)
        
        return final_config


def main():
    parser = argparse.ArgumentParser(description='Amazon Reviews 2014 Data Processor')
    parser.add_argument('--category', type=str, required=True, help='Amazon category to process')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--metadata', type=str, choices=['none', 'raw', 'sentence'], default='sentence', help='Metadata processing mode')
    parser.add_argument('--sent_emb_pca', type=int, help='PCA dimension for sentence embeddings (default: from config)')
    
    args = parser.parse_args()
    
    config_override = {
        'metadata': args.metadata
    }
    
    # If PCA dimension is specified, add it to config overrides
    if args.sent_emb_pca is not None:
        config_override['sent_emb_pca'] = args.sent_emb_pca
    
    processor = AmazonDataProcessor(
        category=args.category,
        cache_dir=args.cache_dir,
        config_path=args.config,
        config=config_override
    )
    
    processor.run_full_pipeline()


if __name__ == '__main__':
    main()
