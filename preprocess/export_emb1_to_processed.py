#!/usr/bin/env python3
"""
把 emb_Beauty.json 中每个 item 的 emb1 导出为与 final_pca_embeddings 格式和顺序一致的 .npy 文件，
保存到 gr/cache/AmazonReviews2014/Beauty/processed/，便于在 model_add_text.py 中仅改读取路径即可切换 text emb。

- 输入: /root/test/preprocess/emb_Beauty.json，格式 item_id -> [emb1, emb2, emb3]，item_id 为原始 ID（如 "B00DJQQEGQ"）
- 输出: processed 目录下的 .npy，形状 (n_items, emb_dim)，dtype float32
  - 第 0 行: PAD（全零）
  - 第 i 行: id2item[i] 对应的 emb1；若 json 中无该 item 则该行保持为零并打日志
- 不做 PCA，仅按 id_mapping 的 id2item 顺序排列。
"""

import json
import os
import numpy as np

# 路径
PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_JSON_PATH = os.path.join(PREPROCESS_DIR, "emb_Beauty.json")
PROCESSED_DIR = os.path.join(
    os.path.dirname(PREPROCESS_DIR),
    "gr", "cache", "AmazonReviews2014", "Beauty", "processed"
)
ID_MAPPING_PATH = os.path.join(PROCESSED_DIR, "id_mapping.json")
OUTPUT_NPY_PATH = os.path.join(PROCESSED_DIR, "emb1_text_embeddings.npy")


def main():
    # 加载 id_mapping（与 final_pca_embeddings 一致的 item 顺序）
    with open(ID_MAPPING_PATH, "r", encoding="utf-8") as f:
        id_mapping = json.load(f)
    id2item = id_mapping["id2item"]  # list: [PAD], item1, item2, ...
    n_items = len(id2item)

    # 加载每个 item 的 embeddings（原始 item id -> [emb1, emb2, emb3]）
    with open(EMB_JSON_PATH, "r", encoding="utf-8") as f:
        emb_data = json.load(f)

    # 从任意一条取 emb1 的维度
    first_key = next(iter(emb_data))
    emb1 = emb_data[first_key][0]
    emb_dim = len(emb1)

    # 构建 (n_items, emb_dim)，与 final_pca_embeddings 格式一致
    arr = np.zeros((n_items, emb_dim), dtype=np.float32)
    missing = []
    for i in range(1, n_items):
        orig_id = id2item[i]
        if orig_id in emb_data:
            arr[i] = np.array(emb_data[orig_id][0], dtype=np.float32)
        else:
            missing.append(orig_id)

    if missing:
        print(f"[WARN] {len(missing)} items not in emb json (rows left zero): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(OUTPUT_NPY_PATH, arr)
    print(f"[OK] Saved shape {arr.shape} to {OUTPUT_NPY_PATH}")
    print(f"     Row 0 = PAD (zeros), row i = id2item[i]'s emb")


if __name__ == "__main__":
    main()

