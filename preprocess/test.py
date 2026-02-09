import json
import numpy as np

PATH = "/root/test/preprocess/emb_beauty.json"

with open(PATH, "r") as f:
    data = json.load(f)

# item 数
num_items = len(data)
print("Number of items:", num_items)

# 取第一个 item
first_item_id = next(iter(data))
embs = data[first_item_id]

print("First item id:", first_item_id)

# 每个 embedding 的 shape
for i, emb in enumerate(embs):
    arr = np.array(emb)
    print(f"emb{i+1} shape:", arr.shape)
