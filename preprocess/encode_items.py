import json
import argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


MODEL_PATH = "change this to your sentence embedding model, e.g. sentence-t5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True,
                        help="Dataset category, e.g. Sports / Beauty")
    args = parser.parse_args()

    category = args.category
    INPUT_JSON = f"cache/AmazonReviews2014/{category}/processed/fused_prompt.json"
    OUTPUT_JSON = f"cache/AmazonReviews2014/{category}/processed/embeddings.json"

    print(f"[INFO] Input: {INPUT_JSON}")
    print(f"[INFO] Output: {OUTPUT_JSON}")

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    sent_emb_model = SentenceTransformer(MODEL_PATH).to(DEVICE)

    item_ids = []
    texts = []

    for item_id, pair in data.items():
        item_ids.append(item_id)
        texts.append(pair[0])
        texts.append(pair[1])

    sent_embs = sent_emb_model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE
    )

    result = {}
    idx = 0

    for item_id in tqdm(item_ids):
        emb1 = sent_embs[idx]
        emb2 = sent_embs[idx + 1]

        emb3 = 0.8 * emb1 + 0.2 * emb2
        # last_shape = sent_embs.shape
        result[item_id] = [
            emb1.tolist(),
            emb2.tolist(),
            emb3.tolist()
        ]

        idx += 2

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f)

    print(f"Saved to {OUTPUT_JSON}")
    # print("Embedding batch shape:", last_shape)


if __name__ == "__main__":
    main()
