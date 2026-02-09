import json
import argparse
from tqdm import tqdm


def load_items(path):
    """
    item_id -> full line
    item_id -> title
    """
    item_full = {}
    item_title = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" ", 1)
            item_id = parts[0]
            rest = parts[1]

            item_full[item_id] = rest

            # extract title
            if "title:" in rest:
                title = rest.split("title:")[1].split(";")[0].strip()
            else:
                title = ""

            item_title[item_id] = title

    return item_full, item_title


def build_prompts(item_txt, sim_txt):

    item_full, item_title = load_items(item_txt)

    fused = {}

    with open(sim_txt, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header

        for line in tqdm(f):
            parts = line.strip().split()

            if len(parts) < 11:
                continue

            anchor = parts[0]
            sim_ids = parts[1:11]  # top10 only

            if anchor not in item_full:
                continue

            main_item = item_full[anchor]

            sim_titles = []
            for sid in sim_ids:
                if sid in item_title and item_title[sid]:
                    sim_titles.append(" ".join(item_title[sid].split()[:20]))

            sim_titles = ";".join(sim_titles)

            fused[anchor] = [main_item, sim_titles]

    return fused


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True,
                        help="Dataset category, e.g. Sports / Beauty")
    args = parser.parse_args()

    category = args.category

    ITEM_TXT = f"cache/AmazonReviews2014/{category}/processed/item_plain_text.txt"
    SIM_TXT = f"cache/AmazonReviews2014/{category}/processed/similar_item_sasrec.txt"
    OUT_JSON = f"cache/AmazonReviews2014/{category}/processed/fused_prompt.json"

    data = build_prompts(ITEM_TXT, SIM_TXT)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Saved to:", OUT_JSON)
    print("Total items:", len(data))
