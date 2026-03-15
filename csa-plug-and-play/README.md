## 💡 Plug-and-play Experiments

This directory provides experiments for evaluating our module on both LIGER and TIGER.

---

#### Step 1: Convert Semantic Embeddings to LIGER Format

Run:

```bash
python scripts/convert_embeddings_to_liger.py \
  --embeddings ../cache/AmazonReviews2014/Beauty/processed/embeddings.json \
  --output ./ID_generation/preprocessing/processed/Beauty_sentence-t5-base_embeddings_new.pt
```

The dataset `Beauty` can be replaced with `Sports_and_Outdoors` or `Toys_and_Games`.

**Note:**  The first argument (`--embeddings`) should point to the item embeddings generated in the main experiments (including textual attributes and collaborative item attributes).  
If embeddings have not been generated yet, please refer to Step 1 in the parent directory `readme.md`. This script only converts the embeddings into the same format used by the original LIGER codebase for compatibility.

---

#### Step 2: Run Plug-and-play Training and Evaluation

Execute:

```bash
bash run.sh
```

This will report the performance of LIGER + Ours and TIGER + Ours across all three datasets.

---

This implementation is built upon the official open-source codebase of LIGER:

https://github.com/facebookresearch/liger

The wrapped module is located at:

```
src/csa_module
```
