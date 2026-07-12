
The wrapped module of our method is located at `csa_module.py`, and can be incorporated into existing generative recommendation methods.

To reproduce the generalizability experiments on LIGER and TIGER:

1. Clone the official open source code of LIGER (https://github.com/facebookresearch/liger), which contains code for both TIGER and LIGER.

2. Convert the extracted embeddings to LIGER format by running
```bash
python convert_embeddings_to_liger.py \
  --embeddings ../cache/AmazonReviews2014/Beauty/processed/embeddings.json \
  --output ./ID_generation/preprocessing/processed/Beauty_sentence-t5-base_embeddings_new.pt
```
The dataset `Beauty` can be replaced with `Sports_and_Outdoors` or `Toys_and_Games`.

This ensures that semantic and collaborative signals are fused into the code sequences. The first argument (`--embeddings`) should point to the item embeddings generated in the main experiments (including textual attributes and collaborative item attributes). If embeddings have not been generated yet, please refer to Step 1 in the parent directory `readme.md`. This script only converts the input embeddings into the same format used by the original codebase for compatibility.

3. In the training script, initialize and invoke our module as follows:
``` python
csa_module = CSAModule(
    hidden_dim=t5_config["d_model"],
    text_dim=item_embedding.shape[1],
).to(device)
fused, csa_losses = csa_module(batch, device, method_config, model, item_embedding)
```
(All input arguments are readily available in the existing LIGER training script)

4. Run
```
bash run.sh
```
This will report the performance of LIGER + Ours and TIGER + Ours across all three datasets.

