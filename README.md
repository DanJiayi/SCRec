### 🚀 Quick Start



#### Step 1: Extract Semantic Embeddings

Generate semantic embeddings for target and collaborative items:

```bash
python3 -m preprocess.build_prompt --category Beauty
python3 -m preprocess.encode_items --category Beauty
```

Supported categories: `Beauty`, `Sports_and_Outdoors`, `Toys_and_Games`.

Replace `Beauty` with your desired category.

---

#### Step 2: Train RQ-VAE and Build Code Sequences

Train RQ-VAE and construct discrete code sequences:

```bash
python3 train_rqvae_from_emb.py
```

By default, this uses the Beauty dataset.  
To switch datasets or modify configurations, edit:
```text
quantization/rqvae_config.yaml
```
and follow the instructions in the inline comments.

---

#### Step 3: Train Generative Model and Evaluate

Run training and evaluation (on three datasets):

```bash
bash run.sh
```
