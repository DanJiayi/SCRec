## 🚀 Quick Start



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


---
## 🔧 Plug-and-play Module

We also implement CSA as a light-weight plug-and-play module and evaluate its performance on both TIGER[1] and LIGER[2]. You can reproduce the plug-and-play experiments in the `csa-plug-and-play` directory by following the instructions in:

```
csa-plug-and-play/readme.md
```

This implementation is built upon the official open-source codebase of LIGER:

https://github.com/facebookresearch/liger

Our wrapped CSA module is located at:

```
csa-plug-and-play/src/csa_module
```

---
### References
[1] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[2] Guanyu Lin, Zhigang Hua, Tao Feng, Shuang Yang, Bo Long, and Jiaxuan You. 2025. Unified semantic and ID representation learning for deep recommenders. arXiv preprint arXiv:2502.16474 (2025).
