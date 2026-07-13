This repository contains the code for the paper "Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation". (Accepted by Recsys2026)

<p align="center">
  <img width="600" alt="image" src="[https://github.com/user-attachments/assets/e9bc5f1e-ca38-41b2-aa1f-4ef69d389aad](https://github.com/user-attachments/assets/1fd83b45-b9a2-440a-abcb-7ab09109fd33)" />
</p>


## 🚀 Quick Start

> **Note:** We provide an extracted codebook under `cache/AmazonReviews2014/Beauty/codebook`, so you can skip Step 1 and Step 2 and directly train the generative model on Beauty dataset.

#### Step 1: Extract Embeddings
Generate embeddings for target and collaborative items:

```bash
python3 -m preprocess.build_prompt --category Beauty
python3 -m preprocess.encode_items --category Beauty
```

Supported categories: `Beauty`, `Sports_and_Outdoors`, `Toys_and_Games`. Replace `Beauty` with your desired dataset.

---

#### Step 2: Train RQ-VAE and Build Code Sequences

Train RQ-VAE and construct discrete code sequences:

```bash
python3 train_rqvae_from_emb.py
```

By default, this uses the Beauty dataset. To switch datasets or modify configurations, edit `quantization/rqvae_config.yaml`
and follow the instructions in the inline comments.

`other_CT_methods` contains the scripts of alternative tokenization methods that incorporate collaborative signals, which are compared with our method in the paper.

---

#### Step 3: Train Generative Model and Evaluate

Run training and evaluation (on three datasets):

```bash
bash run.sh
```


---
### Generalizability

We also implement our proposed method as a reusable module and evaluate its performance on both TIGER[1] and LIGER[2]. Please refer to the  instructions in:

```
csa-plug-and-play/readme.md
```
---
### Acknowledgement

For the generative stage in the main experiments, we use the environment, configurations and base code from [RPG](https://github.com/facebookresearch/RPG_KDD2025) [3](e.g., the dataloader, evaluation, basic pipeline and trainer).

For extracting basic item textual information (title, brand, description, etc.) and similar items, we use the processed data from [GRAM](https://github.com/skleee/GRAM) [4].

For generalizability experients in Section 4.5, we use the official open-source codebase of [LIGER](https://github.com/facebookresearch/liger) [2]

We sincerely thank the authors of the above works for their valuable contributions.


---

### References
[1] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[2] Guanyu Lin, Zhigang Hua, Tao Feng, Shuang Yang, Bo Long, and Jiaxuan You. 2025. Unified semantic and ID representation learning for deep recommenders. arXiv preprint arXiv:2502.16474 (2025).

[3] Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, and Julian McAuley. 2025. Generating long semantic ids in parallel for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 956–966.

[4] Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, and Jongwuk Lee. 2025. Gram: Generative recommendation via semantic-aware multi-granular late fusion. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 33294–33312.
