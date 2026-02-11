
# python3 -m preprocess.build_prompt --category Beauty
# python3 -m preprocess.encode_items --category Beauty
# python3 train_rqvae_from_emb.py

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.5 \
    --manifold_beta=0.2
    


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=100 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.2 \
    --manifold_beta=0.2


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=0.5 \
    --manifold_beta=0.2 \
    --manifold_c=0.2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=0.5 \
    --manifold_beta=0

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=0 \
    --manifold_beta=0.2 \
    --manifold_c=0.2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=0 \
    --manifold_beta=0


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=0.5 \
    --manifold_beta=0.2 \
    --manifold_c=0.1

