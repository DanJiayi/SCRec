
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.5 \
    --manifold_beta=0.2 \
    --epochs=300


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.2 \
    --manifold_beta=0.2 \
    --epochs=300

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=16 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.2 \
    --manifold_beta=1 \
    --epochs=300









CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.5 \
    --manifold_beta=0 \
    --epochs=300


CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.2 \
    --manifold_beta=0 \
    --epochs=300

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Toys_and_Games \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=16 \
    --contrastive_alpha=0.5 \
    --manifold_c=0.2 \
    --manifold_beta=0 \
    --epochs=300