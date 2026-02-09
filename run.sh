# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --category=Beauty \
#     --lr=0.01 \
#     --temperature=0.03 \
#     --n_codebook=32 \
#     --num_beams=20 \
#     --n_edges=200 \
#     --propagation_steps=3

python3 train_rqvae.py

python3 -m dataloader.amazon_data_processor --category CDs_and_Vinyl

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=RPG \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3 \
    --manifold_c=0.5 \
    --manifold_beta=0
    
    
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=RPG \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=100 \
    --n_edges=30 \
    --propagation_steps=5 \
    --manifold_c=0.5 \
    --manifold_beta=0

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=RPG \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --n_edges=20 \
    --propagation_steps=3 \
    --manifold_c=2.0 \
    --manifold_beta=0

CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --lr=0.001 \
    --temperature=0.03 \
    --n_codebook=64 \
    --num_beams=20 \
    --n_edges=500 \
    --propagation_steps=5