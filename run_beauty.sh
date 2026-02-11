CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --manifold_c=0.5 \
    --manifold_beta=0.5
    
    