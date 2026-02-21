# python scripts/convert_embeddings_to_liger.py \
#   --embeddings /root/test/gr/cache/AmazonReviews2014/Beauty/processed/embeddings.json \
#   --output ./ID_generation/preprocessing/processed/Beauty_sentence-t5-base_embeddings_new.pt

# python scripts/convert_embeddings_to_liger.py \
# —dataset 
#   --embeddings /root/test/gr/cache/AmazonReviews2014/Toys_and_Games/processed/embeddings.json \
#   --output ./ID_generation/preprocessing/processed/Toys_and_Games_sentence-t5-base_embeddings_new.pt

# python scripts/convert_embeddings_to_liger.py \
#   --embeddings /root/test/gr/cache/AmazonReviews2014/Sports_and_Outdoors/processed/embeddings.json \
#   --output ./ID_generation/preprocessing/processed/Sports_and_Outdoors_sentence-t5-base_embeddings_new.pt


#--------------------------------liger+csa--------------------------------
python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.5 \
    experiment_id="liger_Beauty_csa" \
    +use_csa=True

python run.py \
    dataset=amazon \
    dataset.name=Sports_and_Outdoors \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.csa_contrastive_alpha=0.5 \
    method.csa_manifold_beta=0.5 \
    method.csa_manifold_c=0.5 \
    experiment_id="liger_Sports_and_Outdoors_csa" \
    +use_csa=True

python run.py \
    dataset=amazon \
    dataset.name=Toys_and_Games \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    experiment_id="liger_Toys_and_Games_csa" \
    +use_csa=True

#--------------------------------tiger+csa--------------------------------

python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.5 \
    experiment_id="tiger_Beauty_csa" \
    +use_csa=True

python run.py \
    dataset=amazon \
    dataset.name=Sports_and_Outdoors \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.csa_contrastive_alpha=0.5 \
    method.csa_manifold_beta=0.5 \
    method.csa_manifold_c=0.5 \
    experiment_id="tiger_Sports_and_Outdoors_csa" \
    +use_csa=True

python run.py \
    dataset=amazon \
    dataset.name=Toys_and_Games \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    method.csa_contrastive_alpha=0.5 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.5 \
    experiment_id="tiger_Toys_and_Games_csa" \
    +use_csa=True