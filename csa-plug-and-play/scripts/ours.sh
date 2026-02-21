# Copyright (c) Meta Platforms, Inc. and affiliates.

# liger
# for dataset_name in Beauty Toys_and_Games Sports_and_Outdoors
# do
#     python run.py \
#         dataset=amazon \
#         dataset.name=$dataset_name \
#         seed=42 \
#         device_id=0 \
#         method=setting \
#         test_method=liger \
#         experiment_id="liger_$dataset_name"
# done


# for dataset_name in steam
# do
#     python run.py \
#         dataset=steam \
#         dataset.name=$dataset_name \
#         seed=42 \
#         device_id=0 \
#         method=setting \
#         test_method=liger \
#         experiment_id="liger_$dataset_name"
# done

# python run.py \
#     dataset=amazon \
#     dataset.name=Beauty \
#     seed=42 \
#     device_id=0 \
#     method=setting \
#     test_method=liger \
#     experiment_id="liger_Beauty"

python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    experiment_id="liger_Beauty_new"

# python run.py \
#     dataset=amazon \
#     dataset.name=Toys_and_Games \
#     +dataset.embedding_suffix="_new" \
#     +dataset.embedding_file_model=sentence-t5-base \
#     seed=42 \
#     device_id=0 \
#     method=setting \
#     test_method=liger \
#     experiment_id="liger_Toys_and_Games_new"

# python run.py \
#     dataset=amazon \
#     dataset.name=Sports_and_Outdoors \
#     +dataset.embedding_suffix="_new" \
#     +dataset.embedding_file_model=sentence-t5-base \
#     seed=42 \
#     device_id=0 \
#     method=setting \
#     test_method=liger \
#     experiment_id="liger_Sports_and_Outdoors_new"


# python scripts/convert_embeddings_to_liger.py \
# --dataset Toys_and_Games \
# --embeddings /root/test/gr/cache/AmazonReviews2014/Toys_and_Games/processed/embeddings.json \
# --output ./ID_generation/preprocessing/processed/Toys_and_Games_sentence-t5-base_embeddings_new.pt

# python scripts/convert_embeddings_to_liger.py \
# --dataset Sports_and_Outdoors \
# --embeddings /root/test/gr/cache/AmazonReviews2014/Sports_and_Outdoors/processed/embeddings.json \
# --output ./ID_generation/preprocessing/processed/Sports_and_Outdoors_sentence-t5-base_embeddings_new.pt

python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    experiment_id="liger_Beauty_csa_10" \
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
    experiment_id="liger_Toys_and_Games_csa_10" \
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
    experiment_id="liger_Sports_and_Outdoors_csa_10" \
    +use_csa=True


python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    method=setting \
    test_method=liger \
    +use_csa=True \
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.2 \
    experiment_id="liger_Beauty_csa_222"