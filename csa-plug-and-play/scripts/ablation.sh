
# Copyright (c) Meta Platforms, Inc. and affiliates.

dataset_name=Beauty

# # Dense (ID)
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.sid_loss_weight=0 \
    method.use_id="item_id" \
    method.embedding_head_dict.embed_target="ground_truth+item_id" \
    method.embedding_head_dict.embed_proj_type="linear" \
    method.embedding_head_dict.use_new_init=True \
    experiment_id="ablation_dense_id"



# # Dense (SID)
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.sid_loss_weight=0 \
    experiment_id="ablation_dense_sid"


# liger
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    experiment_id="ablation_liger" 


# tiger(T)
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.flag_use_output_embedding=False \
    method.embedding_loss_weight=0 \
    experiment_id="ablation_tiger_text"


# tiger
python run.py \
    dataset=amazon \
    dataset.name=$dataset_name \
    seed=42 \
    device_id=0 \
    method=base \
    test_method=tiger \
    experiment_id="ablation_tiger"
