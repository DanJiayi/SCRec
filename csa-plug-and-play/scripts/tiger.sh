
# Copyright (c) Meta Platforms, Inc. and affiliates.

# tiger
for dataset_name in Beauty Toys_and_Games Sports_and_Outdoors
do   
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        experiment_id="tiger_$dataset_name"
done


for dataset_name in steam
do
    python run.py \
        dataset=steam \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=0 \
        method=base \
        test_method=tiger \
        experiment_id="tiger_$dataset_name"
done