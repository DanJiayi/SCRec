python run.py \
    dataset=amazon \
    dataset.name=Sports_and_Outdoors \
    +dataset.embedding_suffix="_new" \
    +dataset.embedding_file_model=sentence-t5-base \
    seed=42 \
    device_id=0 \
    method=setting \
    test_method=liger \
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.2 \
    experiment_id="liger_Sports_and_Outdoors_csa_222" \
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
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.5 \
    experiment_id="liger_Sports_and_Outdoors_csa_225" \
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
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.5 \
    method.csa_manifold_c=0.2 \
    experiment_id="liger_Sports_and_Outdoors_csa_252" \
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
    method.csa_contrastive_alpha=0.2 \
    method.csa_manifold_beta=0.5 \
    method.csa_manifold_c=0.5 \
    experiment_id="liger_Sports_and_Outdoors_csa_255" \
    +use_csa=True



#______________________
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
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.2 \
    experiment_id="liger_Sports_and_Outdoors_csa_522" \
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
    method.csa_manifold_beta=0.2 \
    method.csa_manifold_c=0.5 \
    experiment_id="liger_Sports_and_Outdoors_csa_525" \
    +use_csa=True \

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
    method.csa_manifold_c=0.2 \
    experiment_id="liger_Sports_and_Outdoors_csa_552" \
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
    experiment_id="liger_Sports_and_Outdoors_csa_555" \
    +use_csa=True