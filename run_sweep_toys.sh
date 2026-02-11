#!/usr/bin/env bash
set -euo pipefail

category="Toys_and_Games"
values=(0.2 0.5 1 2 5)

fixed_contrastive_alpha=0.5
fixed_manifold_beta=0.2
fixed_manifold_c=0.5

log_dir="logs_sweep"
result_file="sweep_test_results_${category}.txt"

mkdir -p "$log_dir"
if [[ ! -f "$result_file" ]]; then
  echo "# Sweep results (${category})" > "$result_file"
fi

run_one() {
  local sweep_name="$1"
  local ca="$2"
  local mb="$3"
  local mc="$4"

  local key="sweep=${sweep_name} category=${category} contrastive_alpha=${ca} manifold_beta=${mb} manifold_c=${mc}"
  local log_name="cat_${category}_${sweep_name}_ca_${ca}_mb_${mb}_mc_${mc}.log"
  local log_path="${log_dir}/${log_name}"

  : > "$log_path"

  echo "[RUN] ${key}"

  CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=CSA \
    --category=${category} \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --contrastive_alpha=${ca} \
    --manifold_beta=${mb} \
    --manifold_c=${mc} \
    2>&1 | tee "$log_path"

  test_line=$(grep "Test Results:" "$log_path" | grep -v "Periodic" | tail -n 1 || true)
  if [[ -n "$test_line" ]]; then
    echo "${key} | ${test_line}" >> "$result_file"
  else
    echo "${key} | Test Results: NOT FOUND" >> "$result_file"
  fi
}

# Sweep 1: contrastive_alpha changes; others fixed
for ca in "${values[@]}"; do
  run_one "contrastive_alpha" "$ca" "$fixed_manifold_beta" "$fixed_manifold_c"
done

# Sweep 2: manifold_beta changes; others fixed
for mb in "${values[@]}"; do
  run_one "manifold_beta" "$fixed_contrastive_alpha" "$mb" "$fixed_manifold_c"
done

# Sweep 3: manifold_c changes; others fixed
for mc in "${values[@]}"; do
  run_one "manifold_c" "$fixed_contrastive_alpha" "$fixed_manifold_beta" "$mc"
done

echo "Done. Logs in ${log_dir}, results in ${result_file}"
