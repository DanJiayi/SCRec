#!/usr/bin/env bash
set -euo pipefail

category="Beauty"

contrastive_alphas=(0.2 0.5 1)
manifold_betas=(0 0.2 0.5 1)
manifold_cs=(0.2 0.5 1 2)

# contrastive_alphas=(0.5 2 5)
# manifold_betas=(0.2 2 5)
# manifold_cs=(0.2 5)

log_dir="logs_grid"
result_file="grid_test_results_${category}.txt"

mkdir -p "$log_dir"
# 如果结果文件不存在，则创建并写入表头；已存在则直接追加后续结果
if [[ ! -f "$result_file" ]]; then
  echo "# Grid results (${category})" > "$result_file"
fi

for ca in "${contrastive_alphas[@]}"; do
  for mb in "${manifold_betas[@]}"; do
    for mc in "${manifold_cs[@]}"; do
      if [[ "$mb" == "0" && "$mc" != "1" ]]; then
        continue
      fi

      # match_cnt=0
      # [[ "$ca" == "0.5" ]] && ((match_cnt++))
      # [[ "$mb" == "0.2" ]] && ((match_cnt++))
      # [[ "$mc" == "0.5" ]] && ((match_cnt++))
      # # 至少满足两个条件
      # if (( match_cnt < 2 )); then
      #   continue
      # fi


      key="category=${category} contrastive_alpha=${ca} manifold_beta=${mb} manifold_c=${mc}"

      # 如果该参数组合在结果文件中已存在，检查是否有正常结果
      # 只要有正常结果就跳过；只有全部都是 NOT FOUND 时才重跑
      if [[ -f "$result_file" ]] && grep -q -- "$key" "$result_file"; then
        # 检查是否有正常结果（不包含 NOT FOUND 的记录）
        if grep -- "$key" "$result_file" | grep -qv "Test Results: NOT FOUND"; then
          echo "[SKIP] ${key} already has valid result in ${result_file}"
          continue
        else
          echo "[RERUN] ${key} had only NOT FOUND before, rerunning"
        fi
      fi

      log_name="cat_${category}_ca_${ca}_mb_${mb}_mc_${mc}.log"
      log_path="${log_dir}/${log_name}"

      # 如果已有同名 log，则先清空，保证本次运行的日志是全新的
      : > "$log_path"

      echo "[RUN] category=${category} contrastive_alpha=${ca} manifold_beta=${mb} manifold_c=${mc}"

      CUDA_VISIBLE_DEVICES=0 python main.py \
        --model=CSA \
        --category=${category} \
        --lr=0.01 \
        --temperature=0.03 \
        --n_codebook=32 \
        --num_beams=20 \
        --contrastive_alpha=${ca} \
        --manifold_beta=${mb} \
        --manifold_c=${mc} \
        2>&1 | tee "$log_path"

      # 兼容不同格式的测试输出，匹配非周期性的最终 Test 结果
      test_line=$(grep "Test Results:" "$log_path" | grep -v "Periodic" | tail -n 1 || true)
      if [[ -n "$test_line" ]]; then
        echo "category=${category} contrastive_alpha=${ca} manifold_beta=${mb} manifold_c=${mc} | ${test_line}" >> "$result_file"
      else
        echo "category=${category} contrastive_alpha=${ca} manifold_beta=${mb} manifold_c=${mc} | Test Results: NOT FOUND" >> "$result_file"
      fi
    done
  done
done

echo "Done. Logs in ${log_dir}, results in ${result_file}"
