#!/usr/bin/env bash

export FLASH_ATTENTION_DISABLE=1

SEEDS=(233)

LOGDIR="logs_pretrain"
mkdir -p "$LOGDIR"

for seed in "${SEEDS[@]}"; do
  echo "=============================="
  echo "Running seed=${seed}"
  echo "=============================="

  python train_v2.py \
    --seed "$seed" \
    --table_by_cirid_file ../data/baseline_training_data/full_model/table_by_cirid.pkl \
    --mode pretrain \
    model.d_node=61 \
    2>&1 | tee "${LOGDIR}/seed_${seed}.log"
done