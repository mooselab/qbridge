#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
export FLASH_ATTENTION_DISABLE=1

# BASE="../data/baseline_tunning_data"
BASE="../larger_circuits/data/baseline_tunning_data"

# PRETRAIN_DIR="../model/pretrain"
PRETRAIN_DIR="../model/finetune"
LOGDIR="logs_finetune"; mkdir -p "$LOGDIR"

SEEDS=(3)

for seed in "${SEEDS[@]}"; do
  # CKPT_SHARED="${PRETRAIN_DIR}/seed_${seed}/full_model/model.pth"

  for cut in phase qft simon ghz addition similarity; do
    CKPT_SHARED="${PRETRAIN_DIR}/seed_${seed}/full_model/${cut}/model.pth"
    p="${BASE}/full_model/table_by_cirid_${cut}.pkl"
    ckpt="$CKPT_SHARED"

    [[ -f "$p"    ]] || { echo "no data: $p" >&2;  continue; }
    [[ -f "$ckpt" ]] || { echo "no data: $ckpt" >&2; exit 1; }

    echo "finetune full model: seed=${seed}  cut=${cut}  data=${p}  ckpt=${ckpt}"
    python train_v2.py \
      --seed "$seed" \
      --table_by_cirid_file "$p" \
      --mode finetune \
      --ckpt "$ckpt" \
      --finetune_epochs 200 \
      --cut "$cut" \
      model.d_node=61 \
      2>&1 | tee "${LOGDIR}/seed_${seed}_${cut}.log"
  done
done