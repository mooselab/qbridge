#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
export FLASH_ATTENTION_DISABLE=1
export HYDRA_FULL_ERROR=1

PYTHON="${PYTHON:-$(python -c 'import sys; print(sys.executable)')}"

# RQ 2
# BASE="../data/evaluation_data"
BASE="../larger_circuits/data/evaluation_data"
# RQ 3
# BASE="../data/testing_data"

# FINETUNE_DIR="../model/finetune"
FINETUNE_DIR="../larger_circuits/model/finetune"
LOGDIR="logs_test"; mkdir -p "$LOGDIR"

SEEDS=(1 2 3)

IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft similarity simon}"

ts="$(date +%Y%m%d_%H%M%S)"

for seed in "${SEEDS[@]}"; do
  for cut in "${CUTS[@]}"; do
    p="${BASE}/full_model/table_by_cirid_${cut}.pkl"
    ckpt="${FINETUNE_DIR}/seed_${seed}/full_model/${cut}/model.pth"

    [[ -f "$p"    ]] || { echo "no data: $p, skip seed=${seed} $cut" >&2;  continue; }
    [[ -f "$ckpt" ]] || { echo "no model: $ckpt, skip seed=${seed} $cut" >&2; continue; }

    echo "test full model: seed=${seed}  cut=${cut}  data=${p}  ckpt=${ckpt}"
    "$PYTHON" train_v2.py \
      --seed "$seed" \
      --table_by_cirid_file "$p" \
      --mode test \
      --ckpt "$ckpt" \
      --cut "$cut" \
      model.d_node=61 \
      2>&1 | tee "${LOGDIR}/seed_${seed}_${cut}_${ts}.log"
  done
done