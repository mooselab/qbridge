#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-$(python -c 'import sys; print(sys.executable)')}"
export PYTHON

BASE="../larger_circuits/data/baseline_tunning_data"
# PRETRAIN_DIR="../model/pretrain"
PRETRAIN_DIR="../model/finetune"
LOGDIR="logs_finetune"
mkdir -p "$LOGDIR"

SEEDS=(1 2 3)
IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft simon similarity}"

BACKEND="${1:?Usage: sbatch submit_one_backend.sh BACKEND_NAME}"

# pretrain_model="${PRETRAIN_DIR}/seed_${seed}/${BACKEND}/model.pth"
# if [[ ! -f "$pretrain_model" ]]; then
#   echo "no pretrain model: $pretrain_model" >&2
#   exit 1
# fi

run_one() {
  local seed="$1"
  local cut="$2"
  pretrain_model="${PRETRAIN_DIR}/seed_${seed}/${BACKEND}/${cut}/model.pth"
  if [[ ! -f "$pretrain_model" ]]; then
    echo "no pretrain model: $pretrain_model" >&2
    exit 1
  fi

  local p="${BASE}/${BACKEND}/table_by_cirid_${cut}.pkl"
  local log="${LOGDIR}/seed_${seed}_${BACKEND}_${cut}.log"

  if [[ ! -f "$p" ]]; then
    echo "no data: $p, skip seed=${seed} ${BACKEND}/${cut}" >&2
    return 0
  fi

  echo "finetune: seed=${seed} backend=${BACKEND} cut=${cut} data=${p} ckpt=${pretrain_model}"

  "$PYTHON" -u train.py \
    --table_by_cirid_file "$p" \
    --backend "$BACKEND" \
    --mode finetune \
    --ckpt "$pretrain_model" \
    --finetune_epochs 1000 \
    --cut "$cut" \
    --seed "$seed" \
    model.d_node=61 2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  for cut in "${CUTS[@]}"; do
    run_one "$seed" "$cut"
  done
done

echo "All jobs finished for backend=${BACKEND}"