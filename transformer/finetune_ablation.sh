#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-$(python -c 'import sys; print(sys.executable)')}"
export PYTHON

# BASE="../data/baseline_tunning_data"
BASE="../larger_circuits/data/baseline_tunning_data"
# PRETRAIN_DIR="../model/pretrain"
PRETRAIN_DIR="../model/finetune"

LOGDIR="logs_finetune"
mkdir -p "$LOGDIR"

SEEDS=(1 2 3)

IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft similarity simon}"

DRYRUN="${DRYRUN:-0}"
JOBS="${JOBS:-4}"

print_cmd() {
  local log="$1"; shift
  printf 'CMD: '
  printf '%q ' "$@"
  printf '2>&1 | tee %q\n' "$log"
}

run_one() {
  local seed="$1"
  local backend="$2"
  local cut="$3"

  local p="${BASE}/${backend}/table_by_cirid_${cut}.pkl"
  # local pretrain_model="${PRETRAIN_DIR}/seed_233/${backend}/model.pth"
  local pretrain_model="${PRETRAIN_DIR}/seed_${seed}/${backend}/${cut}/model.pth"
  local log="${LOGDIR}/seed_${seed}_${backend}_${cut}.log"

  if [[ ! -f "$pretrain_model" ]]; then
    echo "no pretrain model: $pretrain_model, skip seed=${seed} ${backend}/${cut}" >&2
    return 0
  fi

  if [[ ! -f "$p" ]]; then
    echo "no data: $p, skip seed=${seed} ${backend}/${cut}" >&2
    return 0
  fi

  echo "finetune: seed=${seed}  backend=${backend}  cut=${cut}  data=${p}  ckpt=${pretrain_model}"

  cmd=( "$PYTHON" -u train.py
    --table_by_cirid_file "$p"
    --backend "$backend"
    --mode finetune
    --ckpt "$pretrain_model"
    --finetune_epochs 1000
    --cut "$cut"
    --seed "$seed"
    "model.d_node=61"
  )

  print_cmd "$log" "${cmd[@]}"

  if [[ "${DRYRUN}" -eq 0 ]]; then
    "${cmd[@]}" 2>&1 | tee "$log"
  fi
}

export BASE PRETRAIN_DIR PYTHON LOGDIR DRYRUN JOBS
export -f print_cmd
export -f run_one

for seed in "${SEEDS[@]}"; do
  "$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJohannesburg':20,'FakeKolkata':27,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t61') for k in sorted(m)]" \
  # "$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeCasablanca':7,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJakarta':7,'FakeJohannesburg':20,'FakeKolkata':27,'FakeLagos':7,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeNairobi':7,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
  | while IFS=$'\t' read -r backend dnode; do
      for cut in "${CUTS[@]}"; do
        run_one "$seed" "$backend" "$cut" &

        while [[ $(jobs -r -p | wc -l) -ge ${JOBS} ]]; do
          sleep 2
        done
      done
    done
done

wait
echo "All jobs finished."