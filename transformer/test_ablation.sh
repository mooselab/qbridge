#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# RQ 1
# BASE="../data/evaluation_data"
BASE="../larger_circuits/data/evaluation_data"
# RQ 3
# BASE="../data/testing_data"

# FINETUNE_DIR="../model/finetune"
FINETUNE_DIR="../larger_circuits/model/finetune"

LOGDIR="logs_test"
mkdir -p "$LOGDIR"

SEEDS=(1 2 3)

IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft similarity simon}"

DRYRUN="${DRYRUN:-0}"

print_cmd() {
  local log="$1"; shift
  printf 'CMD: '
  printf '%q ' "$@"
  printf '2>&1 | tee %q\n' "$log"
}

for seed in "${SEEDS[@]}"; do
  # "$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeCasablanca':7,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJakarta':7,'FakeJohannesburg':20,'FakeKolkata':27,'FakeLagos':7,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeNairobi':7,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
  "$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJohannesburg':20,'FakeKolkata':27,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
  | while IFS=$'\t' read -r backend dnode; do
      for cut in "${CUTS[@]}"; do
        finetune_model="${FINETUNE_DIR}/seed_${seed}/${backend}/${cut}/model.pth"

        if [[ ! -f "$finetune_model" ]]; then
          echo "no finetune model: ${FINETUNE_DIR}/seed_${seed}/${backend}/${cut}/model.pth, skip ${backend}/${cut}" >&2
          continue
        fi
       
        p="${BASE}/${backend}/table_by_cirid_${cut}.pkl"

        if [[ ! -f "$p" ]]; then
          echo "no data: ${BASE}/${backend}/table_by_cirid_${cut}.pkl, skip ${backend}/${cut}" >&2
          continue
        fi

        log="${LOGDIR}/seed_${seed}_${backend}_${cut}.log"
        echo "==== Testing seed=${seed} | ${backend} | cut=${cut} | d_node=61 ===="

        cmd=(
          "$PYTHON" -u train.py
          --seed "$seed"
          --table_by_cirid_file "$p"
          --backend "$backend"
          --mode test
          --ckpt "$finetune_model"
          --cut "$cut"
          "model.d_node=61"
        )

        print_cmd "$log" "${cmd[@]}"

        if [[ "$DRYRUN" -eq 0 ]]; then
          "${cmd[@]}" 2>&1 | tee "$log"
        fi
      done
    done
done