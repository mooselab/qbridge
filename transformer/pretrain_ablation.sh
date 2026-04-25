#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-python3}"
BASE="../data/baseline_training_data"
LOGDIR="logs_pretrain"
mkdir -p "$LOGDIR"

SEEDS=(233)

for seed in "${SEEDS[@]}"; do
  echo "=============================="
  echo "Running seed=${seed}"
  echo "=============================="

  "$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeCasablanca':7,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJakarta':7,'FakeJohannesburg':20,'FakeKolkata':27,'FakeLagos':7,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeNairobi':7,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
  | while IFS=$'\t' read -r backend dnode; do
      p="${BASE}/${backend}/table_by_cirid.pkl"
      if [[ ! -f "$p" ]]; then
        echo "no data: $p, skip ${backend}" >&2
        continue
      fi

      echo "==== Running ${backend} (model.d_node=61, seed=${seed}) ===="

      cmd=(
        "$PYTHON" -u train.py
        --table_by_cirid_file "$p"
        --backend "$backend"
        --mode pretrain
        --seed "$seed"
        model.d_node=61
      )
      "${cmd[@]}" 2>&1 | tee "${LOGDIR}/${backend}_seed${seed}.log"
    done
done