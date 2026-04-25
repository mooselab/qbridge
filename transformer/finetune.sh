#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-python3}"

BASE="../data/baseline_tunning_data"
PRETRAIN_DIR="../model/pretrain"

LOGDIR="logs_finetune"
mkdir -p "$LOGDIR"

IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft similarity simon}"

DRYRUN="${DRYRUN:-0}"

print_cmd() {
  local log="$1"; shift
  printf 'CMD: '
  printf '%q ' "$@"
  printf '2>&1 | tee %q\n' "$log"
}

"$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeCasablanca':7,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJakarta':7,'FakeJohannesburg':20,'FakeKolkata':27,'FakeLagos':7,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeNairobi':7,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
| while IFS=$'\t' read -r backend dnode; do
    pretrain_model="${PRETRAIN_DIR}/${backend}/model.pth"

    if [[ ! -f "$pretrain_model" ]]; then
      echo "no pretrain model: $pretrain_model, skip $backend" >&2
      continue
    fi

    for cut in "${CUTS[@]}"; do
      p="${BASE}/${backend}/table_by_cirid_${cut}.pkl"

      if [[ ! -f "$p" ]]; then
       echo "no data: $BASE/$backend/table_by_cirid_$cut.pkl, skip $backend/$cut" >&2
        continue
      fi

      log="${LOGDIR}/${backend}_${cut}.log"
      echo "==== Finetuning ${backend} | cut=${cut} | d_node=${dnode} ===="

      cmd=(
        "$PYTHON" -u train.py
        --table_by_cirid_file "$p"
        --backend "$backend"
        --mode finetune
        --ckpt "$pretrain_model"
        --finetune_epochs 20
        --cut "$cut"
        "model.d_node=${dnode}"
      )

      print_cmd "$log" "${cmd[@]}"

      if [[ "$DRYRUN" -eq 0 ]]; then
        "${cmd[@]}" 2>&1 | tee "$log"
      fi
    done
  done

