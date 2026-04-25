#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
set -euo pipefail

PYTHON="${PYTHON:-$(python -c 'import sys; print(sys.executable)')}"
export PYTHON


BASE="../data/baseline_tunning_data"
PRETRAIN_DIR="../model/pretrain"

LOGDIR="logs_finetune"
mkdir -p "$LOGDIR"

JOBS="${JOBS:-4}"

IFS=' ' read -r -a CUTS <<< "${CUTS:-addition ghz phase qft similarity simon}"

DRYRUN="${DRYRUN:-0}"

print_cmd() {
  local log="$1"; shift
  printf 'CMD: '
  printf '%q ' "$@"
  printf '2>&1 | tee %q\n' "$log"
}

export BASE PRETRAIN_DIR PYTHON LOGDIR DRYRUN
export -f print_cmd

"$PYTHON" -c "m={'FakeAlmaden':20,'FakeBoeblingen':20,'FakeBrooklyn':65,'FakeCairo':27,'FakeCambridge':28,'FakeCambridgeAlternativeBasis':28,'FakeCasablanca':7,'FakeGuadalupe':16,'FakeHanoi':27,'FakeJakarta':7,'FakeJohannesburg':20,'FakeKolkata':27,'FakeLagos':7,'FakeManhattan':65,'FakeMontreal':27,'FakeMumbai':27,'FakeNairobi':7,'FakeParis':27,'FakeRochester':53,'FakeSingapore':20,'FakeSydney':27,'FakeToronto':27,'FakeWashington':127};[print(f'{k}\t{m[k]+61}') for k in sorted(m)]" \
| while IFS=$'\t' read -r backend dnode; do
    for cut in "${CUTS[@]}"; do
      printf '%s\t%s\t%s\n' "$backend" "$dnode" "$cut"
    done
  done \
| xargs -n3 -P "$JOBS" bash -lc '
    backend="$1"; dnode="$2"; cut="$3"

    p="${BASE}/${backend}/table_by_cirid_${cut}.pkl"
    pretrain_model="${PRETRAIN_DIR}/${backend}/model.pth"
    log="${LOGDIR}/${backend}_${cut}.log"

    #
    if [[ ! -f "$pretrain_model" ]]; then
      echo "no pretrain model: $pretrain_model, skip $backend/$cut" >&2
      exit 0
    fi
    if [[ ! -f "$p" ]]; then
      echo "no data: $p, skip $backend/$cut" >&2
      exit 0
    fi

    cmd=( "$PYTHON" -u train.py
      --table_by_cirid_file "$p"
      --backend "$backend"
      --mode finetune
      --ckpt "$pretrain_model"
      --finetune_epochs 1000
      --cut "$cut"
      "model.d_node=${dnode}"
    )

    print_cmd "$log" "${cmd[@]}"

    if [[ "${DRYRUN}" -eq 0 ]]; then
      "${cmd[@]}" 2>&1 | tee "$log"
    fi
' _

