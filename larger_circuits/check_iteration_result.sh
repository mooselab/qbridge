#!/bin/bash

progress_dir="data/baseline_tunning_progress"

for f in "$progress_dir"/*_similarity.progress.json; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    completed=$(python - <<'PY' "$f"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data.get("completed_iterations", "NA"))
PY
)
    echo "$name  completed_iterations=$completed"
done