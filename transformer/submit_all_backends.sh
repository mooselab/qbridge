#!/bin/bash
set -euo pipefail

BACKENDS=(
  FakeAlmaden
  FakeBoeblingen
  FakeBrooklyn
  FakeCairo
  FakeCambridge
  FakeCambridgeAlternativeBasis
  FakeGuadalupe
  FakeHanoi
  FakeJohannesburg
  FakeKolkata
  FakeManhattan
  FakeMontreal
  FakeMumbai
  FakeParis
  FakeRochester
  FakeSingapore
  FakeSydney
  FakeToronto
  FakeWashington
)

for backend in "${BACKENDS[@]}"; do
  echo "Submitting ${backend}"
  sbatch --job-name="ft_${backend}" submit_one_backend.sh "${backend}"
done