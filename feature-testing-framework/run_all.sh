#!/usr/bin/env bash
# Re-run the MI test with the new unified 18-feature set across every
# registered dataset. mindset's raw file lives outside data/ (data/ is
# gitignored and was never populated here) -- point at the user's Downloads copy.
set -e
source .venv/bin/activate
mkdir -p results/unified

run() {
  name=$1; shift
  echo "=== $name ==="
  python run.py --dataset "$name" "$@" --out "results/unified/${name}.csv" \
    2>&1 | tee "results/unified/${name}.log"
  echo
}

run mindset            --data "$HOME/Downloads/strong-ann.csv"
run synthetic
run umd
run umd_crowd
run umd_demographics
run irf_belong
run irf_burden
run mimic_osteoporosis  --data data/raw/mimic/mimic-raw.csv
run mimic_prostate_cancer --data data/raw/mimic/mimic-raw.csv
run mimic_ovarian_cyst  --data data/raw/mimic/mimic-raw.csv
run mimic_autism        --data data/raw/mimic/mimic-raw.csv
run mimic_cad           --data data/raw/mimic/mimic-raw.csv

echo "ALL DONE"
