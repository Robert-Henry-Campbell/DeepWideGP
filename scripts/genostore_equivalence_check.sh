#!/bin/bash
# Train the synthetic 1causal_10case dataset twice with identical settings and seed:
#   1) through the original pickle loader           (--rd_mode ld_all)
#   2) through the disk-backed genotype store        (--rd_mode store, --snp_stats train)
# and print the test metrics side by side. The two loaders feed the model the same batches
# (tests/test_genostore_datamodule.py), so the metrics should agree closely.
#
# Usage:  CUDA_VISIBLE_DEVICES=<idle gpu> bash scripts/genostore_equivalence_check.sh
# Needs:  pukb/genes/1causal_10case/store  (python -m genostore.from_pickles --src pukb/genes/1causal_10case/ld_all --dst pukb/genes/1causal_10case/store)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
COMMON="--label 1causal_10case --dm snps_covs --snp_embed cov --final_pool atten --batch_size 16 --d_model 16 --e_layers 1 --dropout 0.2 --lr 7e-4 --n_heads 4 --n_fcn 2 --l2 1e-4 --num_workers 4"
$PY main_genome.py $COMMON --rd_mode ld_all --exp_name genostore_equiv_legacy
$PY main_genome.py $COMMON --rd_mode store --snp_stats train --exp_name genostore_equiv_store
$PY - <<'PYEOF'
import csv, glob, os
base = "DeepGP/results/DeepGP_snps_covs/1causal_10case"
rows = {}
for name in ("genostore_equiv_legacy", "genostore_equiv_store"):
    f = max(glob.glob(os.path.join(base, name, "results_*.csv")), key=os.path.getmtime)
    rows[name] = next(csv.DictReader(open(f)))
keys = [k for k in rows["genostore_equiv_legacy"] if k != "pheno"]
print(f"{'metric':<12}{'pickle loader':>16}{'genotype store':>16}{'difference':>12}")
for k in keys:
    a, b = float(rows["genostore_equiv_legacy"][k]), float(rows["genostore_equiv_store"][k])
    print(f"{k:<12}{a:>16.4f}{b:>16.4f}{b - a:>12.4f}")
PYEOF
