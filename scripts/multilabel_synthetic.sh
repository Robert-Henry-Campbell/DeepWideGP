#!/bin/bash
# Multi-disease smoke run on the synthetic six-trait dataset: about three minutes on one GPU.
#
#   CUDA_VISIBLE_DEVICES=<idle gpu> bash scripts/multilabel_synthetic.sh [run_name] [extra flags, e.g. --annot]
#
# Generates the dataset on first use (pukb/genes/multi6/store, not tracked by git), trains for
# three epoch-equivalents and checks every trait against the simulation's own ceiling.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
NAME=${1:-multi6}
[ -f pukb/genes/multi6/store/store.json ] || $PY scripts/generate_synthetic_multilabel.py --name multi6
$PY main_multilabel.py --label multi6 --rd_mode store --snp_stats train --experiment synthetic --exp_name "$NAME" --tune_trials 0 --wide off \
  --input_mode snps --input_mask 0 --lr_schedule constant --warmup_steps 0 --stop_patience 0 --stop_burnin 0 \
  --final_pool atten --d_model 16 --n_heads 4 --e_layers 1 --batch_size 16 --lr 7e-4 --l2 1e-4 --dropout 0.2 \
  --num_workers 4 --max_epoch_equiv 3 --val_per_epoch 2 --stop_patience 0 "${@:2}"
$PY scripts/check_synthetic_run.py "runs/synthetic/$NAME" pukb/genes/multi6/store/truth.json
