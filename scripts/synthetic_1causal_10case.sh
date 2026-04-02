#!/bin/bash

# Synthetic test: 8000 participants, 100 SNPs, 1 causal SNP, 800 cases (10%)
python main_genome.py \
  --label 1causal_10case \
  --dm snps_covs \
  --rd_mode ld_all \
  --exp_name synthetic_1causal_10case \
  --snp_embed cov \
  --final_pool atten \
  --batch_size 16 \
  --d_model 16 \
  --e_layers 1 \
  --dropout 0.2 \
  --lr 7e-4 \
  --n_heads 4 \
  --n_fcn 2 \
  --l2 1e-4 \
  --num_workers 4
