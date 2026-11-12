Multimodal deep learning enhances genomic risk prediction for cardiometabolic diseases in UK Biobank
===
Leveraging Bidirectional Mamba's capacity to capture long-range dependencies across whole-genome SNP data, we propose DeepGP, a deep learning approach to improve genetic prediction for cardiometabolic diseases while providing interpretable insights into risk factors.

## Papar Information
- Authors:  Taiyu Zhu, Upamanyu Ghose, Héctor Climente-González, Joanna M. M. Howson, Sile Hu, Alejo Nevado-Holgado
- Affiliations: University of Oxford, Novo Nordisk Research Centre Oxford
- Preprint: TBA


## Dataset
- UK Biobank Resource under Application Number 53639

## Use
- To train and test the model, run:
  ```
   bash scripts/t2d.sh
  ```



## Directory Hierarchy
```
|—— .gitignore
|—— README.md
|—— args_generator.py
|—— layers
|    |—— Embed.py
|    |—— SelfAttention_Family.py
|    |—— Transformer_EncDec.py
|—— main_genome.py
|—— models
|    |—— BaseModel.py
|    |—— DeepGP.py
|—— scripts
|    |—— t2d.sh
|—— utils.py
```
## Code Details
### Tested Platform
- software
  ```
  Python: 3.10.13
  PyTorch 2.1.1
  PyTorch Lightning 2.0.8
  numpy 1.26.4
  ```
- hardware
  ```
  CPU: AMD EPYC 7R13 Processor
  GPU: NVIDIA A10 Tensor Core GPU
  ```

## References
We would like to express our gratitude to the following GitHub repositories for their valuable  code and contributions:
- [Mamba](https://github.com/state-spaces/mamba)
- [Vision Mamba](https://github.com/hustvl/Vim)

## Synthetic Test Datasets

Two synthetic datasets are provided for testing the pipeline without UK Biobank access. Each has 8000 participants, 100 SNPs spread evenly across 22 chromosomes, 1 causal SNP (on chromosome 1), and 8 random covariates.

### Generating the data

The data generator script lives at `scripts/generate_synthetic.py`. The pre-generated datasets are in `pukb/genes/`. To regenerate or create new variants:

```bash
python scripts/generate_synthetic.py --name 1causal_10case --n_participants 8000 --n_snps 100 --n_cases 800 --n_causal 1
python scripts/generate_synthetic.py --name 1causal_50case --n_participants 8000 --n_snps 100 --n_cases 4000 --n_causal 1
```

Run `python scripts/generate_synthetic.py --help` for all options.

### Dataset descriptions

| Dataset | Participants | SNPs | Cases | Controls | Causal SNPs |
|---|---|---|---|---|---|
| `1causal_10case` | 8000 | 100 | 800 | 7200 | 1 |
| `1causal_50case` | 8000 | 100 | 4000 | 4000 | 1 |

- **1causal_10case**: Highly imbalanced (10% case rate). Tests whether the model can learn a signal under class imbalance — a harder scenario where naive classifiers may predict all-control.
- **1causal_50case**: Balanced (50% case rate). Tests whether the model can learn the causal SNP signal when class balance is not an issue.

### Running the synthetic experiments

```bash
bash scripts/synthetic_1causal_10case.sh
bash scripts/synthetic_1causal_50case.sh
```

These scripts use smaller hyperparameters (`d_model=16`, `batch_size=16`) suitable for the small dataset size and will train on GPU.

### Interpreting results

The model outputs standard binary classification metrics:
- **AUC**: Area under the ROC curve. Random = 0.5. For these tiny synthetic datasets, AUC > 0.6 suggests the model is picking up the causal signal.
- **Accuracy**: Fraction of correct predictions. For `1causal_10case`, a naive all-control classifier gets 90% accuracy, so this metric alone is misleading — check AUC and MCC instead.
- **MCC**: Matthews correlation coefficient. Ranges from -1 to +1 (0 = random). More informative than accuracy under class imbalance.
- **Precision / Recall / F1**: Standard metrics. Recall (sensitivity) measures what fraction of true cases the model finds.

The datasets use an 80/20 train/test split (6400 train / 1600 test). These datasets are intended for **pipeline validation** (verifying the code runs end-to-end), not for drawing scientific conclusions.

### Pickle file format

Each dataset directory (`pukb/genes/<name>/ld_all/`) contains:

| File | Contents | Shape |
|---|---|---|
| `genes.pkl` | List of 22 arrays of SNP names | `[array(n_chr1,), ..., array(n_chr22,)]` |
| `pos.pkl` | List of 22 arrays of positional indices | `[array(n_chr1,), ..., array(n_chr22,)]` |
| `snp_train.pkl` | List of 22 genotype arrays (train) | `[array(n_train, n_chr_i), ...]` |
| `snp_test.pkl` | List of 22 genotype arrays (test) | `[array(n_test, n_chr_i), ...]` |
| `label_train.pkl` | List of 22 identical label arrays (0=ctrl, 1=case) | `[array(n_train,), ...]` |
| `label_test.pkl` | List of 22 identical label arrays | `[array(n_test,), ...]` |
| `covar_train.pkl` | Covariate matrix (train) | `array(n_train, 8)` |
| `covar_test.pkl` | Covariate matrix (test) | `array(n_test, 8)` |

## Large datasets: disk-backed genotype store

The pickle loader above holds every genotype as float32 in RAM, which stops working long before
UK Biobank scale (chromosome 1 of the imputed set alone would need ~745 GiB). For large data,
convert the genotypes once into the compact store in `genostore/` and point the dataset directory
at it with a `store.json`; `main_genome.py` then reads mini-batches from disk and decodes them
on the GPU, with the same batches reaching the model. See `genostore/README.md`.

```bash
# worked example on the synthetic data: same dataset, read through the store
python -m genostore.from_pickles --src pukb/genes/1causal_10case/ld_all --dst pukb/genes/1causal_10case/store
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/genostore_equivalence_check.sh
```

## Many diseases at once: WideDeepGP

`widedeepgp/` widens the model to one output per disease, trained on all of them together.
The single-disease code above (`main_genome.py`, `models/`, `layers/`, `utils.py`) is unchanged.
[methods.md](methods.md) describes the data, the model and the training, and lists every
difference from the single-disease model.

**Read [methods.md](methods.md) section 4.9 before training at full size.** In the pilot on real
data (DeepWideGP-Scripts, branch `widening`, `scripts/widedeepgp-pilot/results.md`) the network
ended slightly below a logistic model on the covariates and learned nothing detectable from
unselected SNPs unless each SNP carried signed association statistics (then about a third of
what a clumped polygenic score adds); on simulated traits it learned from the SNPs only when
each SNP came with signed association statistics, or slowly when the SNPs had been selected by a
GWAS.

```bash
# a dataset <data_dir>/<dataset>/<variant>/ whose store.json names its "labels" and "covariates"
# (genostore/README.md); without --data_dir, datasets are looked for in pukb/genes
CUDA_VISIBLE_DEVICES=<gpu> python main_multilabel.py --data_dir <data_dir> --label <dataset> --rd_mode <variant> \
    --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 --exp_name <run>
python score_snapshots.py --run runs/default/<run> --split val      # every snapshot, with a covariates-only baseline
CUDA_VISIBLE_DEVICES=<gpu> python tune_optuna.py --study <name> --n_trials 20 -- \
    --data_dir <data_dir> --label <dataset> --rd_mode <variant> <the other training arguments>
tensorboard --logdir runs/

# three-minute check on six synthetic traits (generated on first use)
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/multilabel_synthetic.sh
```

| Flag | Meaning | Default |
|---|---|---|
| `--labels a,b` | train on a subset of the dataset's diseases (one disease = the single-disease comparator) | all |
| `--snp_stats` | SNP standardisation: `train` (training-split mean and SD) or `store` (all store samples) | `train` (`main_genome.py`: `store`) |
| `--sampler` | `rotation`: each batch is balanced on one disease in turn; `shuffled` | `rotation` |
| `--loss_outputs` | `all`: every output learns from every batch; `target`: only the balanced disease | `all` |
| `--gamma`, `--w_max` | class-weight exponent (0 none, 0.5 square root, 1 parity) and cap | 0.5, 20 |
| `--no_sampling_correction` | do not divide each person's loss by their sampling rate | off |
| `--annot` | per-SNP annotation vectors as a gain and offset on the token embedding | off |
| `--mixer` | `bimamba1` (the published block) or `mamba3` (needs `scripts/setup_mamba3_env.sh`) | `bimamba1` |
| `--precision bf16`, `--grad_ckpt`, `--accumulate N` | mixed precision, per-chromosome checkpointing, gradient accumulation | off, off, 1 |
| `--warmup_steps`, `--lr_schedule` | linear learning-rate warm-up (optimiser steps); `constant` or `cosine` decay after it | 0, `constant` |
| `--val_per_epoch`, `--stop_patience`, `--max_epoch_equiv` | validations per epoch-equivalent, patience, budget | 4, 8, 30 |
| `--eval_batch_size` | batch size for validation and scoring | the training batch size |
| `--seed` | random seed (initial weights, dropout, batch order); replicate runs use 42, 43, 44 | 42 (`main_genome.py`: 33) |
| `--resume`, `--init_from <snapshot>` | continue a run on the same data with the same settings (it may train longer); start a new run from saved weights (new output layer if the diseases differ in number, names or order) | |

A run directory holds `config.json` (every setting, plus the code directory, its git commit
(none for a copy outside git), a `git_dirty` flag for uncommitted changes to tracked .py files,
and the Python interpreter), TensorBoard files, `metrics.csv`, weight snapshots with
`snapshots.csv` (what each cost), `checkpoints/best.ckpt` and `last.ckpt`, and `results.json`.
Tests: `python -m pytest tests/test_ml_*.py -q` (the model and training tests need a GPU).

## License
BSD 3-Clause License

Copyright (c) 2025, University of Oxford. & Novo Nordisk. All rights reserved.

## Citing
Please use the following BibTeX entry.
```
TBA

