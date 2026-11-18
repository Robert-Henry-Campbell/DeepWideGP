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

**The model is one network trained end to end from dosages, with LD blocks as its pooling**
(`--input_mode blocks`, the default since 17 November 2026; [methods.md](methods.md) section
4.11, and the design with its decisions in DeepWideGP-Scripts
`scripts/widedeepgp-blocks/methods.md`, branch `widening`). Its first layer is a learned table
with `--embed_dim` numbers (16) per SNP: the participant's standardised dosage scales the SNP's
vector, and the vectors of an LD block's SNPs are summed (`--snp_code genotype` gives every SNP
three vectors instead, for 0, 1 and 2 copies, interpolated for fractional dosages). A learned
identity vector per block and per chromosome is added and a small MLP shared by all blocks makes
one token per block. One mixer, the published bidirectional Mamba block (`--mixer bimamba1`) or a
transformer encoder layer (`--mixer transformer`), runs over the block tokens and the covariate
token; the mean over the tokens, beside the covariate vector, feeds the head.
The blocks come from a partition file (`--partition`, required:
`data/reference/partitions/prive431`, `ldsplit2000` or `ldsplit5000`, checked against the
dataset's store and recorded by digest in every checkpoint). Each block's sum is divided by the
square root of its SNP count. The table has its own AdamW group: its learning rate
`--table_kappa` √batch / √SNPs, a per-step weight decay `--table_decay` and a large epsilon
`--table_eps` (so an entry with a tiny, inconsistent gradient barely moves). The rest of the recipe is
Sigurdsson et al. 2023's (methods.md 4.11): learning rate 1e-4, weight decay 1e-3, dropout 0.5,
batch 64, 40% of SNPs masked on every training step (`--input_mask`), a plateau schedule, early
stopping after 16 passes once 2,000 steps are done (`--stop_burnin`). Training draws uniform shuffled
batches and uses plain cross-entropy at natural prevalence (`--sampler shuffled --gamma 0`, methods.md
section 5; the rotation of disease-balanced batches with a square-root class weight is kept as a
comparison variant). The scaling study starts every run from the ridge warm start (`--init_encoder`, scaled to unit-RMS
block tokens by `make_warm_start.py`), fold-routed: `--folds <blocks/folds.npz>` gives the table one copy per fold of the
block ridge, a training participant pooled with the copy fitted without them and every other participant with the copy
of their pseudo-fold, so the warm start carries no participant's own label (methods.md 4.11). A run tunes itself: Optuna over `--table_kappa` and `--table_decay`
(`--tune_space table`; a loaded table gets a slower, undecayed range than one that starts at zero), or also the learning
rate, weight decay and dropout (`run`).
`pretrain_blocks.py` pretrains the
first layer by masked-SNP prediction on the training rows and writes a file for `--init_encoder`.

**The SNP-token mode** (`--input_mode snps`) is the published network widened: one sequence of
SNP tokens per chromosome, each with its own embedding and bidirectional Mamba encoder, pooled
and combined with the covariates by one attention layer (methods.md sections 4.1 to 4.9). It is
the model of the pilot and of the classic single-disease comparator, not the reported
WideDeepGP. Read methods.md section 4.9 before using it on unselected SNPs: in the pilot on real
data (DeepWideGP-Scripts, branch `widening`, `scripts/widedeepgp-pilot/`; written results in git
history at 784009f) it ended slightly below a logistic model on the covariates and learned
nothing detectable from them.

**The linear PRS path** (`--wide on` or `only`, SNP mode only; methods.md section 4.10) gives
every SNP and disease a weight with a ridge penalty beside, or instead of, the network, and a run
with `--wide on` searches in two stages (the linear model alone at nine prior variances side by
side with each disease keeping its best, then the network on top of that PRS). It was the default
from 13 to 16 November 2026, is off by default since, and is never the reported model: its runs
are linear comparators and ablations.

```bash
# a dataset <data_dir>/<dataset>/<variant>/ whose store.json names its "labels" and "covariates"
# (genostore/README.md); without --data_dir, datasets are looked for in pukb/genes. A partition
# folder <data>/reference/partitions/<name>/ holds the partition.npz the run's blocks come from.
# a run: the search of methods.md section 8 on the free GPUs minus one, then the chosen model
python main_multilabel.py --data_dir <data_dir> --label <dataset> --rd_mode <variant> \
    --partition <data>/reference/partitions/prive431 --mixer transformer --d_model 64 --e_layers 4 \
    --n_heads 4 --embed_dim 16 --precision bf16 --batch_size 256 --exp_name <run>
python score_snapshots.py --run runs/default/<run> --split val --snapshots best   # the chosen model
# one fixed setting, no search, on the GPU this process sees
CUDA_VISIBLE_DEVICES=<gpu> python main_multilabel.py ... --tune_trials 0 --table_kappa 0.03 --table_decay 1
# the first layer pretrained by masked-SNP prediction on the training rows, then a run started from it
CUDA_VISIBLE_DEVICES=<gpu> python pretrain_blocks.py --data_dir <data_dir> --label <dataset> --rd_mode <variant> \
    --partition <data>/reference/partitions/prive431 --embed_dim 16 --out <dataset dir>/blocks/pretrained_prive431.pt
python main_multilabel.py ... --init_encoder <dataset dir>/blocks/pretrained_prive431.pt   # --freeze_encoder: kept fixed
# the SNP-token mode (the published network, constant learning rate), with the linear PRS path beside it
python main_multilabel.py ... --input_mode snps --lr_schedule constant --warmup_steps 0 \
    --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 [--wide on]
tensorboard --logdir runs/

# three-minute check of the SNP-token mode on six synthetic traits (generated on first use)
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/multilabel_synthetic.sh
```

| Flag | Meaning | Default |
|---|---|---|
| `--input_mode` | `blocks`: a per-SNP table pooled by LD block, then one mixer over the block tokens and the covariate token; `snps`: the published per-chromosome SNP encoders | `blocks` |
| `--partition` | blocks mode: a `partition.npz` or its folder (`data/reference/partitions/prive431`, `ldsplit2000`, `ldsplit5000`), checked against the store's chromosomes, variant counts and variant IDs; `uniform:<k>` (k SNPs per block) for tests | none (required in blocks mode) |
| `--mixer` | `bimamba1` (the published block), `transformer` (a pre-norm `nn.TransformerEncoderLayer`, feed-forward width 4d, GELU) or `mamba3` (needs `scripts/setup_mamba3_env.sh`) | `bimamba1` |
| `--snp_code` | blocks mode: `dosage` (one vector per SNP times its standardised dosage) or `genotype` (three vectors per SNP, for 0, 1 and 2 copies, interpolated; reads raw dosages) | `dosage` |
| `--embed_dim` | blocks mode: width of the per-SNP table (d_e) | 16 |
| `--table_kappa`, `--table_decay`, `--table_eps` | blocks mode: the table's learning rate κ √batch / √SNPs, its AdamW weight decay (a per-step shrink of every entry) and its Adam epsilon (about eight times a null SNP's per-entry gradient at batch 256, measured 1.3e-8 at the top rung) | 0.03 (searched 0.01 to 0.03), 1 (searched 0.3 to 3), 1e-7 |
| `--tune_min_validations` | validation passes a trial completes before the pruner may stop it | 2 |
| `--init_encoder <file.pt>`, `--freeze_encoder` | blocks mode: start the first layer from a file written by `pretrain_blocks.py` (masked-SNP pretraining) or by the ridge warm start (DeepWideGP-Scripts `scripts/widedeepgp-blocks/make_warm_start.py`), checked for its partition, SNP code and width; keep the loaded parameters fixed | none, off |
| `--labels a,b` | train on a subset of the dataset's diseases (one disease = the single-disease comparator) | all |
| `--snp_stats` | SNP standardisation: `train` (training-split mean and SD) or `store` (all store samples) | `train` (`main_genome.py`: `store`) |
| `--sampler` | `shuffled`: uniform shuffled batches; `rotation`: each batch is balanced on one disease in turn, with the loss's sampling-rate correction (a comparison variant, methods 5.3) | `shuffled` in `main_multilabel.py` (parser: `rotation`) |
| `--loss_outputs` | `all`: every output learns from every batch; `target`: only the balanced disease | `all` |
| `--gamma`, `--w_max` | class-weight exponent (0 none: plain cross-entropy at natural prevalence; 0.5 square root; 1 parity) and cap | 0 in `main_multilabel.py` (parser: 0.5), 20 |
| `--no_sampling_correction` | do not divide each person's loss by their sampling rate | off |
| `--annot` | SNP mode: per-SNP annotation vectors as a gain and offset on the token embedding | off |
| `--final_pool` | pooling of the mixer's tokens: blocks mode `mean` (the only setting it accepts); SNP mode `atten` (the published attention across the 23 vectors), `mean` or `last` | `mean` in blocks mode, `atten` in SNP mode |
| `--precision bf16`, `--grad_ckpt`, `--accumulate N` | mixed precision (the first layer, covariate encoder and mixer in bf16; the mean over tokens, the head and the loss in float32), activation checkpointing (per mixer layer in blocks mode, per chromosome in SNP mode), gradient accumulation | off, off, 1 |
| `--warmup_steps`, `--lr_schedule` | linear learning-rate warm-up (optimiser steps); `constant`, `cosine` (to zero over the budget) or `plateau` (x0.2 after 10 passes without improvement); the table's and the network's rates follow the same schedule | 0, `plateau` (the SNP-mode recipes pass `constant`) |
| `--val_per_epoch`, `--stop_patience`, `--stop_burnin`, `--max_epoch_equiv` | validations per epoch-equivalent, patience, optimiser steps before early stopping may act, budget | 4, 16, 2000, 30 |
| `--input_mask` | blocks mode: share of SNPs set to missing on every training step, the rest rescaled | 0.4 |
| `--tune_space` | what a run's search tunes: `table` (the table's pace and decay) or `run` (also learning rate, weight decay, dropout) | `run` |
| `--eval_batch_size` | batch size for validation and scoring | the training batch size |
| `--seed` | random seed (initial weights, dropout, batch order) | 1 (`main_genome.py`: 33) |
| `--wide` | linear PRS path (SNP mode only): `off` (the network alone), `on` (beside the network), `only` (PRS and covariates alone) | `off` |
| `--prs_var`, `--wide_kappa` | the PRS's ridge penalty (prior variance of the PRS) and the pace of its weights (sets their learning rate) | 1 (a grid in a search), 0.03 (the runs of 13 to 16 November 2026 used 0.1) |
| `--tune_trials` | trials of a run's search: Optuna over the network's learning rate, weight decay and dropout, plus `--table_kappa` and `--table_decay` in blocks mode (with `--wide on`, the second stage, the network on top of the linear model); 0 = no search, train the given settings once | 12 |
| `--prs_grid` | with `--wide on` or `only`: prior variances of the linear stage, trained side by side; each disease keeps its best (`none`: one variance, `--prs_var`) | 0.003,0.01,…,10,30 (half-decade steps) in a search |
| `--tune_gpus` | GPUs for the trials: `auto` (the free ones minus one), a list, or `none` (here, one after another) | `auto` |
| `--val_min_participants`, `--val_max_epoch_equiv`, `--min_participants` | validate at most once per this many participants, at least once per this many epoch-equivalents; train at least this many participants | 20,000, 1, 128,000 |
| `--resume`, `--init_from <snapshot>` | continue a run on the same data with the same settings (it may train longer); start a new run from saved weights (new output layer if the diseases differ in number, names or order); both refuse weights trained on another block partition | |

A run that tunes itself holds `deep/`, with its Optuna study and one folder per trial, plus
`results.json` (the chosen stage and model, every trial's settings, value and cost) and
`search.json` (its arguments; a rerun with other arguments is refused). With `--wide on` the
stages are `linear/` (the grid run; `linear/per_disease/` is the linear model with each disease's
best prior variance, a run folder of its own, and `per_disease.csv` lists the choices) and
`joint/`, and `results.json` records the linear model alone as the ablation `linear_only`.
Rerunning the command continues the search. Each trial's folder holds `config.json` (every
setting, plus the code directory, its git commit (none for a copy outside git), a `git_dirty`
flag for uncommitted changes to tracked .py files, and the Python interpreter; in blocks mode
also a `blocks` section: the partition and its digest, the numbers of blocks and tokens, the
table's width and code, the parameters by module (table, identity vectors, block MLP, mixer,
covariate encoder, head) with their split into embedding and non-embedding
parameters, and the sha256 of the `--init_encoder` file), TensorBoard files, `metrics.csv`,
weight snapshots with `snapshots.csv` (what each cost; the table is stored in bfloat16),
`checkpoints/best.ckpt` and `last.ckpt`, and `results.json`.
Tests: `python -m pytest tests/test_ml_*.py -q`. On the CPU: `tests/test_ml_blocks.py` (blocks
mode with the transformer mixer: the partition and its checks, the pooling against numpy segment
sums, the ridge PRS in the first layer, the table's optimiser group, `--init_encoder`,
a run with bf16 snapshots, scoring and a resume, a search and a seed retrain, masked
pretraining, and a learning check on the synthetic traits) and the linear path's
`tests/test_ml_wide.py`. On a GPU: `tests/test_ml_blocks_gpu.py` (both mixers in bf16,
activation checkpointing, the memory of the largest planned model at 2,000 tokens) and the
SNP-mode model and training tests.

## License
BSD 3-Clause License

Copyright (c) 2025, University of Oxford. & Novo Nordisk. All rights reserved.

## Citing
Please use the following BibTeX entry.
```
TBA

