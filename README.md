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
| `label_train.pkl` | List of 22 identical label arrays (1=ctrl, 2=case) | `[array(n_train,), ...]` |
| `label_test.pkl` | List of 22 identical label arrays | `[array(n_test,), ...]` |
| `covar_train.pkl` | Covariate matrix (train) | `array(n_train, 8)` |
| `covar_test.pkl` | Covariate matrix (test) | `array(n_test, 8)` |

## License
BSD 3-Clause License

Copyright (c) 2025, University of Oxford. & Novo Nordisk. All rights reserved.

## Citing
Please use the following BibTeX entry.
```
TBA


