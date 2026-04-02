
## 2026-03-05 — Session: Synthetic datasets + environment fixes

### Codebase changes

1. **`scripts/generate_synthetic.py`** (NEW) — Script to generate synthetic GWAS datasets matching the pickle format expected by `SNPPCACHRDataModule`. Supports configurable participant count, SNP count, case count, causal SNP count, and random seed.

2. **`pukb/genes/1causal_10case/ld_all/`** (NEW) — Synthetic dataset: 100 participants, 100 SNPs, 10 cases (imbalanced), 1 causal SNP on chr1.

3. **`pukb/genes/1causal_50case/ld_all/`** (NEW) — Synthetic dataset: 100 participants, 100 SNPs, 50 cases (balanced), 1 causal SNP on chr1.

4. **`scripts/synthetic_1causal_10case.sh`** (NEW) — Run script for imbalanced dataset (d_model=16, batch_size=16).

5. **`scripts/synthetic_1causal_50case.sh`** (NEW) — Run script for balanced dataset.

6. **`models/DeepGP.py`** — Two compatibility fixes:
   - Added try/except fallback for `mamba_ssm.ops.triton.layernorm` vs `layer_norm` (Vim fork uses `layer_norm.py`, upstream uses `layernorm.py`)
   - Fixed typo: `if_devide_out` -> `if_divide_out` in the `partial(Mamba, ...)` call inside `create_block()` (the Vim fork corrected the spelling)

7. **`README.md`** — Added "Synthetic Test Datasets" section documenting dataset generation, descriptions, how to run, how to interpret results, and pickle file format specification.

### Environment changes (`/home/robert/envs/widedeepgp`)

The original environment had version mismatches. Changes made:

- **mamba_ssm**: Replaced pre-built `1.2.0.post1` (from `pip install mamba-ssm`) with **1.1.1 built from source** from the [Vision Mamba (Vim) fork](https://github.com/hustvl/Vim/tree/main/mamba-1p1p1). This provides the `bimamba_type` parameter that DeepGP requires. Built against PyTorch 2.10.0.
- **causal-conv1d**: Rebuilt from source (1.6.0) to match PyTorch 2.10.0 ABI (old pre-built .so had `c10::optional` vs `std::optional` symbol mismatch).
- **Removed stale `.so`**: Deleted orphaned `/home/robert/envs/widedeepgp/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so` left over from the old mamba-ssm 1.2.0 install.
- **transformers**: Downgraded from 5.x to **4.39.3** (mamba_ssm's `__init__.py` imports `GreedySearchDecoderOnlyOutput` which was removed in newer transformers).
- **PyTorch**: Ended up at **2.10.0+cu128** (not the 2.1.1 from the README, but both mamba_ssm and causal_conv1d are now compiled against it).
- **torchvision**: Updated to **0.25.0** (compatible with PyTorch 2.10).
- **numpy**: Pinned to **1.26.4** (< 2.0).

### Known issues / notes

- FutureWarnings about `torch.cuda.amp.custom_fwd` are harmless — mamba_ssm uses the deprecated API.
- FutureWarning about `timm.models.layers` import is harmless.
- PyTorch is 2.10.0 instead of the README's 2.1.1 — but the CUDA extensions are compiled against it so everything is ABI-compatible.
- The Vim fork was cloned to `/tmp/Vim/` for the build.
- Status: imports all work, but haven't yet confirmed a full training run completes end-to-end.

## 2026-03-05 (continued) — Dataset size fix + causal-conv1d fix

### Additional environment change

- **causal-conv1d**: Replaced 1.6.0 with **1.4.0 built from the Vim fork** (`/tmp/Vim/causal-conv1d/`). The 1.6.0 version's `causal_conv1d_fwd()` expects 8 args but the Vim fork's mamba_ssm 1.1.1 calls it with 7.

### Synthetic dataset regeneration

Regenerated both datasets with **8000 participants** (up from 100). The original 100-participant datasets only produced ~5 training batches, but `main_genome.py` has `val_check_interval=300` hardcoded, requiring at least 300 batches. With 8000 participants (6400 train, 5440 after val split, 340 batches at batch_size=16), this constraint is satisfied.

- `1causal_10case`: 8000 participants, 800 cases (10%), 7200 controls
- `1causal_50case`: 8000 participants, 4000 cases (50%), 4000 controls

Updated README.md and shell script comments to reflect new sizes.

### Key constraint discovered

`main_genome.py` uses `val_check_interval=300` — synthetic datasets must produce >= 300 training batches per epoch. With batch_size=16 and a 15% validation split, need at least ~5700 participants total (before train/test split).

## 2026-03-05 — Session: torch.load checkpoint fix

### Codebase changes

1. **`main_genome.py`** — Fixed `_pickle.UnpicklingError` when reloading best checkpoint after training. PyTorch 2.6+ changed `torch.load` to default to `weights_only=True`, which rejects `argparse.Namespace` objects saved by PyTorch Lightning in checkpoints. Added `torch.serialization.add_safe_globals([argparse.Namespace])` near the top of the file.
