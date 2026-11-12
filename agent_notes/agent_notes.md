
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

## 2026-11-10 (late evening) — Session: disk-backed genotype store + GPU-capacity benchmark

Asked to (i) measure how many SNPs fit on a 48 GB GPU and (ii) first replace the loader that
holds everything as float32 in RAM. Results and methods of (i):
`DeepWideGP-Scripts/scripts/deepgp_gpu_capacity/{methods,results}.md`.

### Codebase changes

1. **`genostore/`** (NEW package: `format.py`, `store.py`, `torch_decode.py`, `datamodule.py`,
   `convert.py`, `from_pickles.py`, `prefetch.py`, `README.md`) — compact on-disk genotype
   store (2-bit codes for dosage 0/1/2 + one byte per fractional dosage, ~0.29 byte/genotype),
   a `.pgen` converter, a memory-mapped reader, GPU-side decoding, and `SNPStoreDataModule`, a
   drop-in for `SNPPCACHRDataModule`. Read `genostore/README.md` first.
2. **`main_genome.py`** — two lines: imports the store module and uses it when the dataset
   directory contains `store.json`; otherwise the original pickle loader runs unchanged.
3. **`args_generator.py`** — two new optional flags, `--snp_stats` and `--pos_mode` (store
   datasets only).
4. **`tests/test_genostore.py`, `tests/test_genostore_datamodule.py`,
   `tests/test_genostore_pgen.py`** (NEW) — codec/store round trips, resume, equivalence of the
   new data module with the pickle loader on the synthetic dataset, `.pgen` conversion.

5. **`scripts/genostore_equivalence_check.sh`** (NEW) — trains the synthetic dataset through both
   loaders and prints the test metrics side by side.
6. **`pukb/genes/1causal_10case/store/`** (NEW, 2.7 MB) — the synthetic dataset converted with
   `python -m genostore.from_pickles`; select it with `--rd_mode store`.
7. **`README.md`** — short "Large datasets" section pointing to `genostore/README.md`.

Model code (`models/`, `layers/`) and `utils.py` were not touched. The equivalence runs left
`DeepGP/{logs,results}/DeepGP_snps_covs/1causal_10case/genostore_equiv_*` behind.

### Environment

- **No change to `/home/robert/envs/widedeepgp`.** `pgenlib` is still absent there; the
  converter and its test were run with `/mnt/sdg/robert/miniforge3/envs/gsem-py/bin/python`
  (pgenlib 0.95.2, numpy 2.4). Installing a freshly downloaded pgenlib wheel was blocked by the
  permission layer.

### Things worth knowing

- `/mnt/sdh` (where the `.pgen` files are) is one shingled HDD shared with other jobs. A plain
  reader got ~0.3 MB/s while other jobs were seeking on it; queued 1.28 MB readahead requests
  got 70 MB/s (`genostore/prefetch.py`). `fincore` reports "100% cached" for files you cannot
  write — it is not telling the truth there.
- The `.pgen` files carry dosage tracks: on chr22, ~4% of genotypes are fractional at 0.01
  resolution and hard-calling would set ~1.2% to missing. Hence a dosage-preserving store.
- Only small, ID-stripped test subsets were extracted (to `/dev/shm`, deleted afterwards).
  Bulk conversion of all participants was blocked as a data-handling decision for the user;
  the command is in `genostore/README.md`.
- chr2 (658,133 SNPs), not chr1 (600,143), is the longest sequence in the imputed set.
- `main_genome.py` with `devices=1` uses GPU 0, which other people occupy: set
  `CUDA_VISIBLE_DEVICES`.

### Headline numbers of the GPU benchmark (571 trials; details in the Scripts repo)

- Published model (d_model 32, 1 layer, fp32): 6.1 KiB of GPU memory per sample per SNP at peak
  (1.7 KiB kept until backward, summed over all chromosomes held; 4.35 KiB transient for the one
  chromosome being back-propagated). Rule of thumb for other sizes:
  (d_model/32) x (4.35 + 1.6 x layers) KiB; bf16 needs 0.55x.
- One sequence: chr1 length fits at batch 13, chr2 length at batch 12; 126k SNPs at batch 64.
- All 22 chromosomes, largest batch — imputed (7.81 M SNPs): 3 as written, 5 bf16, 11 with
  per-chromosome activation checkpointing, 20 with both. HapMap3 (1.05 M): 25 / 44 / 85 / 156.
- Checkpointing (reference implementation: `checkpointed_class()` in the Scripts repo's
  `bench_common.py`) gives identical gradients and costs 25–33% more time per step.
- Speed: 11–18 M sample-SNPs/s for the published size, so ~4 h per epoch for HapMap3 and
  26–41 h for all imputed SNPs on one GPU (216k training participants).
- bf16 is not a Trainer flag here: `nn.BCELoss` refuses autocast.
- The two loaders train identically within GPU run-to-run drift (same losses to 1e-7 for the
  first 20 steps from one seed).

---

## 2026-11-11 (afternoon) — Session: benchmark re-run on the true imputed SNPs, timing ladder

Follow-up to the entry above, after the user reported that `/mnt/sdh` was free.

### What changed

- **This repo:** only `genostore/README.md` (store sizes updated to the measured 0.277
  byte/genotype; disk-read time of the full `.pgen` set added). No code change.
- **Scripts repo, `scripts/deepgp_gpu_capacity/`:** `bench_common.py` (region specs `chrom:C`
  and `genome` for a store converted from the imputed files), `run_sweep.py` (stages
  `imputed_real`, `imputed_real_genome`, `speed_ladder`, `imputed_ladder`), `analyze.py` (tables
  `imputed_standin_vs_true`, `speed_by_batch`; `limit_bracketed` column), new
  `multilabel_head_check.py`; `methods.md` and `results.md` revised.
- **Environment:** no change.

### What was done with data

A second 512-participant, ID-stripped subset — this time of the imputed `.pgen` files, all 22
chromosomes (7,813,490 SNPs with dosages) — was extracted to `/dev/shm` with the idle-only disk
policy (638 GB read in 2 h 42 min), verified (0 mismatches on the audit rows) and deleted after
the runs. No genotype copy remains anywhere.

### Results worth knowing (610 trials in total)

- **Every limit is identical on the chromosomes' own imputed SNPs:** chr1 batch 13, chr2 batch 12
  (21 in bf16); whole genome 3 / 5 / 11 / 20 (as written / bf16 / checkpointed / both). Peak
  memory within 0.013% of the array-SNP stand-in runs. Memory depends on shape only.
- **Imputed store size, measured:** 0.277 byte/genotype (2.7% fractional dosages; the earlier
  "~4%" came from chr22 slices) -> chr1 55 GB, all imputed 721 GB, HapMap3 ~97 GB for 333,444
  participants.
- **Fastest measured setting per model, hours per epoch on one GPU, 215,872 participants.**
  HapMap3: d32x1 4.2 (fp32) / 3.4 (bf16); d64x1 8.1 / 5.6; d64x2 14.3 / 10.6; d128x2 30.5 / 21.2;
  d256x4 121 / 86. All imputed (bf16): d32x1 23–25; d64x1 43; d64x2 113; d128x2 226; d256x4 899.
  bf16 is 1.2–1.5x faster than fp32 as well as smaller; checkpointing costs about a quarter of
  the speed; the largest batch is not the fastest. Timings repeat to about 15%.
- **A multi-disease output layer costs nothing:** 1 -> 1,000 outputs adds 0.004 GiB and no
  measurable step time (the head sees 23 pooled vectors per person).
- **`val_check_interval=300` (batches) in `main_genome.py` is expensive at small batches:** each
  validation pass costs 3–5% of a training epoch, so at batch 25 (29 passes per epoch) validation
  would take longer than training. Early stopping (patience 10) then acts after 3,000 batches.
- Correction to the report behind the previous entry: over all trials the batch-consistency
  control holds to 1.1e-5 in fp32 (2e-6 at the published model size) and to bf16 rounding
  (<= 0.004; exactly equal in 104 of 122 bf16 runs). The first report quoted 2e-6 / 5e-5, figures
  that predated the larger-model stages of the sweep and were not refreshed.
- Pitfall met while queueing jobs: a shell loop of the form `while pgrep -f PATTERN; do sleep; done`
  never ends when PATTERN also appears in the loop's own command line (pgrep matches the
  wrapping shell). Wait on a PID or a marker file instead.

## 2026-11-11 (evening) — Session: the DeepGP widening plan (one model, 30 diseases)

Plan: `/home/robert/.claude/plans/call-this-deepgp-widening-crystalline-waterfall.md` (revision 3,
approved by Robert). Scientific description: `methods.md` at the repo root. Work is on branch
`widening` (not pushed). The single-disease code (`models/`, `layers/`, `utils.py`,
`main_genome.py`, existing tests) is untouched.

### Codebase changes (this repo)

- **Baseline commit** of what was uncommitted: Robert's April 0/1 label re-encoding and the
  November genostore work.
- **`genostore/convert.py`:** `--extract FILE` converts a listed SNP subset straight from the
  `.pgen` (reads only those records; sparse read-ahead that respects the idle-disk policy).
- **`genostore/datamodule.py`:** `store.json` may name `"labels"` and `"covariates"` (then only
  those columns are read, so a disease can never become a covariate), `samples_val.tsv` is used
  as the validation set when present, labels are checked per column (0, 1 or empty), a batch
  sampler can be attached (`train_sampler`), per-SNP training statistics are cached, and
  `"annotations": {"file", "names"}` points at per-chromosome annotation matrices. A dataset
  without `"labels"` behaves as before.
- **`genostore/from_pickles.py`:** 2-D labels become named columns.
- **`args_generator.py`:** `args_initial(argv=None)` and the multi-disease flags (inert for
  `main_genome.py`).
- **New package `widedeepgp/`:** `labels.py`, `samplers.py` (rotation of disease-balanced
  batches; every batch a function of seed and batch number), `losses.py` (sampling-rate
  correction, class weights, per-disease normalisation), `metrics.py`, `blocks.py` (sequence-block
  factory: the published block, or Mamba-3 in a bidirectional wrapper), `model.py`
  (`WideDeepGP`; same parameter names as the published class), `callbacks.py` (snapshots, run
  log, early stopping and Optuna pruning that survive a resume), `train.py` (`run_training`).
  Annotation vectors live in `model.py` (the plan named a separate `embed.py`; not needed).
- **New entry points:** `main_multilabel.py`, `score_snapshots.py`, `tune_optuna.py`.
- **New scripts:** `scripts/generate_synthetic_multilabel.py`, `scripts/multilabel_synthetic.sh`,
  `scripts/check_synthetic_run.py` (+ `multilabel_synthetic_seeds.csv`, the five-seed table its
  bands are computed from), `scripts/setup_mamba3_env.sh`.
- **New tests:** `tests/test_ml_{labels,sampler,loss}.py` (CPU), `test_ml_{model,train}.py`
  (GPU, fork environment), `test_ml_mamba3.py` (GPU, Mamba-3 environment),
  `tests/ml_helpers.py`, `tests/conftest.py` (`slow` marker).
- **Docs:** `methods.md` (new), README section "Many diseases at once".

### Codebase changes (DeepWideGP-Scripts)

New analysis folder `scripts/build-ard30-dataset/`: `make_snp_list.R` (HapMap3+ ∩ lab PGEN),
`build_dataset.py` (labels via the code of `find-ukb-ard-phenos`, covariates, splits,
annotations), `describe_dataset.py` (aggregate description), `test_build_dataset.py`,
`methods.md`, `results/` (aggregate tables only).

### Environment changes

- `/home/robert/envs/widedeepgp`: `optuna 5.0.0` added (with its dependencies alembic, colorlog,
  sqlalchemy, Mako, greenlet, tqdm). Nothing else changed.
- **New environment `/mnt/sdg/robert/envs/widedeepgp-m3`** for Mamba-3 (upstream `mamba_ssm`
  2.3.2.post1 from source, commit e9594ce; torch 2.10.0+cu128, numpy 1.26.4, Lightning 2.0.8),
  built by `scripts/setup_mamba3_env.sh`. Upstream and the Vision-Mamba fork are both called
  `mamba_ssm`, so they cannot share an environment.

### Data written (all under `/mnt/sdg/robert/deepwidegp/data/`, outside git)

- `reference/hm3plus_in_ukb_pgen.tsv`: the 1,258,200 HapMap3+ SNPs present in the lab PGEN.
- `genostore/hm3plus/`: those SNPs for all 333,444 participants (116 GB; verified, 0 mismatches
  on the audit rows of every chromosome).
- `datasets/ard30/hm3plus/`: train 215,872 / tune 20,000 / test 40,000 unrelated White British;
  57,445 related held out; `store.json`, `labels.json`, `annotations.npz`, `build_log.json`.
- `datasets/ard30/hm3plus_asian_black/`: 9,408 Asian and 7,631 Black participants, held out.

### Things worth knowing

- **Only 1,258,200 of the 1,444,196 HapMap3+ SNPs are in the lab PGEN** (MAF ≥ ~1% and other QC
  filters). The 10 Nov synthesis assumed all HapMap3 SNPs were present.
- **Field 34 (year of birth) is not in the phenotype extract.** It is reconstructed as
  assessment date − age at recruitment − 0.5 (error ≤ half a year).
- **The disease list changed after the plan was written** (J18, J22 dropped as infectious, I84
  and K64 merged, …). The builder reads the list from the selection file and checks all 90 case
  counts against it, so the file is authoritative; rarest is now I21 (3.7%).
- **A batch balanced on one disease is not balanced for the others.** With the real labels the
  rarest disease is 7.8% of a rotation's draws, not 50%. All 30 outputs learn from every batch;
  the loss divides by each person's sampling rate (0.55 for someone with none of the 30
  diseases, > 3.1 for the top 1%) and weights cases by the square root of the odds.
- **bf16:** pooling, head and loss stay in float32; averaging a chromosome in bf16 distorted
  the between-person signal by about 10% in a probe.
- **Lightning 2.0.8 traps handled in `train.py`/`callbacks.py`:** a resumed run repeats the
  validation its checkpoint was written after (the run log, early stopping and Optuna pruning
  ignore the repeat); the optimiser steps on an epoch's last batch (validation intervals are
  multiples of the accumulation factor); a step-triggered checkpoint is written before the step
  is counted (the resumable `last.ckpt` is written after validation only).
- **Mamba-3 (upstream) pitfalls:** its declared dependencies (tilelang, apache-tvm-ffi,
  quack-kernels) upgrade torch and numpy and make `import mamba_ssm` fail on Python 3.10, so it
  is installed with `--no-deps`; newer setuptools has no `pkg_resources`, which Lightning 2.0.8
  imports, so `setuptools<70` is pinned; the single-input kernel always computes in bf16
  internally; the first call of each kernel compiles for 1–2 minutes.
- **Do not edit a shell script while it is running** (bash reads it incrementally; an edit
  mid-run made the environment build execute a garbled line). Run long jobs from a frozen copy.
- `pkill -f PATTERN` kills the calling shell too when PATTERN is in its own command line; kill
  by PID.

## 2026-11-12 (night) — Session continued: dry run, capacity, pilot, and what the pilot found

Write-up: `DeepWideGP-Scripts/scripts/widedeepgp-pilot/results.md` (methods beside it). The
summary for the model is in `methods.md`, section 4.9.

### Codebase changes (this repo, branch `widening`)

- `widedeepgp/callbacks.py`: step metrics (throughput, learning rate, gradient norm) are logged
  on the batches Lightning 2.0 actually writes (they were dropped when my cadence and
  Lightning's did not coincide); peak GPU memory is tracked for training and validation apart;
  the metrics of each validation pass are kept so the best step's need not be recomputed.
- `widedeepgp/train.py`: validation intervals are rounded up (a run used to overshoot its
  budget by one interval); `results.json` gains seconds per batch, participants per second and
  peak memory; the tuning set is no longer scored a second time at the end of a run.
- `widedeepgp/model.py`, `blocks.py`: Mamba-3 encoders use the re-entrant checkpoint (its
  backward reads saved tensors twice); Mamba-3 head size is the largest power of two that
  divides the inner width.
- `score_snapshots.py`: per-person score files store identifiers as text (they needed pickle).
- `scripts/check_synthetic_run.py`: bands computed from the five-seed table, separate for
  annotated runs; `scripts/make_architecture_figure.py` and `figures/`; `genostore/README.md`
  updated; `methods.md` section 4.9 and measured capacity.

### Codebase changes (DeepWideGP-Scripts, branch `widening`, committed without switching the
shared working tree off `main`)

New analysis folder `scripts/widedeepgp-pilot/`: `run_pilot.py` (launcher; leaves one GPU
free; live cap in `<runs>/<experiment>/max_gpus`), `run_capacity.py` + `capacity_probe.py`,
`genetic_signal.py` (AUROC with genotypes permuted between participants), `simulated_control.py`
(traits simulated on the proxy's real genotypes; linear references; input variants),
`gwas_annotation.py` (GWAS z-scores of the 30 diseases as annotation), `analyze_pilot.py`,
`test_pilot.py`, `methods.md`, `results.md`, `results/`.

### Data written (under `/mnt/sdg/robert/deepwidegp/data/`)

`datasets/ard30/hm3plus_proxy8` (pilot proxy), `hm3plus_proxy8_sim` and its variants
`_selected`, `_oracle`, `_gwas`, `_gwasclump` (simulated-trait control), `hm3plus_proxy8_gwas`
and `hm3plus_proxy8_gwasclump` (real labels with GWAS z-scores as annotation),
`genostore/hm3plus_proxy8_sim_selected` (47 MB store of 2,476 selected SNPs). Runs:
`/mnt/sdg/robert/deepwidegp/runs/{dryrun,pilot,pilot_sim,sim_selected_tune}`; the pilot ran from
a frozen export of commit 8f03ca6 in `runs/pilot_code_8f03ca6`.

### Things worth knowing

- **The network learns the covariates and nothing from unselected SNPs at pilot scale**, on
  real labels and on simulated traits with known signal (AUROC 0.51 where linear methods get
  0.72–0.81). It learns slowly from GWAS-selected SNPs and on all SNPs when each SNP carries a
  signed association z-score as annotation. Mamba-3 and unsigned marks on causal SNPs do not
  help. Read `results.md` before planning any full-size run.
- **Measured at full size (1,258,200 SNPs, d32x1, one 48 GB GPU):** largest batch 21 / 36 / 69 /
  130 (fp32 / bf16 / checkpointing / both), 4.7 GPU-h per epoch-equivalent in fp32 and 3.6 in
  bf16; Mamba-3 9 / 18 / 98 / 122 and a kernel crash above ~122 participants x 105,260 tokens.
- **A model's AUROC on these diseases is mostly year of birth and sex** (0.66 from covariates
  alone on the proxy). Use `genetic_signal.py` to see what a model takes from the genotypes.
- **Thousands of one-SNP regions make the store loader crawl** (4 s per batch); write a small
  store of the selected SNPs instead (`simulated_control.py select`).
- **A GPU looks free between two short jobs.** The pilot launcher started a run on the GPU a
  capacity search was using; run such searches on the highest-numbered GPU, which the launcher
  always leaves alone.
- `pgrep -f PATTERN` matches a waiting shell whose own command line contains the pattern;
  write the pattern with a bracket (`run_capacity[.]py`) or wait on a file instead.
