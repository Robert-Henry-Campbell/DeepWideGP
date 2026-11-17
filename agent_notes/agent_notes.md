
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

## 2026-11-12 (afternoon) — Session continued on Opus 5.5: second seeds, audit, corrections

The session moved from Fable 5.2 to Opus 5.5 at about 12:30. Write-ups:
DeepWideGP-Scripts (branch `widening`) `scripts/widedeepgp-pilot/results.md` and `methods.md`;
model summary in `methods.md` section 4.9 here.

### What happened

- All 18 pilot runs finished overnight (9 arms x seeds 1 and 2). The genotype-permutation
  scoring was rerun for all 18 (the seed-1 values reproduced the earlier ones to 3e-8) and the
  pilot tables and figures were regenerated with both seeds. With two seeds, target-only
  (-0.0011) and no sampling-rate correction (-0.0023) differ from the default under the pre-set
  rule, both in the direction the plan's defaults predicted; every decision stands.
- An audit by 13 agents (plan vs work, tests, the pilot write-up, both methods documents, notes
  and memory, data inventory and governance; an adversarial check of every finding; a
  completeness critic) checked about 1,230 claims (1,085 passed, 147 flagged) and confirmed
  about 100 problems. A second round (5 agents on disjoint files, each reviewed) and the lead
  fixed them. The final test run, after the last code change, is recorded in the next entry.
- A 200-step bf16 counterpart of the full-size dry run
  (`/mnt/sdg/robert/deepwidegp/runs/dryrun/bf16_b18`, run from `runs/pilot_code_8f03ca6`)
  matched the 32-bit run (loss within 0.2%, AUROC 0.6110 against 0.6108) in 0.74 times the time
  per step and 0.59 times the memory.
- Gate evidence that lived only in the session scratchpad or in git-ignored folders was copied
  (small files only, synthetic data) to `/mnt/sdg/robert/deepwidegp/runs/gate_evidence/` (README
  there): plan 8C one-label regression, 8F two-worker Optuna check, the five-seed band runs, and
  Mamba-3 gate 5 (`multi6_mamba3_gate5`). The pilot used the code of commit 8f03ca6 (an export
  in `runs/pilot_code_8f03ca6`), which contains every code change of the previous entry except
  717c755.

### Code changes (this repository)

- Exact sampling correction for odd batch sizes: the sampler takes round(B/2) cases
  (`samplers.cases_per_batch`) and the loss's sampling rates now use that share (hp key
  `case_share`; `labels.rotation_rates`/`sampling_rate` take it; old models default to 0.5).
- `main_multilabel.parse_args` / `tune_optuna`: training-split SNP statistics
  (`--snp_stats train`) unless the flag is given; `tune_optuna.load_study` reads a study without
  writing to its journal.
- `train.run_training` refuses a model or dataset without covariates (`--dm snps`, `--use_cov`,
  no covariates named); `check_resume_tables` refuses to resume over other SNP positions or
  annotation vectors; models record `annot_digest` (a model with annotations refuses different
  matrices; older annotated models load with a warning); `config.json` records `code_dir`,
  `git_commit`, `git_dirty` (tracked .py files differ from the commit) and `python`.
- `score_snapshots.py --snapshots best` follows `results.json`'s `best_checkpoint`.
- `genostore/format.py`, `convert.py`: stores are created owner-only (0700 directories, 0600
  files) whatever the umask.
- Docstrings (`callbacks.py`, `__init__.py`), `scripts/make_architecture_figure.py` (Figure 1
  draws the annotation maps once, shared by all chromosomes; 96 annotations and a flag).
- Tests: odd-batch rates and loss, parse_args defaults, refusals, digest checks on load and on
  resume, provenance, read-only study loading, the resume fallback read from `results.json`;
  new `tests/test_genostore_permissions.py`.

### Code changes (DeepWideGP-Scripts, branch `widening`)

- `widedeepgp-pilot/genetic_signal.py`: file lock around its CSV merge; other experiments write
  `genetic_signal_<experiment>.csv`; rows replaced per (arm, seed[, label]); the data module is
  rebuilt when a run's dataset differs; labels without cases are left out of every mean; scores
  the checkpoint `results.json` names.
- `simulated_control.py`, `gwas_annotation.py`: refuse to run unless the dataset's third table
  is a draw from the training split (never the real test set); the selected-SNP store is
  owner-only; Figure 5 reads every run at the end of training.
- New: `position_code_capacity.py`, `check_control_data_path.py`, `test_pilot_checks.py`;
  `analyze_pilot.py` reports the gain over the covariates; `capacity_probe.py` passes the
  annotations so no false digest warning appears.
- `build-ard30-dataset`: `describe_dataset.py` writes no maximum over participants; new
  `proxy_region_loci.py` and `results/proxy_region_loci.csv` (the 20 loci behind the proxy
  region, listed from memory on 11 Nov).

### Data and environment

- `data/genostore/` made owner-only (`chmod -R go-rwx`).
- `docs/2026-11-11_widening_plan_rev3.md`: a copy of the approved plan (untracked, like the
  synthesis document, so it survives a clean-up of `~/.claude`).
- No environment changes.

### Things worth knowing

- Every pilot network ends slightly below a logistic model on the 12 covariates (default
  -0.0014 in mean tuning AUROC). Any redesign must show, on real labels, a positive gain over
  that model and a positive genotype-dependent part.
- Pan-UKB summary statistics are not out-of-sample: Pan-UKB is UK Biobank and includes our
  tuning and test participants. FinnGen R12 is on GRCh38; data/reference/map_hm3_plus.rds has
  pos_hg38 for all but 1,006 of its 1,444,196 SNPs (606 of the model's 1,258,200) and an rsID
  for all but 408 (260 of the model's, which carry a chr:pos_ref_alt identifier instead); no
  model SNP lacks both, so a match by GRCh38 position with an rsID fallback needs no lift-over
  tool.
- The genotype stores hold every sample of the lab PGEN, including 120 + 16 participants on the
  withdrawal list of 25 April 2024; every table excludes them. Whether to rebuild the stores
  without them is Robert's decision.
- `scripts/find-ukb-ard-phenos` in DeepWideGP-Scripts (another session's folder) is in no
  commit, but `build-ard30-dataset/build_dataset.py` imports it.
- The legacy single-disease test rewrites the tracked synthetic output
  `DeepGP/results/.../synthetic_1causal_10case_test/preds.pkl`; it is left uncommitted.
- Shell pitfall: in `a && b && (x) & (y) & wait`, the whole `a && b && (x)` chain runs in a
  background subshell, so variables it sets are unset in `(y)`. Set variables on lines of their
  own first.
- A second Claude session (started 13:54 on 12 Nov) is planning PRS comparators in this project
  and shares the working trees.

## 2026-11-12 (evening) — Final checks, last fixes, commits

Everything in the two entries above, with the corrections, is committed on branch `widening`
(not pushed): DeepWideGP 2310856 (code and tests) and the commit that adds this entry
(documents, figure, notes); DeepWideGP-Scripts 7a1d789 (`scripts/widedeepgp-pilot`,
`scripts/build-ard30-dataset`, committed through a temporary index; the working tree stays on
`main`).

### What happened

- Review rounds after the corrections round, each by fresh agents reading the code, the
  documents and the primary sources (runs, logs, reference files, Lightning's source):
  1. a final check (code; documents): 8 low code issues, 2 medium and 13 low document issues;
  2. a check of those fixes: 5 low code issues, 1 medium and 5 low document issues;
  3. a check of those: 1 medium and 4 low code issues (what a resume keeps), 4 low document
     issues;
  4. a check of those: 6 low issues (copied run directories, the last checkpoint's name,
     wording);
  5. a check of those, cut short when the session restarted, whose toy Lightning flows showed
     that the new `RunCheckpoint` did not restore checkpoints saved by a plain
     `ModelCheckpoint` (Lightning keys a callback's state by its class name) nor a folder
     reached by another path; fixed, and the flows rerun on the fixed code.
  Every issue was fixed. Most of them concern `--resume`, which the pilot used once (a copied
  run folder) and which full-size runs will rely on.
- Tests of the code as committed (2310856); logs in
  `/mnt/sdg/robert/deepwidegp/runs/gate_evidence/final_tests_2026-11-12/`:
  - On the CPU, 22:13:36 to 22:14:07 UTC: the label, sampler, loss and Mamba-3 tests and the
    four genostore files, 44 passed and 9 skipped; the CPU tests of `test_ml_train.py` and
    `test_ml_model.py`, 13 passed (their GPU tests skip there); DeepWideGP-Scripts' pilot and
    dataset tests, 27 passed.
  - On the GPU: not run yet. Another user's job has held all five GPUs since about 18:08.
    `run_gpu_suites_when_free.sh` in that folder, started detached at 22:16 UTC, waits until
    two GPUs are free (Robert's rule: the free GPUs minus one), then runs the quick and slow
    training tests, the model tests and the capacity probe on an export of 2310856
    (`code_2310856/`, outside git) and writes `gpu_summary.txt`; it gives up after 24 hours.
    The last full GPU run, `third_run/` (17:38 to 17:51 UTC: quick training tests 16, model
    9, slow 3 passed), was on the code before `RunCheckpoint` and the version-ranked
    `last_checkpoint`.
  - Earlier runs kept there: `first_run/`, `second_run/` and `third_run/`, full runs that
    passed on the code of earlier rounds; `fourth_run_cpu_only/`, the CPU suites just before
    the last `RunCheckpoint` fix; and `stopped_run/`, stopped at 16:55 so that
    `check_resume_args` could be added, whose two FAILED lines
    (`test_a_resume_with_rewritten_annotations_is_refused`,
    `test_one_label_lands_in_the_published_band`) are the tests the stop killed; both pass in
    every full run.

### Code changes

DeepWideGP:
- `widedeepgp/train.py`:
  - `check_resume_tables(ckpt_path, hp, loss=None)` also refuses a checkpoint trained on other
    labels (`label_names`, names or order), on a training split with other label counts, or
    with any other saved setting except those in `RESUMABLE` (`total_optimizer_steps`,
    `grad_ckpt`). Given the run's loss module, it also compares the loss's constants with the
    checkpoint's (relative tolerance 1e-6), which covers checkpoints saved before `case_share`
    was recorded. The loss's constants are persistent buffers and the optimiser's state is
    restored too, so a resume with other labels or settings would have trained with the
    checkpoint's values while `config.json` recorded the new ones. The 20 last checkpoints of
    the pilot's 18 runs and of `pilot_sim/selected_long_s1` (its `last.ckpt` and
    `last-v1.ckpt`) all pass, against settings and a loss rebuilt from their own saved
    settings with the live code.
  - New `check_resume_args`: refuses a resume whose `--snp_stats`, `--scaling`,
    `--batch_size`, `--accumulate`, `--seed`, `--steps_per_epoch_equiv` or `--val_per_epoch`
    differ from the run's existing `config.json` (`RUN_ARGS`), which the resumed run
    overwrites.
  - New `last_checkpoint`: `--resume` continues from `last.ckpt` or the highest
    `last-vN.ckpt`, and refuses when another last checkpoint is newer than that one (a copy
    that did not keep the files' times). Lightning writes `last-vN.ckpt` beside a `last.ckpt`
    it did not write itself, as in `pilot_sim/selected_long_s1`; before, `--resume` always read
    `last.ckpt`, so a further resume of that folder would have started again from the copied
    6-epoch-equivalent weights.
  - Checkpoints are written by `RunCheckpoint` (below); `--init_from` passes the source's label
    names to `load_weights`.
  - `code_version` runs `git --no-optional-locks status`, so recording the code version never
    writes the index of a checkout that another process may be using. The module docstring
    lists `git_dirty`.
- `widedeepgp/callbacks.py`:
  - New `RunCheckpoint`, a `ModelCheckpoint` that, when a checkpoint's saved folder is named
    differently from the run's (a copied run directory, or the same folder through a symlink
    or a relative `--run_dir`) and the checkpoint files are there, moves the saved best and
    last paths into the run's folder before restoring them. Lightning restores the best score
    only when the folder is named exactly as when the state was saved (this fsspec reports the
    local protocol as a tuple, so Lightning keeps paths as given), so a copy, once resumed,
    started a new best (`best-v1.ckpt`) compared with nothing from before the resume, and
    `results.json` could report a later and worse pass as the run's best. Its `state_key` is
    `ModelCheckpoint`'s, so the pilot's checkpoints (saved by a plain `ModelCheckpoint`)
    restore into it; with its own class name they would not have, and resuming any of them
    would also have started a new best. Checked with a toy LightningModule on the CPU, 21
    training runs of which 19 resume (in place, copies with and without file times, copies of
    copies, a symlink, a moved folder, a relative run path, pilot-era checkpoints in place and
    copied): each keeps the best checkpoint and score of an in-place resume, and the three
    copied originals that were compared are unchanged. A copy without `best.ckpt`, or one taken
    while the run was writing, falls back to Lightning's behaviour.
  - `EarlyStoppingOnce` keeps the resumed run's `--stop_patience`; Lightning's
    `EarlyStopping.load_state_dict` would put back the checkpoint's, silently.
- `widedeepgp/model.py`: `load_weights(..., source_labels=None)`: with `allow_new_head`, the
  output layer also starts afresh when the source's label names differ in names or order.
  Before, a list of other diseases of the same length loaded the saved rows as they were, each
  into another disease's output.
- `score_snapshots.py`: `--snapshots last` scores the checkpoint `last_checkpoint` picks.
- `args_generator.py`: the `--resume` help text says what a resume keeps.
- Tests. `test_ml_train.py`, on the CPU: `test_resume_checks_every_table` (every branch, on
  checkpoints holding only settings and loss constants),
  `test_resume_keeps_the_batches_and_snp_scaling`,
  `test_a_resumed_run_keeps_its_own_stopping_patience`,
  `test_a_resume_starts_from_the_latest_last_checkpoint`,
  `test_run_training_resumes_from_the_latest_last_checkpoint`,
  `test_a_copied_run_directory_carries_on_with_its_best_and_last`; on the GPU,
  `test_starting_from_the_weights_of_a_run_on_other_diseases` and
  `test_a_copied_run_resumes_as_if_in_place`; the GPU test of a refused resume also checks
  that `run_training` refuses other arguments, other settings and (for a checkpoint without
  `case_share`) other loss constants, that `config.json` is left alone, and that a resume with
  the original annotations goes ahead; the code-version test computes the expected commit the
  way `code_version` does (a copy inside another checkout has none); the `results.json` test
  also covers `--snapshots last`. `test_ml_model.py`: the same two diseases in the other order
  get a fresh output layer, and a source without names loads as before.

DeepWideGP-Scripts:
- `capacity_probe.py` passes the annotations to `model_hparams` only where it takes them, so
  the probe also runs on the frozen pilot code (`DEEPGP_REPO=runs/pilot_code_8f03ca6`).
- `position_code_capacity.py`: docstrings (least squares with the cut-off of
  `numpy.linalg.lstsq(rcond=None)`, which is the default only from NumPy 2.0; it is not exact).

Documents: the bf16 dry run at full length agreed with the 32-bit one in training loss to within
0.2% (0.19% at step 149), not 0.1%; `git_dirty` is documented; "nothing" became "nothing
detectable" or "next to nothing" where the pilot shows a small effect; `map_hm3_plus.rds` lacks
pos_hg38 for 1,006 SNPs (606 of the model's), and 408 of its identifiers (260 of the model's)
are chr:pos_ref_alt, not rsIDs, but no model SNP lacks both; the pilot methods describe the bf16
repeat of the dry run and the code each dry run used (the 32-bit run's code was edited after
the run had loaded it and before it was committed, in ways that do not change that run); the
limitations say what the proxy can rank; the dataset methods say that `build_log.json` holds
each covariate's minimum and maximum; results.md section 7 describes a GRCh38 position match,
not a lift-over; methods.md section 4.9 gives the GWAS-annotation scheme as a ninth, added
during the pilot; "signed" information is said to be needed on unselected SNPs only (the
network learned, slowly, from GWAS-selected SNPs without a sign); the least-squares cut-off is
named as lstsq's with rcond=None; methods.md section 7, the README and the pilot methods'
"Longer training" bullet describe what `--resume` and `--init_from` keep and which checkpoint a
resume reads.

Correction to the 12 Nov (night) entry above: its "Things worth knowing" says the network
learns "nothing from unselected SNPs". The evidence is nothing detectable on the real labels
(-0.0003 to +0.0003 for the eight planned arms, every interval including zero) and next to
nothing on the simulated traits (0.50 to 0.52).

### Things worth knowing

- `--resume` continues the same run: the same data, settings, batches and seed; it may train
  longer (budget, early-stopping patience) and switch activation checkpointing. To start from
  saved weights with anything else, use `--init_from`. A single-disease model saved by
  `main_genome.py` records no disease names, so `--init_from` loads its output layer as it is
  when the sizes match.
- To extend a finished run, copy its folder and resume the copy (the copy keeps the run's best
  checkpoint and score), or resume it in place. `pilot_sim/selected_long_s1` was copied with
  the pilot's code, so it holds stale `best.ckpt` and `last.ckpt` beside `best-v1.ckpt` and
  `last-v1.ckpt`; `results.json`, `--snapshots best|last` and `--resume` all pick the `-v1`
  files.

## 2026-11-12 (late evening) — Cleanup and seed 42, at Robert's request

Robert asked to delete the caches that are no longer useful, and to "set the same seed for
everything, make it 42".

### Code changes

- `main_multilabel.parse_args`: `--seed` defaults to 42 (`SEED`), in the same way as
  `--snp_stats` defaults to `train`; a `--seed` that is given wins. `args_generator.py` keeps 33
  for `main_genome.py` (decision A2: the published configuration stays reproducible).
- `tune_optuna.py`: `--sampler_seed` defaults to 42, and trials train with seed 42 through
  `parse_args`.
- `tests/test_ml_train.py` (`test_the_multi_disease_entry_points_scale_snps_on_the_training_split`)
  checks both defaults; methods.md section 6 ("Seeds": 42, replicates 42, 43, 44) and the README
  flag table say so.
- Not changed yet: the cohort split (`build_dataset.py`, seed 20261111) and the proxy draw
  (`make_proxy_dataset.py`, seed 20261112). Changing them re-splits the cohort and means
  rebuilding the dataset tables, so Robert was asked first. The pilot's analysis scripts keep
  their seeds, so they still reproduce the reported numbers.
- A run started with another seed now resumes only with its own `--seed` (for example
  `--seed 33` for the dry runs); `check_resume_args` refuses otherwise.

### Environment (cleanup)

- Deleted under `/mnt/sdg/robert/deepwidegp/runs` (2.5 GiB): the snapshot weights of the pilot,
  control and dry runs (961 files), and the checkpoints and snapshots of the 11 Optuna trials
  (`sim_selected_tune`). Every run keeps `config.json`, `metrics.csv`, `results.json`, `tb/` and
  `snapshots.csv` (whose `file` column now names deleted files, so `score_snapshots.py
  --snapshots all` no longer works on them); the pilot and control runs keep
  `checkpoints/best*.ckpt` and `last*.ckpt`.
- Deleted: `DeepWideGP/runs/synthetic` (git-ignored; the Mamba-3 gate run is kept in
  `runs/gate_evidence/multi6_mamba3_gate5`), the legacy regression test's untracked outputs of
  12 Nov, and on the root filesystem the review agents' scratch folders and pytest's temporary
  folders (1.6 GiB; 25 GB free after). `preds.pkl` is back to its committed version.
- Moved: the genostore equivalence-check outputs of 11 Nov (untracked
  `DeepGP/{logs,results}/.../genostore_equiv_*`) to
  `runs/gate_evidence/genostore_equiv_2026-11-11/`.
- Kept: Robert's own test outputs of 2 April (untracked, in `synthetic_1causal_10case_test`),
  `docs/`, `.codex`, everything in `data/`, the frozen code copies, and the export
  `runs/gate_evidence/final_tests_2026-11-12/code_2310856` that the waiting GPU test job uses.

## 2026-11-13 (night, about 00:30 UTC) — One seed (1); a linear PRS inside the model; a search inside every run; stores without the withdrawn participants

Robert's decisions of 12–13 Nov (Opus 5.5 continuing the Fable session):

- "nix the 42 seed thing ... if you already did substantial work with a single seed (not 42)
  revert to that": the project's single seed is **1**, the seed the pilot, its positive control
  and its tuning ran with (commit 7ba8b09). This supersedes the seed-42 entry above. The cohort
  split (20261111) and the proxy draw (20261112) are unchanged.
- Rebuild the genotype stores without the withdrawn participants (see "Data" below).
- Remove the two single-participant maxima from the committed history of DeepWideGP-Scripts.
- Every run of WideDeepGP redoes its hyper-parameter search, because the scaling study trains
  on nested subsets of the training split and regularisation should change with training size.
- A linear PRS inside the model. Robert then suggested REGENIE/plink2 for it; after the
  explanation (an external GWAS on the same participants would leak the training labels into a
  fixed input and differ from the deep part's fit; the PRS comparators of the other plan use
  REGENIE already) he agreed to the in-model path.

### Code changes (this repository, branch `widening`)

- `widedeepgp/model.py`: the linear PRS path, `--wide off|on|only`. For disease j,
  logit_j = deep_j + b_j + M^(-1/2) Σ_s x_s θ_sj + Σ_k u_k v_kj, with x the standardised dosages
  the encoders read (M = 1,258,200 at full size, so θ holds 37.7M weights for 30 diseases),
  computed in float32 outside bf16 autocast. θ and v start at zero, b_j at logit(π′_j) (the
  weighted prevalence, π′ = πc/Z); with `--wide on` the deep head's last layer also starts at
  zero, so a run starts as the linear model. Ridge penalty added to the loss (not AdamW decay):
  mean over diseases of λ_j‖θ_j‖², λ_j = r_j/(2Nτ), r_j = π′(1−π′)/(π(1−π)) = c_j/Z_j² (the
  loss's curvature relative to an unweighted logistic loss; 1.11–3.82 for the 30 diseases), τ =
  `--prs_var`, the prior variance of the PRS on the logit scale. Own parameter group with
  learning rate κτ√B/(r̄√M) (`--wide_kappa`, geometric-mean r̄) unless `--lr_wide` is given;
  `--wide_freeze` keeps θ fixed. `load_weights` treats the path's parameters as optional (older
  checkpoints load with the path at zero) and per-disease (a new label set starts it afresh).
  `--wide only` builds no encoders, so it trains on the CPU; it refuses annotations and
  `--loss_outputs target`.
- `widedeepgp/tuning.py` (new): the search inside every run. `--wide on`: stage "linear"
  (`--wide only`; batch ≥ 256, accumulate 1, twice the budget, patience 10, one validation per
  epoch-equivalent, no pruning; prs_var 1e-2–30, wide_kappa 1e-2–0.3, lr 3e-4–1e-2, all log),
  then stage "joint" (deep network on top, started from the linear stage's best with θ fixed:
  lr 1e-4–3e-3, l2 1e-6–1e-2, dropout 0–0.4, successive halving). The joint model is the run's
  model only if strictly better than the linear one. `--wide only`: the linear stage alone;
  `--wide off`: stage "deep". TPE, 5 random start-up trials, sampler seeded with the run's seed
  plus the worker's index; one worker process per GPU (`--tune_gpus auto` = free GPUs minus one,
  waits for two free, picked again for each stage; a list; or `none`); exact trial counts per
  worker; a lock file (one search per run) and `search.json` (a rerun with other arguments is
  refused); trials a crash left running are marked failed and redone; afterwards every trial's
  weights are deleted except the chosen trial's and the linear stage's best. `results.json` at
  the run level names the chosen stage and trial (relative path).
- `widedeepgp/train.py`: `wide_hparams` (τ, κ, the learning rate, freezing, the SNP statistics);
  `load_wide_init` for `--init_wide_from` (checks positions, sequence lengths, labels, label
  counts and SNP statistics); the validation interval in participants seen (at least every
  `--val_min_participants`, at most every `--val_max_epoch_equiv`) and a budget floor
  (`--min_participants`); the snapshot cap; resume checks refuse a switched `--wide`.
- `widedeepgp/callbacks.py`: `snapshot_steps(..., max_count)`; RunLog logs the PRS learning
  rate, gradient norm and penalty; RunCheckpoint keeps ModelCheckpoint's state key and re-roots
  saved paths when the run folder is reached through another path; EarlyStoppingOnce keeps the
  run's patience on resume.
- `main_multilabel.py`: defaults `--wide on`, `--tune_trials 12`, `--tune_trials_linear 8`,
  `--tune_gpus auto`, `--val_min_participants 20000`, `--val_max_epoch_equiv 1`,
  `--min_participants 128000`, `--snapshots_max 40`; a run without `--exp_name` gets one (so its
  workers share the folder); `--tune_trials 0` trains the given settings once.
- `tune_optuna.py`: search spaces "linear" and "deep"; `open_study(..., n_startup_trials,
  prune)`. `args_generator.py`: the new flags, inert for `main_genome.py` (its `--gpus` int is
  untouched, hence `--tune_gpus`). `score_snapshots.py`: a tuned run is scored through its chosen
  trial. `scripts/multilabel_synthetic.sh`: `--tune_trials 0 --wide off` (the published check).
- `tests/test_ml_wide.py` (new, 20 tests: 16 CPU, 1 slow CPU learning check, 3 GPU). CPU suite
  `tests/test_ml_*.py tests/test_genostore*.py -m "not slow"`: 73 passed, 28 skipped (GPU),
  13 Nov 00:20 UTC.
- `methods.md` (new section 4.10; sections 5.2, 6, 7, 8 and 10 updated) and `README.md`.

### DeepWideGP-Scripts (branch `widening`; nothing pushed)

- History rewritten to drop the two single-participant maxima: 3861ac5→cf662eb,
  e06fb7c→9b987ff, d206659→2deff11, 7a1d789→d1a3511. The old commits survive only in the local
  reflog of `widening` until git expires them; no branch reaches them, so a push would not
  carry them.
- New (committed separately): `build-ard30-dataset/rebuild_stores_without_withdrawn.sh` and
  `check_rebuilt_store.py`.

### Data

- Genotype stores rebuilt without the participants on the withdrawal list of 25 April 2024:
  the Asian/Black store is done (17,055 → 17,039 participants; the converter's own audit
  found 0 mismatches on all 22 chromosomes; against the old store, 0 of 407 million genotypes
  differ for 2,000 participants on chromosomes 1, 6 and 22, the variant lists are identical
  and no withdrawn participant is left; old copy deleted). The main HapMap3+ store
  (333,444 → 333,324) was converting at the time of writing.

### Pending

- GPU tests of this commit (all five GPUs held by another user's jobs since about 22:15 UTC).
- Gates before the first real run: the linear model on the simulated traits against GBLUP
  (CPU run `runs/wide_gate/sim_linear_tau1_k0.1` under way), the joint stage on a GPU, then real
  labels at pilot size.
- The architecture figure does not draw the PRS path yet.

## 2026-11-13 (night, 00:40 UTC) — The linear stage becomes one grid run with a prior variance per disease

Why: the first gate run of the linear model alone (CPU, `runs/wide_gate/sim_linear_tau1_k0.1`:
τ = 1, κ = 0.1, constant rate, no search) tracked GBLUP on the four polygenic simulated traits
but the 20-causal-SNP trait (heritability 0.3) stayed behind longest (−0.015 at 11
epoch-equivalents, −0.006 at 22), and GBLUP had each trait's own heritability. One τ for all 30
diseases would also tune the linear model less finely than the polygenic-score comparators,
which each get a per-disease choice. And 8 Optuna trials of the linear model meant 8 passes over
the genotypes per epoch, the expensive part at full size.

### Code changes

- `widedeepgp/model.py`: `prs_grid` (hp; `--prs_grid`, needs `--wide only`). The path then has
  K × n outputs (grid-major: column k·n + j), the loss's constants tiled K times, penalty
  r_j/(2Nτ_k) per column, and `optimizer_step` scales each column's AdamW update by τ_k/prs_var
  (the rule's learning rate at τ_k); `training_step` returns the sum over grid points (the
  logged loss stays their mean), so every column gets the gradient a model of its own would.
  A first version reparametrised θ = sφ instead: then the φ gradients were shrunk by s/K, close
  to AdamW's epsilon (1e-8) for small τ, and the test below caught 11% deviations. Validation
  scores every column; `val/auroc_mean` (the monitor) is the mean over diseases of each one's best
  column in that pass; buffers `best_*` keep, per disease, the weights of its best column and
  pass so far (strict improvement, float64 AUROC); `per_disease_state()` returns them as weights
  of a model without a grid.
- `widedeepgp/train.py`: `parse_grid`; `wide_hparams` adds `prs_grid` (refused unless
  `--wide only`); `write_per_disease` writes `<run>/per_disease/` (config.json, results.json,
  per_disease.csv, checkpoints/best.ckpt hard-linked as last.ckpt), from the last checkpoint's
  `best_*` buffers, and scores it on the tuning set again (`val_auroc_rescored`).
- `widedeepgp/tuning.py`: the linear stage is one grid run (`linear_run`, `linear_stage`,
  `linear_summary`; on the first picked GPU through a worker, or here), resumed from its last
  checkpoint after a crash; default grid 0.03, 0.1, 0.3, 1, 3, 10, 30; covariate learning rate
  1e-2; cosine to zero over the run's budget (30 epoch-equivalents, ≥128,000 participants); one
  validation per epoch-equivalent but at most one per 20,000 participants; patience 10; one
  snapshot. The joint stage starts from `linear/per_disease`. `--tune_trials_linear` removed (it
  existed only in 9e69019); `tune_optuna.py` loses the "linear" space.
- `main_multilabel.py`: `--wide_kappa 0.1` by default (the weights settle in about 10
  epoch-equivalents; 0.03 left them unsettled within a cosine over 30).
- Tests: `test_each_grid_point_trains_as_a_model_of_its_own` (6 AdamW steps through the model's
  own `optimizer_step`: every column equals a separate model with its τ to 1e-4);
  `test_a_run_tunes_itself` rewritten for the grid (choices match metrics.csv, rescoring matches,
  the grid's weights deleted and the per-disease model kept, score_snapshots scores it). CPU
  suite: 74 passed, 28 skipped (GPU).
- methods.md sections 4.10, 5.2, 6, 7, 8, 10; README.

## 2026-11-13 (00:50 UTC) — A moved run folder: the linear stage is read where it is

- `widedeepgp/tuning.py` `linear_summary`: the per-disease model and the checkpoints are located
  inside the stage's folder (`linear/per_disease/checkpoints/<name>`), not at the absolute paths
  recorded when the stage ran, so a run folder that was moved or copied still continues. The
  overnight GPU gates rely on this: the joint stage of the simulated-trait and real-label gates
  starts from a copy of the linear stage computed on the CPU (identical settings).
- `tests/test_ml_wide.py` `test_a_run_tunes_itself`: also continues a run whose `linear/` was
  moved. CPU: 17 passed, 3 skipped (GPU) in test_ml_wide.py.

## 2026-11-13 (01:30 UTC) — The schematic draws the linear PRS path; stores rebuilt

- `scripts/make_architecture_figure.py`: a "Linear PRS path" box (all SNPs and covariates in,
  30 logits added to the head's), the figure 117 mm tall instead of 97 (183 mm wide, 7 pt), the
  component table with the path (37,746,390 parameters: 1,258,200 × 30 SNP weights, 12 × 30
  covariate weights, 30 intercepts) and totals with and without it; the legend and the caption
  in methods.md say so (the "not yet drawn" sentence is gone). Redrawn:
  `figures/widedeepgp_architecture.{png,pdf}`, `_data.csv`, `_legend.md`.
- Data: both genotype stores were rebuilt without the withdrawn participants and swapped in at
  01:19 UTC (333,324 and 17,039 participants; every check passed; DeepWideGP-Scripts `widening`
  67aba7f). Store rows shifted, so the cached training-split SNP statistics are recomputed once
  per dataset (new cache files; the old ones are unused).
- Gate G1, fixed τ = 1 (CPU, 40 epoch-equivalents, finished 00:58): best mean tuning AUROC
  0.7166 against GBLUP 0.7132; the 20-SNP trait −0.003, the others +0.003 to +0.008. Training
  took 266 s of the 52 min wall time: the linear model is bound by reading genotypes.

## 2026-11-13 (03:55 UTC) — Gates on the proxy: the linear stage passes; the default grid reaches down to 0.003

- G1 with the grid stage (simulated traits; `runs/wide_gate/sim_linear_grid`, CPU, 01:19–02:03):
  per-disease mean tuning AUROC 0.7187 against GBLUP 0.7132 (sim20 −0.0055, the others +0.007 to
  +0.010; τ 3 for four traits, 1 for sim200); at one τ (3) at the end of training 0.7185, so
  selection adds next to nothing there; covariates alone 0.505 (they have no effect on these
  traits). Rescoring reproduces every choice exactly.
- G3, linear half (real labels of the pilot's proxy, 30 diseases; `runs/wide_gate/real_linear_grid`,
  CPU, 02:03–03:46): covariates alone 0.6607 (as in the pilot); the linear model at one τ (0.1)
  at the end of training 0.6616 (+0.0009, 24 of 30 diseases above their covariates-only AUROC);
  with τ and pass chosen per disease 0.6641 (+0.0034, optimistic: chosen on the tuning set). The
  pilot's network ended at 0.6593 (−0.0014) and the pilot's clumped PRS added +0.0014. Table:
  `real_linear_grid/linear/per_disease/gate_g3_linear_vs_covariates.csv` (aggregate). Best τ on
  real labels is small (0.03–0.3; mean-best 0.1), and 6 of 30 diseases chose the lowest grid
  value, 0.03.
- Code: `widedeepgp/tuning.py` `DEFAULT_PRS_GRID` now 0.003, 0.01, …, 30 (nine values), so that a
  disease whose SNPs carry nothing usable can fall back to (almost) its covariates; methods.md
  section 8 and README say so. A rerun of the real-label linear stage with this grid is queued
  (`real_linear_grid9`); the GPU gate on real labels starts from it. The GPU-test export
  (code_81ded30) differs from this commit only in that constant and documents.

## 2026-11-13 (04:00 UTC) — GPU tests of 81ded30: all pass but one tuner test, which timed out

- Ran 02:55–03:22 UTC on GPUs 1–4 (all five were free at 02:54), export
  `runs/gate_evidence/final_tests_2026-11-13/code_81ded30`: test_ml_wide.py 21 passed (with the
  three GPU tests of the PRS path and the slow CPU learning check), test_ml_model.py 9 passed,
  test_ml_train.py slow 2 passed (1 skipped), capacity probe 4 of 4 for both code versions;
  test_ml_train.py quick 19 passed, 1 failed: `test_the_tuner_runs_shared_trials_and_prunes`
  timed out (900 s). Its trials go through `main_multilabel.parse_args`, whose new
  `--min_participants 128000` floor turned 1.5 epoch-equivalents of 600 synthetic people into
  about 1,000 validation passes. Fixed in the test (`--min_participants 0
  --val_min_participants 0`, as the other small tests); the rerun of that test waits for free GPUs
  (`runs/gate_evidence/final_tests_2026-11-13/rerun_tuner_test_when_free.sh`), and the GPU gates
  wait for it.

## 2026-11-15 (20:30 UTC) — A resumed run scores the whole tuning set; methods brought up to date

What happened since 13 Nov 04:00: the tuner test's GPU rerun passed (13 Nov 08:44); the GPU gate
on the simulated traits ran (`runs/wide_gate/sim_joint`, 13 Nov 08:46–11:26, four GPUs): the
network on top added nothing (best of 12 trials 0.71869 against the linear model's 0.71873; the
linear model kept). The nine-value real-label linear stage (`real_linear_grid9`) was killed at
pass 28 of 30 by a session restart (it ran as a session-bound background job); resumed on 15 Nov
(detached), it finished, but its per-disease choices were all taken at the resumed pass, which
gave the chosen model 0.6573 on the whole tuning set (covariates alone 0.6607). The real-label
network gate started from it was stopped after four minutes.

### Code changes

- `widedeepgp/model.py` `on_validation_epoch_end`: Lightning restores the validation pass a
  checkpoint was written in part-way through, so a resumed run's repeated pass scored only its
  last batch (16 of 90 participants in a reproduction; one batch of the tuning set in the real
  run). A pass that scored fewer participants than the tuning set holds (`_tuning_set_size`) now
  scores the whole tuning set with `train.predict_split` before its metrics reach the best
  checkpoint, early stopping or the grid's choices; a repeated pass then reproduces the original.
  RunLog had already kept the partial pass out of metrics.csv, and the pilot's one resumed run
  (`pilot_sim/selected_long_s1`) is unaffected (its best checkpoint is a recorded pass).
- `tests/test_ml_wide.py` `test_a_resumed_run_scores_the_whole_tuning_set` (fails without the
  fix: one pass over 16 participants). CPU suite: 75 passed, 28 skipped (GPU).
- `methods.md`: section 2.5 the ladder shared with the PGI arm (Robert, 15 Nov: "the same in every
  way"), run from small to large, and its consequences; section 7 the resume problem and its fix;
  section 8 the gate results and the open question Robert raised (the linear stage overlaps the
  PGI arm; to be planned before any ladder run, not changed yet); section 9 the comparison
  protocol shared with the PGI arm and the published DeepGP as an extra comparator for I10 (Robert,
  15 Nov; planned, not run); section 12 the tests of the PRS path and the search, and the GPU
  results of 13 Nov.
- Gate documentation: `DeepWideGP-Scripts/scripts/widedeepgp-gates/` (methods, results, tables,
  the run scripts).

## 2026-11-15 (21:00 UTC) — The run's model is always the network on top (Robert's decision)

Robert: "I want the recorded result to be ridge + widedeepgp no matter which does better."

- `widedeepgp/tuning.py` `choose`: the run's model is the last stage's best trial; with the
  linear path on, the joint stage (the network on top of the per-disease linear model). The linear
  model alone is recorded in the run's `results.json` as `linear_only` (its folder, its mean tuning
  AUROC, and `network_minus_linear_only`), and keeps its weights (the joint stage starts from it).
  Before, the linear model was the run's model whenever the network did not beat it.
- Docstrings (`tuning.py`, `main_multilabel.py`), README, methods.md sections 8 and 10 say so;
  section 8's "Open" paragraph is now "Decided".
- `tests/test_ml_wide.py` `test_the_run_reports_the_network_on_top_whatever_its_value` (a joint
  trial at 0.715 is reported over a linear model at 0.72; the ablation is recorded; the right
  weights kept). CPU suite: 76 passed, 28 skipped (GPU).
- Explained to Robert: nothing in WideDeepGP uses folds; the ridge is fitted once per run on the
  whole training set, and the same fixed ridge weights (each disease's chosen prior variance and
  pass) plus the trained network predict for everyone else (tuning, test, held-out groups).

## 2026-11-15 (21:05 UTC) — The ladder is queued as one detached chain

- Robert: run the rungs smallest to largest without waiting for his go-ahead between them,
  ideally as one job. `runs/ladder/run_ladder_2026-11-15.sh` (copy and methods in
  `DeepWideGP-Scripts/scripts/widedeepgp-ladder/`), started detached at 20:56 UTC (setsid nohup;
  parent PID 1): it waits for the real-label gate (`runs/wide_gate/real_search`, running since
  20:44 UTC, detached) to finish cleanly, then runs `hm3plus_n13199`, `_n26287`, `_n53498`,
  `_n107743` and `hm3plus` one after another, each a whole search (`--wide on --tune_gpus
  auto`, width 32, one layer, batches 16 x 4, bf16), stopping only if a rung fails; rerunning it
  continues.
- methods.md section 2.5: the run order (one chained job, no pause for review).
- A project-wide `/mnt/sdg/robert/deepwidegp/CLAUDE.md` (outside git) now holds the reusable
  lessons for every session in the project, first among them that long jobs must be launched
  detached (setsid nohup, with a log and a completion line), never as session background tasks.

## 2026-11-15 (21:40 UTC) — Trial resume; no time limits; Robert's answers on the open questions

Robert's decisions: model size is varied later, several times, at the largest training size only
(each ladder varies one factor); drop the single-disease WideDeepGP comparators (E11, I25, I48,
M17, K80 and any like them): the single-disease comparator is the classic DeepGP, for I10 (the
highest N_eff of the 30) at the five sizes; the headline model is WideDeepGP; add trial resume
and remove every time limit; nothing is to start on 29 Nov.

### Code changes

- `widedeepgp/tuning.py` `fail_stale`: a trial a crash left running is marked failed and queued
  again (`enqueue_trial`) with the same settings; if its folder holds a last checkpoint, the
  queued trial carries `resume_from` and `objective` continues it in that folder (`cfg.resume`),
  otherwise it starts again afresh. Before, it was replaced by a new trial from scratch.
- `widedeepgp/callbacks.py` `OptunaPruning(steps_per_pass=...)` (passed by `run_training`):
  reports a pass's number in the run (optimiser steps over steps per pass), so a resumed trial
  is compared at the right rung of successive halving.
- Tests: `test_a_trial_a_crash_interrupted_carries_on_from_its_checkpoint` (run_training
  replaced by a stub: the same settings come back, the one with a checkpoint resumes in its own
  folder, the other starts afresh) and `test_the_pruner_sees_a_resumed_run_s_pass_numbers`.
  CPU suite: 78 passed, 28 skipped (GPU).
- methods.md: section 7 (continuing a search, no time limits; snapshots give test performance
  against compute after the unlock, FLOPs computed afterwards), section 9 (the classic DeepGP
  comparator as specified: PGI arm's I10 GWAS, p < 0.05, plink2 clumping at r² 0.1 within 250 kb,
  published model and training, 12-trial search with the published settings first, seed 1; model
  size varied later at the largest size).
- The ladder chain (`runs/ladder/run_ladder_2026-11-15.sh`, restarted 21:22 UTC) has no time
  limit, runs each rung from a frozen export of `widening` (`runs/ladder/code_<commit>`), and
  runs the classic DeepGP comparator after each rung when `runs/classic_deepgp/run_rung.sh`
  exists.

## 2026-11-15 (22:00 UTC) — The classic DeepGP comparator is being built and queued

- `--tune_first_given` (761585c): the network stage's first trial uses the given --lr, --l2 and
  --dropout (the classic DeepGP's published 7e-4, 1e-4, 0.2); test
  `test_a_search_can_try_the_given_settings_first`. CPU suite: 79 passed, 28 skipped.
- `DeepWideGP-Scripts/scripts/classic-deepgp-comparator/`: `build_datasets.py` (I10 lead SNPs
  with p <= 0.05 from the PGI arm's REGENIE GWAS of each training size, plink2 clumping at r² 0.1
  within 250 kb on the PGI arm's LD panel; one pass over the main store writes a small store per
  size, `data/genostore/classic_i10_n<size>`, and datasets `data/datasets/ard30/classic_i10_n<size>`;
  check of 200 participants on chromosomes 1, 6 and 22), started detached at 21:52 UTC
  (`run_build.sh`, log `data/classic_deepgp_work/build.log`); `run_rung.sh` (copy of
  `runs/classic_deepgp/run_rung.sh`, the ladder chain's hook): `main_multilabel.py --labels I10
  --wide off --sampler shuffled --gamma 0 --batch_size 64 --lr 7e-4 --l2 1e-4 --dropout 0.2
  --precision 32 --tune_trials 12 --tune_first_given`, seed 1, from the ladder's frozen export;
  methods.md.
- methods.md section 9: the clumping's LD comes from the PGI arm's panel (corrected).

## 2026-11-16 (11:10 UTC) — Workers take the next free trial

- Rung 1 of the ladder (13,199 participants, started 15 Nov 22:52 UTC from code_5b472c5): the
  ridge stage took 32 min (per-disease mean tuning AUROC 0.6634); the network stage's 12 trials
  were split as fixed shares of 3 per GPU worker, and one worker's first trial ran 27 passes in
  9.5 h, so three GPUs stood idle for about six hours while it finished its share. The gate on
  the real labels of the proxy (`runs/wide_gate/real_search`, finished 15 Nov 22:52 UTC):
  network on top 0.6640 against the ridge alone 0.6641 (reported model the network on top).
- `widedeepgp/tuning.py`: `claim_trial` and a new `worker` loop: each worker claims the next
  trial (`study.ask()`, waiting re-runs first) under an fcntl lock on `<stage>/.claim.lock`
  while the stage has fewer than its total finished or running, runs it and tells the result;
  `run_workers` passes the stage's total to every worker. Same trials and search, no idle GPUs.
- `tests/test_ml_wide.py` `test_free_workers_take_the_next_trial` (threads: a free worker takes
  the other seven trials while the slow first trial runs; four racing workers start exactly 5).
  CPU suite: 80 passed, 28 skipped. methods.md section 8 says so.
- The ladder chain picks this up from rung 2 (each rung exports the newest commit of `widening`).

## 2026-11-16 (20:05 UTC) — Three fixes, and what their review found (branch fix/2026-11-16, on d019bb9)

Robert asked for three fixes found while the code was being simplified; an independent review of
the first commit (98cbe9b) found more of the same kinds. They are kept apart from the
simplification and go onto it only after its old-against-new check.
- `--snp_embed vocab` (model.py): `_encode` passed the raw float dosages to the published
  201-entry `nn.Embedding`, which fails at the first forward pass (no run has used vocab).
  `vocab_codes` turns the model input back into the store's codes (dosage x 100, rounded, in
  float32: exact for all 201 levels); `_embed` is the embedding step, shared by `_encode` and the
  tests. The published `DataEmbeddingVocab.forward` truncates after rounding to two decimals and
  puts 6 levels (0.53, 0.59, 1.05, 1.06, 1.17, 1.18) one code low (methods.md 11). Vocab reads raw
  dosages, so `wide_hparams` refuses it with the linear PRS path (it needs `--wide off`), and
  config.json's `snp_scaling` says "none" when SNPs were not standardised.
- Participant IDs and values out of messages (genostore/datamodule.py, score_snapshots.py): the
  unknown-participant KeyError counts IDs instead of listing three; a cell that is not a number
  (e.g. an ID in a label or covariate column after a shifted row) raises without quoting it, with
  `from None` so pandas' chained message is not printed either; one-label checks count other
  values instead of listing them; `--save_scores` refuses a folder in the run directory or a git
  work tree, and writes its files 0600.
- Resume (callbacks.py, model.py, train.py): since 512c755 the model rescores a resumed run's
  repeated validation pass on the whole tuning set, but RunLog dropped those metrics, so
  run_training scored the tuning set again for results.json (the GPU test
  `test_a_resumed_run_reports_a_best_step_from_before_the_resume` failed with ['val', 'val'];
  it was written on 12 Nov, before 512c755). RunLog now keeps the repeated pass's metrics in
  `metrics_at` (no metrics.csv row, history entry or count), which saves 7-10 min per resume at
  full size. The fallback that scores an earlier best step from its checkpoint treated a grid run
  as if it had no grid (results.json's top-level fields were wrong for a resumed `--prs_grid` run
  whose best pass came before the resume; the search's choices read `per_disease` and were not
  affected): `model.val_metrics` is now the one computation for a validation pass and that
  fallback; `_grid_means` is pure and `_keep_grid_best` keeps the best_* buffers.
- Tests: `tests/test_ml_vocab.py`, `tests/test_genostore_identifiers.py` (new); in
  `tests/test_ml_train.py`, `test_a_resumed_run_scores_only_what_it_lacks` (CPU, `--wide only`:
  the repeated pass is the best / an earlier best / an earlier best with a grid), a CPU test of
  RunLog's repeated pass, and the GPU resume test allows an earlier best step. 11 of the new
  tests fail on d019bb9, as they should (the other 2 check behaviour that was already right).
  CPU suite: 93 passed, 29 skipped, 1 failed (`test_free_workers_take_the_next_trial`, a timing
  test that also fails 5 of 5 on d019bb9 at today's load). GPU (GPU 2): test_ml_vocab 6 passed,
  test_ml_train quick 24 passed, test_ml_wide 26 passed and the same timing test failed,
  test_ml_model 9 passed.
- Reported, not changed: `utils.py:36` prints every distinct label value and `main_genome.py`
  writes per-person predictions into the tracked `DeepGP/results` (the published single-disease
  code, kept unchanged; not run on real data). In DeepWideGP-Scripts (branch widening),
  `classic-deepgp-comparator/build_datasets.py:136` and `widedeepgp-pilot/gwas_annotation.py:107`
  look up rows by ID without a membership check, so a KeyError would print the ID.

## 2026-11-16 (21:05 UTC) — Leaner WideDeepGP code, same behaviour (the /simplify pass, integrated)

Robert asked for /simplify over all WideDeepGP code except the tests (project CLAUDE.md section 8:
lean code, thorough tests). Five groups of files were simplified on their own branches from
d019bb9 and reviewed adversarially: `simplify/g1_model` (model, blocks, losses, labels, metrics,
samplers), `g2_training` (train, tuning, callbacks), `g3_genostore_write` (convert, format,
from_pickles), `g4_genostore_read` (datamodule, prefetch, store, torch_decode) and
`g5_entry_scripts` (main_multilabel, tune_optuna, score_snapshots, the WideDeepGP part of
args_generator, scripts/ but t2d.sh). They are merged on `simplify/integrate`. The published
single-disease code (models/, layers/, utils.py, main_genome.py, scripts/t2d.sh) and the existing
tests are unchanged; the ladder's frozen copies are unaffected.

### Code changes

- Mostly docstrings and comments condensed to one line of why, with pointers to methods.md.
- Dead code removed (no caller in the repository, DeepWideGP-Scripts, PGI or the run scripts):
  `WideDeepGP.population_logits` and `predict_step`, `labels.summarise`, `WideLoss.extra_repr`,
  `SNPStoreDataModule.shuffle_targets`, the Prefetcher's write-only counters, `convert_chrom`'s
  unused `n_audit`, the samplers' `kind` attributes.
- Shared code: `tuning._spawn_worker` for both kinds of search worker; the run lock as a context
  manager; one stall/deadline check in convert.py; `model.WIDE_KEYS` (the linear path's weights)
  used by `load_weights` and `train.load_wide_init`; `genostore.prefetch.willneed` and `read_bps`
  (read-ahead requests, others' disk reads) used by the prefetcher, `disk_busy` and convert.py's
  read-ahead; `train.build_data` names the dataset with `datamodule.dataset_dir`;
  `tune_optuna.suggest` without its duplicated suggestions (same draws).
- train.py's and tuning.py's lists of what a run folder holds now point to README.md.
- New CPU tests that pin the behaviour of each group, run by name (the suite's
  `tests/test_ml_*.py tests/test_genostore*.py` globs do not match them):
  `python -m pytest tests/test_lean_*.py -q` (111 tests: g1 20, g2 16, g3 22, g4 28, g5 25).

### Verification (old d019bb9 against the integrated code)

- CPU suite: 1 failed, 79 passed, 28 skipped, as on d019bb9; the failure is the timing test
  `test_free_workers_take_the_next_trial` at its "free worker took all the rest" assertion,
  which fails on the old code too under today's load. The 111 new tests pass (each group also
  ran its own file on the old code).
- Equivalence harness (`runs/simplify_2026-11-16/harness`, synthetic data, GPU 4): 26 of 26
  checks pass against the golden outputs of d019bb9, 25 of them bit for bit and
  train_real_kernels (the fork's nondeterministic CUDA kernels) within its tolerances.
- PGI's tests that import genostore or widedeepgp.metrics, and DeepWideGP-Scripts' tests: the
  same results on old and new code (PGI: 1 failed, 198 passed, 1 skipped on both, the failure being
  `test_the_real_method_registry`, which reads PGI's locally modified methods.yaml; Scripts: 12, 5
  and 10 passed on both).
- Size of the 32 in-scope files: 5,683 to 5,071 lines (-612, -10.8%) and 4,090 to 3,883 code lines
  (-207, -5.1%); the new tests add 2,358 lines (1,939 code).

### Things worth knowing

- Pre-existing problems found and left as they were (this pass changed no behaviour):
  `--snp_embed vocab` fails at the first forward pass; a resumed run scores the tuning set twice
  (the GPU test `test_a_resumed_run_reports_a_best_step_from_before_the_resume` fails on old and
  new); `SNPStoreDataModule._read_table`'s error for unknown participants quotes up to three IDs
  (against CLAUDE.md section 3); `from_pickles` writes its sample tables under the caller's
  umask; `generate_synthetic_multilabel._auc` ranks ties by position. Branch `fix/2026-11-16`
  (another session) fixes some of these in the same files, so merging it with
  `simplify/integrate` will need conflicts resolved.

## 2026-11-16 (22:00 UTC) — The simplification and the fixes, verified, on widening (simplify/final)

- widening moves from d019bb9 to the branch simplify/final: the five group commits with their
  review fixes and the cross-group pass (entry above), then four commits of this session:
  tuning.py imports torch and the training stack inside the functions that use them again (the
  simplification had moved them to module level: importing tuning took 13 s and loaded torch,
  now 0.5 s as before); the simplification's new tests renamed so the suite command runs them
  (test_ml_model_pieces, test_ml_training_pieces, test_genostore_writing, test_genostore_reading,
  test_ml_entry_points); fix/2026-11-16 merged onto the simplified code (conflicts resolved onto
  its structure; two of the new tests pinned behaviour the fixes change on purpose and were
  updated); and vocab_codes refusing inputs that are not raw dosages with a ValueError (the
  harness's model check gives every configuration the same non-dosage inputs, and out-of-range
  codes ended in a CUDA device-side assert that broke the rest of the check).
- Size of the in-scope code (widedeepgp/, genostore/, main_multilabel.py, tune_optuna.py,
  score_snapshots.py, args_generator.py, scripts/ but t2d.sh; 32 files): 5,683 lines (4,090 code
  lines) at d019bb9; 5,071 (3,883) after the simplification; 5,138 (3,929) with the fixes.
  Tests: 2,489 lines, now 5,124.
- Verification of the final code (logs in /mnt/sdg/robert/deepwidegp/runs/simplify_2026-11-16):
  CPU suite 207 passed, 29 skipped, 1 failed (test_free_workers_take_the_next_trial, a timing
  test that fails the same way on d019bb9 under today's load). GPU (final_gpu/): test_ml_wide 26
  passed and the same timing failure, test_ml_train quick 24 passed (the resume test passes
  now), slow 2 passed and 1 skipped, test_ml_model 9, test_genostore_datamodule 4,
  test_ml_mamba3 6, test_ml_vocab 9, capacity probe 2 of 2. The old-against-new harness
  (harness/, 26 checks, golden outputs out_old_1 from d019bb9): 21 identical; the other 5 differ
  only where the fixes change behaviour on purpose (the one-label error counts other values:
  datamodule, datamodule_gpu; per-person score files are 0600: train_linear, train_network; the
  vocab model refuses non-dosage input: model). The simplification alone (5458e6c) matched
  d019bb9 on all 26 harness checks, the CPU and GPU suites, PGI's tests of genostore and
  widedeepgp.metrics, and DeepWideGP-Scripts' tests.
- The harness can check later refactors: harness.py --code DIR --out DIR --gpu N, then
  compare.py out_old_1 DIR (usage in harness.py's docstring).

## 2026-11-17 (03:00 UTC) — LD-block mode: the end-to-end block model, the default (plan revision 7, steps 1 to 5)

- Tag `pre-blocks` marks `widening` before this work (7dc184d). `main_multilabel.py` now defaults to
  `--input_mode blocks`, `--wide off` (the linear PRS path stays in the code as an option and is never the
  reported model), `--lr_schedule cosine --warmup_steps 200`, and `--final_pool mean` in blocks mode
  (`concat` maps to `atten` in snps mode as before); `--wide_kappa` is no longer set here (parser 0.03).
  New flags in `args_generator.py`: `--input_mode`, `--partition`, `--snp_code`, `--embed_dim`, `--table_var`,
  `--table_kappa`, `--init_encoder`, `--freeze_encoder`; `--mixer transformer`.
- `widedeepgp/blockencoder.py` (new): `BlockEncoder`, the first layer of blocks mode: a learned table with
  `embed_dim` numbers per SNP (dosage code: one vector times the standardised dosage; genotype code: three
  vectors per SNP with hat-function weights, raw dosages), pooled by LD block as one matmul per contiguous
  block (an exact segment sum), learned identity vectors per block and chromosome, a shared MLP
  (d_e -> 4 d_e -> d, GELU) and dropout; `load_init` for `--init_encoder` files; `table_learning_rate`.
- `widedeepgp/model.py`: the blocks branch (one sequence of block tokens plus the covariate token through
  `--e_layers` mixer blocks, LayerNorm, mean over tokens concatenated with the covariate vector into the
  head) **plus a linear readout of the table summed over all SNPs** (`readout`, Linear(embed_dim, n_labels),
  added to the logits). Why: in a probe on synthetic traits (4,000 participants, 132 SNPs, six traits) the
  model without it learned the three traits with shared causal SNPs only after a phase transition and never
  the two traits with causal SNPs of their own (tuning AUROC 0.48 and 0.39 against ceilings 0.78 and 0.97;
  with one SNP per block it learned nothing at all; a smaller table learning rate and shuffled batches did
  not help): a zero-initialised table takes its gradient direction from a random, fast-moving downstream
  network and its Adam steps never accumulate. With the readout every trait is learned after one pass
  (mean heritable AUROC 0.83 against a ceiling of 0.86; the SNP-mode model reached 0.75 on the same data).
  A rank-embed_dim ridge PRS is now an exact special case (head at zero), pinned by a test. The table has its
  own AdamW group (`lr_table` = table_kappa x table_var x sqrt(batch) / sqrt(M), no decay) and an L2 penalty
  sum(T^2) / (2 N table_var) in the loss (`train/table_penalty`); `--grad_ckpt` checkpoints per mixer layer;
  `snapshot()` writes weights-only snapshots with the table in bfloat16 (Snapshots no longer calls
  Lightning's save_checkpoint); `parameter_counts()` by module for config.json's `blocks` section;
  `big_params()` for RunLog (`train/lr_table`, `train/grad_norm_table`). `widedeepgp/blocks.py`:
  `TransformerBlock` (pre-norm nn.TransformerEncoderLayer, 4 d feed-forward, GELU) with the Block interface.
- `genostore/datamodule.py`: `load_partition` (a partition.npz of the design document's layout, checked
  against the store's chromosomes, variant counts and a sha256 of the variant IDs, or `uniform:<k>`; blocks
  absent from a region subset are dropped and the rest renumbered; non-contiguous or cross-sequence blocks
  refused), `partition_digest` (in every hp as `partition_digest`; refused on load, `--init_from`, resume);
  the genotype code reads raw dosages (`scale_snps`). `widedeepgp/train.py`: `block_hparams`, the `blocks`
  config section (partition, n_tokens, n_parameters_by_module, embedding and non-embedding counts,
  init_encoder_sha256), `--init_encoder` loading, `file_digest`. `tune_optuna.suggest` adds `table_var`
  (log-uniform 0.01 to 100) in blocks mode. `pretrain_blocks.py` (new): masked-SNP pretraining of the first
  layer (15% of SNPs hidden, genotype class predicted from the block's hidden layer through a per-SNP output
  vector) on a dataset's training rows, writing an `--init_encoder` file with a manifest.
- Scripts: `multilabel_synthetic.sh` passes `--input_mode snps --lr_schedule constant --warmup_steps 0`
  explicitly; `check_synthetic_run.py` picks a run's seed-band arm from its config (plain, annotated,
  blocks_transformer, blocks_bimamba1). The blocks arms' five seeds each (uniform:5 blocks, d_model 16, embed_dim 8,
  three epoch-equivalents; runs/blocks_bands) reach a mean heritable AUROC of 0.78 for both mixers against the
  published SNP-token arm's 0.70 on the same store; the null trait sits at 0.48; bands (lowest seed minus the
  larger of 0.04 and 2.5 SD): 0.74 on the mean, 0.68 to 0.78 per trait.
- Tests: `tests/test_ml_blocks.py` (18, CPU: the partition and its checks, pooling equals numpy segment sums
  at embed_dim 1/16/64, the genotype code, the ridge special case, names and counts, refusals, the table's
  learning rate and penalty, init files, a run with bf16 snapshots, scoring and a resume, a one-trial search
  and a seed retrain reproducing the chosen configuration, learning on the synthetic traits, pretraining)
  and `tests/test_ml_blocks_gpu.py` (7: both mixers in bf16, checkpointing changes nothing, the S5 models
  at 2,001 tokens and batch 128 fit: transformer d256x8 22.8 GiB, Mamba d256x14 25.2 GiB; the size table).
  Existing tests updated for the new defaults (test_ml_entry_points, test_ml_wide, test_ml_training_pieces,
  test_ml_model_pieces, test_ml_train). CPU suite: 227 passed, 39 skipped, 1 deselected (the timing test
  `test_free_workers_take_the_next_trial`, which fails under today's load as before). GPU: test_ml_model,
  test_ml_train (not slow), test_ml_vocab, test_genostore_datamodule: 46 passed.
- Mixer parameter counts (for the chains' size table): transformer S1-S5 (d16x1, d32x2, d64x4, d128x6,
  d256x8) 3.3k, 25k, 200k, 1.19M, 6.32M; Mamba matched by depth (1, 2, 6, 9, 14 layers) 3.9k, 22k, 209k,
  1.09M, 6.25M.
- Not implemented (recorded in the plan): the frozen cross-fitted first-layer deep ablation (a third input
  mode); E5 scoring on the held-out groups (another store; needs a scoring step that reuses the run's
  standardisation). Documentation of this change: README and methods.md (section 4.11), updated the same day.

## 2026-11-17 (15:00 UTC) — The linear readout removed at Robert's instruction; the first chain stopped and wiped

- Robert, after reading the explanation of the readout (the table summed over all SNPs mapped linearly to
  the 30 logits and added to the head's): "I never expected it to work on 4000 people. I only hope it will
  work on 200,000 people." The readout is removed from `widedeepgp/model.py` (no `readout` parameter; the
  logits are the head's alone; `parameter_counts()["head"]` is the head), the docstring of
  `widedeepgp/blockencoder.py` and the tests (`test_the_first_layer_contains_a_ridge_prs` pins the
  ridge at the pooled level; the learning check became a mechanics check: three recorded passes, a loss
  that does not rise, the null trait near 0.5, because without the readout the solo synthetic traits stay
  at chance at 3,000 participants). README, methods.md (section 4.11: "The head", the finding of 17 Nov kept
  as a dated note; the parameter table without the readout's 480 entries at d_e 16), the design document
  (section 12, item 1) and the plan record the decision and the probe that motivated the readout.
- The chain started at 04:25 UTC (runs/blocks, code export 84d49a9) was stopped at the 12th of 18 E0
  searches by killing its process session, and runs/blocks, runs/blocks_bands (the blocks arms' synthetic
  seed bands, measured with the readout; their rows are removed from scripts/multilabel_synthetic_seeds.csv)
  and the probes' scratch data were deleted. Kept, as inputs independent of the model: the simulated E0
  dataset hm3plus_n13199_sim and its linear comparators (blocks/), the bottom rung's linear comparators,
  the partitions. The masked-pretraining file under the simulated dataset was deleted (encoder-specific).
- Not relaunched: without the readout the E0 gate at 13,199 participants is expected to fail (the probe),
  so where the gate sits is Robert's decision before the chain runs again.
- Addendum (17:20 UTC): the blocks arms' synthetic seed bands re-measured without the readout (five seeds x two
  mixers, three epoch-equivalents on the 8,000-participant six-trait store; scripts/multilabel_synthetic_seeds.csv,
  commit 61e6af3): the three traits whose causal SNPs are shared by three diseases reach 0.77-0.79 for both
  mixers (the SNP-token arm: 0.70-0.74), the two traits with two causal SNPs of their own stay at chance
  (0.48-0.57), the null trait at 0.50. The bands therefore require 0.70-0.73 on the shared traits and nothing on
  the solo ones; check_synthetic_run.py picks the arm from the run's configuration.
