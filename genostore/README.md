# genostore — disk-backed genotypes for DeepGP

Added 2026-11-10. Replaces "unpickle every genotype into RAM as float32" for large datasets.
The original pickle loader (`utils.SNPPCACHRDataModule`) is untouched and still used for any
dataset directory that does not contain a `store.json`.

## Why

`SNPCHRDataset` holds every genotype as a float32 tensor. For the UKB imputed set that is
333,444 × 600,143 × 4 B ≈ 745 GiB for chromosome 1 alone (the server has 754 GiB) and ≈ 10 TB
for the genome. The store keeps genotypes on disk at ~0.28 byte each, reads only the rows of
the current mini-batch, and decodes them on the GPU.

| | pickle loader | genostore |
|---|---|---|
| chr1, 333,444 people | 745 GiB of RAM | ~55 GB on disk, a few MB of RAM per batch |
| genome (7.81 M SNPs) | ~10 TB of RAM | ~720 GB on disk |
| HapMap3 (1.05 M SNPs) | ~1.3 TB of RAM | ~100 GB on disk |

(Disk sizes use the 0.277 byte/genotype measured on 11 Nov 2026 on the real imputed genotypes
of 512 participants, all 22 chromosomes: 0.274–0.281 by chromosome, with 2.4–3.1% non-integer
dosages. The size scales with that share. The HapMap3 figure assumes the genome-wide average.)

## Encoding ("base2+ovr8")

A genotype is the ALT-allele dosage, stored as `q = round(dosage × 100)` (0–200, 255 = missing)
— the resolution DeepGP's `vocab` embedding already uses. About 97% of imputed genotypes are
exactly 0, 1 or 2, so each genotype gets a 2-bit code (0/1/2, or 3 = "exception"), and each
exception's true byte is appended to that sample's exception run in order. Lossless with
respect to `q`. Variants are grouped in blocks of 16,384; files are block-major so that a
variant-major `.pgen` can be converted in one streaming pass. Details: `format.py` docstring.

## Files

| File | Role | Needs |
|---|---|---|
| `format.py` | encoding, block writer (resumable), store creation | numpy |
| `store.py` | `GenoStore` reader (memory-mapped, row gathers) | numpy |
| `torch_decode.py` | decode + standardise on the GPU | torch |
| `datamodule.py` | `SNPStoreDataModule`, drop-in for the pickle data module | torch, lightning |
| `convert.py` | `.pgen` → store | numpy, **pgenlib** |
| `from_pickles.py` | legacy pickle dataset → store dataset | numpy |
| `prefetch.py` | polite large-request reader for busy rotational disks | stdlib |

`pgenlib` is only needed for conversion. It is not installed in the `widedeepgp` venv; the
conversions on 2026-11-10 were run with `/mnt/sdg/robert/miniforge3/envs/gsem-py/bin/python`.

## Use

Convert (one-off; resumable — rerun the same command after an interruption):

```bash
# every sample, one or more chromosomes
python -m genostore.convert --pgen-dir /mnt/sdh/upamanyu/UKB/Genomics_PGEN \
    --pattern "UKB_chr{chrom}" --chroms 1 --out /path/to/store

# a random subset of people (e.g. for tests); --anonymize-ids drops the participant IDs
python -m genostore.convert ... --chroms $(seq 1 22) --keep-n 4096 --seed 0 --out /path/to/store

python -m genostore.convert verify --out /path/to/store     # audit rows vs. what the reader returns
```

For a SNP subset such as HapMap3+, pass `--extract FILE`: a text file whose header names
`CHROM` and `ID` (or `CHROM` and `POS`). Only the listed variants' records are read from the
`.pgen`, so no intermediate copy is needed:

```bash
python -m genostore.convert --pgen-dir /path/Genomics_PGEN --pattern "UKB_chr{chrom}" \
    --chroms $(seq 1 22) --extract snps.tsv --out /path/to/store
```

Source-disk etiquette (`--disk-policy`, see `prefetch.py`): the default `idle` only reads the
source disk while nobody else is reading it, and otherwise consumes pages another job has
already pulled into the page cache; `--opportunistic` picks whichever chromosome is readable
that way. `share` reads alongside other jobs at `--duty`. On an SSD or a cached file pass
`--no-prefetch`.

Describe a dataset (same place the pickles used to live, so `--label` / `--rd_mode` select it):

```
pukb/genes/<label>/<rd_mode>/store.json           {"store": "/path/to/store", "pos_mode": "index"}
pukb/genes/<label>/<rd_mode>/samples_train.tsv    IID <TAB> label <TAB> covariates...
pukb/genes/<label>/<rd_mode>/samples_test.tsv
```

then train exactly as before: `python main_genome.py --label <label> --rd_mode <rd_mode> ...`.
`main_genome.py` picks the store-backed module when `store.json` exists.

Optional keys in `store.json`: `"regions"` — a list with one entry per input sequence, each a
list of `[chrom, start, stop]` segments (variant indices) that are concatenated. Default is
one sequence per chromosome. This is how sub-chromosome blocks would be expressed (the model
itself still assumes 22 sequences). `"pos_mode"`: `"index"` (0..L-1, as the synthetic data
uses) or `"pos"` (the store's position column, bp for UKB).

Multi-disease datasets (`main_multilabel.py`, `methods.md`) add to `store.json`:

```
"labels":      ["I10", "E11", ...]     label columns of the sample tables; values 0, 1 or empty
"covariates":  ["birth_year", "sex", ...]   covariate columns; every other column is ignored
"annotations": {"file": "annotations.npz", "names": [...]}   optional per-SNP annotation vectors:
                                        one [SNPs, A] array per chromosome under the key chr<chrom>
```

and a third table, `samples_val.tsv`, which is used as the validation set as given (without it
15% of the training table is split off). Naming the columns means a disease can never be read
as a covariate. A dataset without `"labels"` behaves as before.

An existing pickle dataset can be converted for comparison:
`python -m genostore.from_pickles --src pukb/genes/1causal_10case/ld_all --dst pukb/genes/1causal_10case/store`.

## Behaviour differences from the pickle loader (deliberate)

- **SNP scaling statistics.** Default `--snp_stats store` standardises with the mean/SD stored
  at conversion time (all samples in the store). `--snp_stats train` recomputes them on the
  training split, which is what the pickle loader does. The first avoids a pass over the
  training genotypes at start-up; it uses test-set genotypes (never labels) in the scaling.
- **Monomorphic SNPs** (SD = 0) become 0 instead of inf/NaN.
- **Missing genotypes** become the SNP mean (0 after scaling).
- `--non_add` and `--use_sim` are not implemented for store datasets.
- Batches arrive encoded and are decoded in `on_after_batch_transfer`; `DataLoader` workers
  only read bytes. Shuffling draws the same permutation as before for a given seed.

## Tests

```bash
python -m pytest tests/test_genostore.py tests/test_genostore_datamodule.py -q   # widedeepgp env
/mnt/sdg/robert/miniforge3/envs/gsem-py/bin/python tests/test_genostore_pgen.py  # needs pgenlib
```

`test_genostore_datamodule.py` converts the synthetic `1causal_10case` pickles and checks that
the store-backed module returns the same people, order, covariates and (to 2e-5) genotype
tensors as the pickle loader, on CPU and GPU.

Training-level check (11 Nov 2026, `scripts/genostore_equivalence_check.sh` and the Scripts
repo's `loader_step_equivalence.py`): from one seed, losses under the two loaders agree to 1e-7
for the first 20 steps and then drift apart by as much as two pickle-loader runs drift from each
other; full runs gave test AUC 0.767–0.771 through the store (3 runs) against 0.768–0.771 through
the pickle loader (7 runs).

## Not done yet

- A full-cohort conversion of all imputed SNPs has not been run. The HapMap3+ subset has (11 Nov
  2026, `--extract`): 1,258,200 SNPs for 333,444 participants in about 2 h 40 min wall-clock
  under the idle-only disk policy (about 450-950 SNPs/s with 24 threads while reading), 116 GB
  (0.277 byte/genotype), 0 mismatches on the audit rows of all 22 chromosomes.
- Reads go through a memory map. The first pass over a store that is not in the page cache is
  slow (about 150 MB/s by page faults, which made the one-off training-split SNP statistics of
  215,872 participants take ten minutes); reading the files once with a few parallel
  `dd`/`cat` processes warms the cache at disk speed (2.3 GB/s here, 50 s for 116 GB).
- The layout is block-major (one 4 KB read per sample per 16,384 SNPs). If training I/O ever
  becomes the bottleneck, a row-major repack is the obvious optimisation.
- GPU-capacity results that motivated this: `DeepWideGP-Scripts/scripts/deepgp_gpu_capacity/`.
