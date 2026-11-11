# -*- coding: utf-8 -*-
"""Store-backed replacement for utils.SNPPCACHRDataModule.

Same interface and same batches -- ([x_seq_1, ..., x_seq_22, cov], y) with x float32
[B, L_i, 1] -- but genotypes stay on disk in the compact store and are decoded on the GPU
per batch, instead of being unpickled into RAM as float32 (which needs ~10 TB for the full
UKB imputed set).

Dataset directory (same place the legacy pickles live, so --label/--rd_mode still select it):
    <data_dir>/<label>/<rd_mode>/store.json          {"store": "<path>", "pos_mode": "index"|"pos",
                                                       "regions": optional, see below}
    <data_dir>/<label>/<rd_mode>/samples_train.tsv   IID <TAB> label <TAB> covariate columns...
    <data_dir>/<label>/<rd_mode>/samples_test.tsv
    <data_dir>/<label>/<rd_mode>/samples_<ve>.tsv    optional external cohorts (--ve)

Many diseases at once (the widened model): store.json names the columns to use,
    {"store": ..., "labels": ["I10", "E11", ...], "covariates": ["birth_year", "sex", "pc1", ...]}
Then y is [B, n_labels] (0, 1, or NaN where a label does not apply), every column that is not
named is ignored, and samples_val.tsv, when present, is the validation set (otherwise 15% of
the training table is split off at random). Without a "labels" key nothing changes: one "label"
column, every other column a covariate, stratified split.

"regions" defines the input sequences. Default: one sequence per chromosome in the store.
Otherwise a list with one entry per sequence, each a list of [chrom, start, stop] segments
(variant indices, stop exclusive) that are concatenated -- this is how sub-chromosome blocks,
or sequences spanning several chromosomes, are expressed.
"""
import json
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from . import torch_decode as td
from .store import GenoStore


def dataset_dir(configs):
    return f"{configs.data_dir}/{configs.label}/{configs.rd_mode}"


def is_store_dataset(configs):
    return os.path.exists(os.path.join(dataset_dir(configs), "store.json"))


def default_regions(store):
    return [[[c, 0, store.n_variants(c)]] for c in store.chroms]


class LazyVariantIDs:
    """len() is free; the IDs themselves are only read from disk if somebody indexes this."""

    def __init__(self, store_dir, segments):
        self._dir, self._segs, self._ids = store_dir, segments, None
        self._len = sum(v1 - v0 for _, v0, v1 in segments)

    def __len__(self):
        return self._len

    def _load(self):
        if self._ids is None:
            gs = GenoStore(self._dir)
            self._ids = np.concatenate([gs.variants(c)["id"][v0:v1] for c, v0, v1 in self._segs])
        return self._ids

    def __getitem__(self, i):
        return self._load()[i]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._load(), dtype=dtype)


class StoreBatchDataset(Dataset):
    """Indexing with a *list of sample indices* returns that whole batch, still encoded."""

    def __init__(self, store_dir, regions, rows, cov, y):
        self.store_dir, self.regions = store_dir, regions
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cov = None if cov is None else np.ascontiguousarray(cov, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.y = y[:, None] if y.ndim < 2 else y
        self._gs = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, batch_idx):
        if self._gs is None:                       # one reader per (worker) process
            self._gs = GenoStore(self.store_dir)
        target = None
        if isinstance(batch_idx, tuple):           # a sampler that balances on one label says which
            target, batch_idx = batch_idx
        bi = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        rows = self.rows[bi]
        geno = []
        for seq in self.regions:
            segs = []
            for c, v0, v1 in seq:
                r = self._gs.read_raw(c, rows, v0, v1)
                segs.append([torch.from_numpy(r.full) if r.full is not None else torch.empty(0, dtype=torch.uint8),
                             torch.from_numpy(np.ascontiguousarray(r.full_ovr)),
                             torch.from_numpy(r.tail) if r.tail is not None else torch.empty(0, dtype=torch.uint8),
                             torch.from_numpy(np.ascontiguousarray(r.tail_ovr))])
            geno.append(segs)
        out = {"geno": geno, "y": torch.from_numpy(self.y[bi])}
        if self.cov is not None:
            out["cov"] = torch.from_numpy(self.cov[bi])
        if target is not None:
            out["target"] = torch.tensor(int(target))
        return out


class SNPStoreDataModule(pl.LightningDataModule):
    def __init__(self, configs, store_dir=None, regions=None):
        super().__init__()
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.data_mode = configs.dm
        self.cont = configs.cont_pheno
        self.scaling = configs.scaling
        self.snp_embed = configs.snp_embed
        self.ve = configs.ve
        self.shuffle = configs.shuffle
        self.snp_stats = getattr(configs, "snp_stats", "store")
        if getattr(configs, "non_add", False) or len(getattr(configs, "use_sim", "")) > 0:
            raise NotImplementedError("--non_add / --use_sim are only implemented for the legacy pickle loader")
        self.dir = dataset_dir(configs)
        spec = {}
        if os.path.exists(os.path.join(self.dir, "store.json")):
            with open(os.path.join(self.dir, "store.json")) as f:
                spec = json.load(f)
        store_dir = store_dir or spec["store"]
        if not os.path.isabs(store_dir):
            store_dir = os.path.join(self.dir, store_dir)
        self.store_dir = store_dir
        self.store = GenoStore(store_dir)
        regions = regions or getattr(configs, "geno_regions", None) or spec.get("regions") or default_regions(self.store)
        self.regions = [[[str(c), int(v0), int(v1)] for c, v0, v1 in seq] for seq in regions]
        self.pos_mode = getattr(configs, "pos_mode", None) or spec.get("pos_mode", "index")
        # Named columns (multi-disease datasets). ``configs.labels`` may pick a subset, e.g. one
        # disease for a single-disease model trained on the same dataset.
        self.label_names = list(spec["labels"]) if spec.get("labels") else None
        self.cov_names = list(spec["covariates"]) if spec.get("covariates") is not None else None
        self.multilabel = self.label_names is not None
        pick = getattr(configs, "labels", None)
        if pick:
            pick = [p for p in (pick.split(",") if isinstance(pick, str) else list(pick)) if p]
            if not self.multilabel:
                raise ValueError("--labels needs a dataset whose store.json names its label columns")
            unknown = [p for p in pick if p not in self.label_names]
            if unknown:
                raise KeyError(f"--labels: {unknown} are not labels of this dataset ({self.label_names})")
            self.label_names = pick
        if self.multilabel:
            if self.cont:
                raise ValueError("continuous phenotypes are not supported with named label columns")
            if self.cov_names is None:
                raise ValueError('store.json names "labels" but not "covariates"; name both (covariates may be [])')
            both = sorted(set(self.label_names) & set(self.cov_names))
            if both:
                raise ValueError(f"columns named as both label and covariate: {both}")
        self.train_sampler = None                 # set by the training code to balance batches

        # what main_genome.py reads off the data module
        self.genes_chr = [LazyVariantIDs(store_dir, seq) for seq in self.regions]
        self.genes = range(sum(len(g) for g in self.genes_chr))
        self.pos_chr = [self._positions(seq) for seq in self.regions]
        self.seq_len = [len(g) for g in self.genes_chr]

        row_of = {s: i for i, s in enumerate(self.store.samples)}
        tr = self._read_table("train", row_of)
        te = self._read_table("test", row_of)
        if not self.cont and not self.multilabel:
            for name, t in (("train", tr), ("test", te)):
                vals = np.unique(t["y"])
                if not np.all(np.isin(vals, [0, 1])):
                    raise ValueError(f"{name} labels must be encoded as 0/1, found {vals.tolist()}")
        if self.multilabel and os.path.exists(os.path.join(self.dir, "samples_val.tsv")):
            self.splits = {"train": tr, "val": self._read_table("val", row_of), "test": te}
        else:
            # one label: stratified, as the pickle loader does; a label matrix cannot be stratified on
            strat = None if (self.cont or self.multilabel) else tr["y"]
            train_idx, val_idx = train_test_split(np.arange(len(tr["y"])), test_size=0.15, shuffle=True,
                                                  stratify=strat, random_state=333)
            self.splits = {"train": {k: v[train_idx] for k, v in tr.items()},
                           "val": {k: v[val_idx] for k, v in tr.items()},
                           "test": te}
        self.n_cov = int(self.splits["train"]["cov"].shape[1])
        self.n_labels = len(self.label_names) if self.multilabel else 1
        self._row_of = row_of
        self._prepare_scaling()
        self._dev = {}

    # ---- construction helpers ------------------------------------------------------------
    def _positions(self, seq):
        if self.pos_mode == "index":
            return np.arange(sum(v1 - v0 for _, v0, v1 in seq))
        if self.pos_mode == "pos":
            return np.concatenate([self.store.positions(c)[v0:v1] for c, v0, v1 in seq])
        raise ValueError("pos_mode must be 'index' or 'pos'")

    def _read_table(self, split, row_of):
        df = pd.read_csv(os.path.join(self.dir, f"samples_{split}.tsv"), sep="\t", dtype={"IID": str})
        missing = [s for s in df["IID"] if s not in row_of]
        if missing:
            raise KeyError(f"samples_{split}.tsv: {len(missing)} IDs are not in the store, e.g. {missing[:3]}")
        rows = np.array([row_of[s] for s in df["IID"]], dtype=np.int64)
        if not self.multilabel:
            covs = [c for c in df.columns if c not in ("IID", "label")]
            return {"rows": rows, "y": df["label"].to_numpy(), "cov": df[covs].to_numpy(dtype=np.float64)}
        absent = [c for c in self.label_names + self.cov_names if c not in df.columns]
        if absent:
            raise KeyError(f"samples_{split}.tsv lacks the columns {absent} named in store.json")
        y = df[self.label_names].to_numpy(dtype=np.float32)          # empty cell -> NaN -> not applicable
        bad = ~(np.isnan(y) | (y == 0) | (y == 1))
        if bad.any():
            col = self.label_names[int(np.argwhere(bad)[0, 1])]
            raise ValueError(f"samples_{split}.tsv: labels must be 0, 1 or empty; column {col} has other values")
        cov = df[self.cov_names].to_numpy(dtype=np.float64)
        if np.isnan(cov).any():
            raise ValueError(f"samples_{split}.tsv: covariates {self.cov_names} contain missing values")
        return {"rows": rows, "y": y, "cov": cov}

    def _prepare_scaling(self):
        self.snp_mean = self.snp_inv_std = None
        tr = self.splits["train"]
        if self.scaling:
            if self.snp_embed == "cov":
                if self.snp_stats == "train":
                    mean, std = self._train_snp_stats(tr["rows"])
                else:                       # statistics stored with the genotypes (all store samples)
                    mean, std = [], []
                    for seq in self.regions:
                        ms = [self.store.stats(c) for c, _, _ in seq]
                        mean.append(np.concatenate([m[0][v0:v1] for m, (_, v0, v1) in zip(ms, seq)]))
                        std.append(np.concatenate([m[1][v0:v1] for m, (_, v0, v1) in zip(ms, seq)]))
                self.snp_mean = [np.asarray(m, dtype=np.float32) for m in mean]
                # a SNP with no variation carries no information: map it to 0 rather than inf/NaN
                self.snp_inv_std = [np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0).astype(np.float32) for s in std]
            if tr["cov"].shape[1]:
                cm, cs = tr["cov"].mean(axis=0), tr["cov"].std(axis=0)
                for sp in self.splits.values():
                    sp["cov"] = (sp["cov"] - cm) / cs
                self.cov_cm, self.cov_cs = [cm], [cs]
            if self.cont:
                lm, ls = tr["y"].mean(axis=0), tr["y"].std(axis=0)
                for sp in self.splits.values():
                    sp["y"] = (sp["y"] - lm) / ls
                self.label_mean_std = (lm, ls)

    def _train_snp_stats(self, rows):
        """Training-split SNP statistics, computed once per (training rows, sequences) and kept in
        the dataset directory: a pass over the training genotypes takes minutes at biobank scale."""
        import hashlib
        h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
        h.update(json.dumps(self.regions).encode())
        path = os.path.join(self.dir, f"snp_stats_train_{h.hexdigest()[:16]}.npz")
        if os.path.exists(path):
            z = np.load(path)
            k = len(self.regions)
            return [z[f"mean{i}"] for i in range(k)], [z[f"std{i}"] for i in range(k)]
        mean, std = self.compute_snp_stats(rows)
        try:
            tmp = path + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, **{f"mean{i}": m for i, m in enumerate(mean)}, **{f"std{i}": s_ for i, s_ in enumerate(std)})
            os.replace(tmp, path)
        except OSError:                           # read-only dataset directory: just recompute next time
            pass
        return mean, std

    def compute_snp_stats(self, rows, chunk=256, device=None):
        """Per-SNP mean/std of the dosage over ``rows`` (what the legacy loader computes on train)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ds = StoreBatchDataset(self.store_dir, self.regions, rows, None, np.zeros(len(rows)))
        s0 = [torch.zeros(n, dtype=torch.float64, device=device) for n in self.seq_len]
        s1 = [torch.zeros(n, dtype=torch.float64, device=device) for n in self.seq_len]
        cnt = [torch.zeros(n, dtype=torch.float64, device=device) for n in self.seq_len]
        for i in range(0, len(rows), chunk):
            geno = ds[np.arange(i, min(len(rows), i + chunk))]["geno"]
            for k, q in enumerate(self.decode_q(geno, device)):
                ok = q != td.MISSING
                x = torch.where(ok, q, 0).to(torch.float64) * 0.01
                s0[k] += x.sum(0); s1[k] += (x * x).sum(0); cnt[k] += ok.sum(0)
        mean = [(a / c.clamp(min=1)).cpu().numpy() for a, c in zip(s0, cnt)]
        std = [np.sqrt(np.maximum((b / c.clamp(min=1)).cpu().numpy() - m * m, 0.0)) for b, c, m in zip(s1, cnt, mean)]
        return mean, std

    # ---- decoding (runs on the device the batch was moved to) ------------------------------
    def decode_q(self, geno, device=None):
        """Encoded batch -> list of uint8 q tensors [B, L_i], one per sequence."""
        w, out = self.store.w, []
        for seq, segs in zip(self.regions, geno):
            parts = []
            for (c, v0, v1), (full, full_ovr, tail, tail_ovr) in zip(seq, segs):
                if device is not None:
                    full, full_ovr, tail, tail_ovr = (t.to(device) for t in (full, full_ovr, tail, tail_ovr))
                m = self.store.n_variants(c)
                parts.append(td.decode_segment(full, full_ovr, tail, tail_ovr, w, m % w, v0 % w, v1 - v0))
            out.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=1))
        return out

    def decode_batch(self, batch):
        """Encoded batch (already on the target device) -> (x_all, y) exactly as the model expects."""
        y = batch["y"]
        dev = y.device
        if self.snp_mean is not None and dev not in self._dev:
            self._dev[dev] = ([torch.from_numpy(m).to(dev) for m in self.snp_mean],
                              [torch.from_numpy(s).to(dev) for s in self.snp_inv_std])
        mean, inv = self._dev.get(dev, (None, None))
        x_all = []
        for k, q in enumerate(self.decode_q(batch["geno"])):
            x_all.append(td.to_model_input(q, None if mean is None else mean[k], None if inv is None else inv[k]))
        if self.data_mode == "snps_covs":
            x_all.append(batch["cov"])
        if "target" in batch:
            return x_all, y, batch["target"]
        return x_all, y

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return self.decode_batch(batch) if isinstance(batch, dict) and "geno" in batch else batch

    # ---- Lightning plumbing --------------------------------------------------------------
    def shuffle_targets(self):
        if self.shuffle:
            self.splits["train"]["y"] = np.random.permutation(self.splits["train"]["y"])

    def _dataset(self, sp):
        cov = sp["cov"] if self.data_mode == "snps_covs" else None
        return StoreBatchDataset(self.store_dir, self.regions, sp["rows"], cov, sp["y"])

    def _loader(self, ds, shuffle):
        sampler = BatchSampler(RandomSampler(ds) if shuffle else SequentialSampler(ds), self.batch_size, drop_last=False)
        return DataLoader(ds, sampler=sampler, batch_size=None, num_workers=self.num_workers)

    def setup(self, stage: str):
        if stage == "fit":
            self.snp_train, self.snp_val = self._dataset(self.splits["train"]), self._dataset(self.splits["val"])
        if stage == "test":
            self.snp_test = self._dataset(self.splits["test"])
        if stage == "predict":
            self.snp_predict = self._dataset(self.splits["test"])

    def train_dataloader(self):
        if self.train_sampler is not None:        # yields whole batches (optionally with a target label)
            return DataLoader(self.snp_train, sampler=self.train_sampler, batch_size=None,
                              num_workers=self.num_workers)
        return self._loader(self.snp_train, True)

    def val_dataloader(self):
        return self._loader(self.snp_val, False)

    def test_dataloader(self):
        if self.ve is not None:
            print(f"Validating with external cohort: {self.ve}")
            return self.eval_dataloder(self.ve)
        return self._loader(self.snp_test, False)

    def predict_dataloader(self):
        if self.ve is not None:
            print(f"Validating with external cohort: {self.ve}")
            return self.eval_dataloder(self.ve)
        return self._loader(self.snp_predict, False)

    def eval_dataloder(self, pheno):
        sp = self._read_table(pheno, self._row_of)
        if not self.cont and not self.multilabel:
            vals = np.unique(sp["y"])
            if not np.all(np.isin(vals, [0, 1])):
                raise ValueError(f"{pheno} labels must be encoded as 0/1, found {vals.tolist()}")
        if self.scaling:
            if sp["cov"].shape[1]:
                sp["cov"] = (sp["cov"] - self.cov_cm[0]) / self.cov_cs[0]
            if self.cont:
                sp["y"] = (sp["y"] - self.label_mean_std[0]) / self.label_mean_std[1]
        return self._loader(self._dataset(sp), False)
