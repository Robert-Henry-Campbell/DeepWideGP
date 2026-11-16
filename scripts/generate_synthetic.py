"""Generate a synthetic single-disease dataset for testing DeepGP, in the pickle format of
SNPPCACHRDataModule (README.md, "Pickle file format"): pukb/genes/<data_name>/<rd_mode>/*.pkl.
"""

import os
import pickle
import numpy as np
import argparse


def generate_dataset(data_name, n_participants, n_snps, n_cases, n_causal,
                     rd_mode="ld_all", base_dir="pukb/genes", seed=42):
    np.random.seed(seed)
    n_chr = 22
    snps_per_chr = [n_snps // n_chr + (1 if c < n_snps % n_chr else 0) for c in range(n_chr)]
    first = [sum(snps_per_chr[:c]) for c in range(n_chr)]
    genes_chr = [np.array([f"rs_syn_{i}" for i in range(s, s + k)]) for s, k in zip(first, snps_per_chr)]
    pos_chr = [np.arange(k) for k in snps_per_chr]

    # dosages 0/1/2 drawn with a minor allele frequency from U(0.05, 0.5), chromosome by chromosome
    all_snp_data = []
    for k in snps_per_chr:
        maf = np.random.uniform(0.05, 0.5, size=k)
        geno = np.column_stack([np.random.binomial(2, m, size=n_participants) for m in maf])
        all_snp_data.append(geno.astype(np.float64))

    # the liability: causal SNPs on chromosome 1 plus noise; the n_cases highest are the cases
    causal_indices = np.random.choice(snps_per_chr[0], size=n_causal, replace=False)
    betas = np.random.normal(0.5, 0.1, size=n_causal)
    liability = all_snp_data[0][:, causal_indices] @ betas
    liability += np.random.normal(0, 0.5, size=n_participants)
    threshold = np.sort(liability)[::-1][n_cases - 1]
    labels = (liability >= threshold).astype(int)
    if labels.sum() != n_cases:                      # ties at the threshold
        labels = np.zeros(n_participants, dtype=int)
        labels[np.argsort(liability)[::-1][:n_cases]] = 1

    n_test = max(n_participants // 5, 2)
    n_train = n_participants - n_test
    indices = np.random.permutation(n_participants)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    covariates = np.random.randn(n_participants, 8)  # age, sex and six PCs
    covariates[:, 1] = np.random.binomial(1, 0.5, size=n_participants)
    covariates[:, 0] = np.random.normal(55, 10, size=n_participants)

    out_dir = os.path.join(base_dir, data_name, rd_mode)
    os.makedirs(out_dir, exist_ok=True)
    for fname, obj in [
        ("genes.pkl", genes_chr),
        ("pos.pkl", pos_chr),
        ("snp_train.pkl", [g[train_idx] for g in all_snp_data]),
        ("snp_test.pkl", [g[test_idx] for g in all_snp_data]),
        ("label_train.pkl", [labels[train_idx] for _ in range(n_chr)]),   # 22 copies: the loader checks they agree
        ("label_test.pkl", [labels[test_idx] for _ in range(n_chr)]),
        ("covar_train.pkl", covariates[train_idx]),
        ("covar_test.pkl", covariates[test_idx]),
    ]:
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(obj, f)

    print(f"Dataset '{data_name}' created in {out_dir}")
    print(f"  {n_participants} participants, {n_snps} SNPs across {n_chr} chromosomes")
    print(f"  {n_cases} cases / {n_participants - n_cases} controls")
    print(f"  {n_causal} causal SNP(s) on chromosome 1 at indices {causal_indices}")
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"  SNPs per chromosome: {snps_per_chr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GWAS data for DeepGP")
    parser.add_argument("--name", required=True, help="Dataset name (used as directory name)")
    parser.add_argument("--n_participants", type=int, default=100)
    parser.add_argument("--n_snps", type=int, default=100)
    parser.add_argument("--n_cases", type=int, default=10)
    parser.add_argument("--n_causal", type=int, default=1)
    parser.add_argument("--rd_mode", default="ld_all")
    parser.add_argument("--base_dir", default="pukb/genes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.name, args.n_participants, args.n_snps, args.n_cases, args.n_causal, args.rd_mode,
                     args.base_dir, args.seed)
