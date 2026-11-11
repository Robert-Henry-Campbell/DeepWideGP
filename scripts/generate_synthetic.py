"""
Generate synthetic GWAS datasets for testing DeepGP.

Creates pickle files in the same format expected by SNPPCACHRDataModule:
  pukb/genes/<data_name>/<rd_mode>/
    genes.pkl      - list of 22 arrays of SNP names per chromosome
    pos.pkl        - list of 22 arrays of positional indices per chromosome
    snp_train.pkl  - list of 22 arrays, each (n_train, n_snps_chr)
    snp_test.pkl   - list of 22 arrays, each (n_test, n_snps_chr)
    label_train.pkl - list of 22 identical label arrays (values 0=control, 1=case)
    label_test.pkl  - list of 22 identical label arrays
    covar_train.pkl - array (n_train, 8)
    covar_test.pkl  - array (n_test, 8)
"""

import os
import pickle
import numpy as np
import argparse


def generate_dataset(data_name, n_participants, n_snps, n_cases, n_causal,
                     rd_mode="ld_all", base_dir="pukb/genes", seed=42):
    np.random.seed(seed)

    # Spread SNPs roughly evenly across 22 chromosomes
    n_chr = 22
    snps_per_chr = [n_snps // n_chr] * n_chr
    for i in range(n_snps % n_chr):
        snps_per_chr[i] += 1

    # Generate SNP names and positions per chromosome
    genes_chr = []
    pos_chr = []
    snp_idx = 0
    for c in range(n_chr):
        names = np.array([f"rs_syn_{snp_idx + j}" for j in range(snps_per_chr[c])])
        positions = np.arange(snps_per_chr[c])
        genes_chr.append(names)
        pos_chr.append(positions)
        snp_idx += snps_per_chr[c]

    # Generate genotypes: values in {0, 1, 2} (additive encoding)
    # Minor allele frequency drawn uniformly from [0.05, 0.5]
    all_snp_data = []
    mafs = []
    for c in range(n_chr):
        maf = np.random.uniform(0.05, 0.5, size=snps_per_chr[c])
        mafs.append(maf)
        # Binomial(2, maf) for each SNP
        geno = np.column_stack([
            np.random.binomial(2, m, size=n_participants) for m in maf
        ])
        all_snp_data.append(geno.astype(np.float64))

    # Select causal SNP(s) — pick from the first chromosome for simplicity
    # Build a liability score from the causal SNP(s)
    causal_chr = 0
    causal_indices = np.random.choice(snps_per_chr[causal_chr], size=n_causal, replace=False)
    causal_genotypes = all_snp_data[causal_chr][:, causal_indices]

    # Effect sizes for causal SNPs
    betas = np.random.normal(0.5, 0.1, size=n_causal)
    liability = causal_genotypes @ betas

    # Add noise and convert to probability via logistic
    noise = np.random.normal(0, 0.5, size=n_participants)
    liability += noise

    # Threshold to get exactly n_cases cases
    sorted_liability = np.sort(liability)[::-1]
    threshold = sorted_liability[n_cases - 1]
    labels = (liability >= threshold).astype(int)
    # Ensure exactly n_cases (handle ties at threshold)
    if labels.sum() != n_cases:
        labels = np.zeros(n_participants, dtype=int)
        top_indices = np.argsort(liability)[::-1][:n_cases]
        labels[top_indices] = 1

    # Labels use 0=control, 1=case
    label_values = labels

    # Train/test split: 80/20
    n_test = max(n_participants // 5, 2)
    n_train = n_participants - n_test

    indices = np.random.permutation(n_participants)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Split SNP data per chromosome
    snp_train = [all_snp_data[c][train_idx] for c in range(n_chr)]
    snp_test = [all_snp_data[c][test_idx] for c in range(n_chr)]

    # Labels: list of 22 identical arrays (the code asserts they're all identical)
    label_train = [label_values[train_idx] for _ in range(n_chr)]
    label_test = [label_values[test_idx] for _ in range(n_chr)]

    # Covariates: 8 dimensions (age, sex, 6 PCs)
    covariates = np.random.randn(n_participants, 8)
    covariates[:, 1] = np.random.binomial(1, 0.5, size=n_participants)  # sex
    covariates[:, 0] = np.random.normal(55, 10, size=n_participants)     # age
    covar_train = covariates[train_idx]
    covar_test = covariates[test_idx]

    # Write pickles
    out_dir = os.path.join(base_dir, data_name, rd_mode)
    os.makedirs(out_dir, exist_ok=True)

    for fname, obj in [
        ("genes.pkl", genes_chr),
        ("pos.pkl", pos_chr),
        ("snp_train.pkl", snp_train),
        ("snp_test.pkl", snp_test),
        ("label_train.pkl", label_train),
        ("label_test.pkl", label_test),
        ("covar_train.pkl", covar_train),
        ("covar_test.pkl", covar_test),
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

    generate_dataset(
        data_name=args.name,
        n_participants=args.n_participants,
        n_snps=args.n_snps,
        n_cases=args.n_cases,
        n_causal=args.n_causal,
        rd_mode=args.rd_mode,
        base_dir=args.base_dir,
        seed=args.seed,
    )
