"""WideDeepGP: DeepGP widened to predict many diseases with one model.

The single-disease code (models/, layers/, utils.py, main_genome.py) is left as published.
This package holds everything the multi-disease model adds:

    labels.py     label matrices: validation, prevalence, sampling rates, class weights
    samplers.py   disease-balanced ("rotation") and shuffled batch samplers
    losses.py     sampler-independent weighted cross-entropy
    metrics.py    per-disease metrics over a whole evaluation set
    blocks.py     the sequence-block factory (current Mamba block, Mamba-3)
    embed.py      token embedding with per-SNP annotation gain and offset
    model.py      WideDeepGP, the Lightning module
    callbacks.py  snapshots, throughput, Optuna pruning
    train.py      one training run, shared by the command line and the tuner

methods.md at the repository root describes the science.
"""
