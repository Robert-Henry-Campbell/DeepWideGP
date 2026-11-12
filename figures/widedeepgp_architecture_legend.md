**Figure 1. WideDeepGP.** Each of the 22 chromosomes is a sequence of standardised SNP dosages
with its own token embedding and its own bidirectional state-space encoder; the encoder output is
averaged over the chromosome's SNPs. The covariates pass through a small perceptron. One
attention layer runs across the resulting 23 vectors, and a shared hidden layer feeds one logit
per disease. Stacked outlines mark the parts that exist once per chromosome, with weights that
are not shared between chromosomes. Tinted boxes with a heavy outline are new or changed relative
to single-phenotype DeepGP (tagged "new" or "changed"); plain boxes are as published. Training
draws batches that are balanced on one disease at a time, in rotation, and every output learns
from every batch through a loss that corrects for how often each participant is drawn. The
annotation step is optional and is off in the base model. Its gain and offset are computed from
96 functional annotations and a no-annotation flag per SNP (97 channels) by one pair of maps
shared by all chromosomes, so that box is drawn once, outside the stack. With a model width of
32 and one encoder layer the network has 465,662 trainable parameters (table:
`widedeepgp_architecture_data.csv`).
