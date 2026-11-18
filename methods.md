# Methods: WideDeepGP, a multi-disease extension of DeepGP

*DeepWideGP repository, DPhil chapter 6. Written 11–12 November 2026. This document describes
the data, the model and how it is trained. A pilot on real data is reported in
DeepWideGP-Scripts (branch `widening`, `scripts/widedeepgp-pilot/`; its written results were
removed on 16 November 2026 and are in git history at 784009f); what it showed
about the model itself is summarised in section 4.9 and should be read before the model is
trained at full size.*

**Status, 17 November 2026.** The LD-block model decided on 16 November is implemented and is
the default of `main_multilabel.py` (`--input_mode blocks`, `--wide off`): one network trained
end to end from dosages, whose first layer is a learned table with a vector per SNP pooled by LD
block, followed by a mixer (the published bidirectional Mamba block or a transformer encoder
layer) over the block tokens. Section 4.11
describes it; sections 4.10, 6, 7, 8, 10, 12 and 13 were brought up to the new default on the
same day, with the earlier rules kept as dated notes. The decision followed the first training
size of the ladder, where the network added nothing to the ridge PRS path (ridge alone 0.6634,
ridge plus network 0.6633; the simulated-trait gate had shown the same). The linear PRS path of
section 4.10 stays in the code as an option, off by default, and is never part of the reported
model; the SNP-token mode of sections 4.1 to 4.9 remains as the published network widened
(`--input_mode snps`), used by the classic single-disease comparator. The study is styled after
Kaplan et al. (2020), with model size, participants and compute as its axes; its design, its
decisions and its costs are in DeepWideGP-Scripts `scripts/widedeepgp-blocks/methods.md` (branch
`widening`).

## 1. Question

DeepGP (Zhu et al., 2025) predicts one disease from genome-wide SNP data with a deep sequence
model: one bidirectional state-space encoder per chromosome, pooled and combined with
covariates. We ask whether one such network can predict many age-related diseases (ARDs) at
once, so that diseases share what the encoders learn, and what has to change in the data, the
network and the training to make that possible. This document covers the model that answers
that question, WideDeepGP, and marks every departure from the single-phenotype model
(section 10).

## 2. Data and phenotyping

### 2.1 Cohort

UK Biobank is a prospective cohort of about 500,000 adults aged 40–69 years at recruitment
(2006–2010) (Sudlow et al., 2015). We used the local extract of application 15181 (18 November
2024; 502,399 participants) and the laboratory's quality-controlled imputed genotypes.

| Step | Participants |
|---|---|
| In the phenotype extract | 502,399 |
| After removing withdrawals (list of 25 April 2024) | 502,165 |
| In the quality-controlled imputed-genotype files | 333,444 |
| Of those: not withdrawn | 333,324 |
| Of those: White-British genetic ancestry (field 22006; Bycroft et al., 2018) | 333,317 |
| Of those: no relative identified in UK Biobank (field 22021 = 0) | **275,872** |

All training, tuning and testing use the 275,872 unrelated White-British participants. Field
22021 = 0 means that UK Biobank's kinship inference found no relative of third degree or
closer among all genotyped participants, so no participant in any split has a relative in
another split.

**Held out and not used.** Labels, covariates and genotypes were prepared with the same
definitions for two further groups, which take no part in training, tuning or testing and are
kept for later validation:

- the 57,445 White-British participants with an identified relative;
- the 9,408 Asian and 7,631 Black participants of the laboratory's second genotype set (17,055
  genotyped, of whom 16 had withdrawn; none lacks a covariate).

### 2.2 Diseases

The prediction targets are 30 phenotypes selected by the analysis `find-ukb-ard-phenos`
(DeepWideGP-Scripts; 11 November 2026), whose `methods.md` gives the full account. In summary:

**Source of the ARD set.** The age-related diseases are the both-sex ARD set of the ardmr
resource: Global Burden of Disease (GBD) causes classified as age-related from trend statistics
of GBD estimates. Its 127 entries map, through GBD's cause-of-death code lists, to 909
three-character ICD-10 categories.

**Candidates.** A later ICD-10 edition moved haemorrhoids from I84 to K64, and in these data
I84 records end and K64 records begin around 2016. The two categories were therefore merged
into one phenotype (a case has either code), leaving 908 candidates.

**Selection.** Eight criteria were applied in a fixed order, and the 30 phenotypes with the most
cases were then kept. Each excluded phenotype is attributed to the first criterion it failed.

| Criterion | Excluded | Remaining | Excluded phenotypes |
|---|---|---|---|
| Candidates | – | 908 | |
| 1. Linked to an age-related GBD cause | 0 | 908 | |
| 2. More than 10,000 cases among unrelated White-British participants | 844 | 64 | |
| 3. Not a sign or symptom (ICD-10 chapter XVIII) | 5 | 59 | R00, R06, R07, R11, R19 |
| 4. Not highly sex-skewed (more than 95% of cases of one sex) | 3 | 56 | C50, C61, N40 |
| 5. Not a residual "other" category | 13 | 43 | B96, E87, I51, K22, K31, K52, K59, K62, K63, K92, M13, M25, M79 |
| 6. Not a record of health-care contact (Z00–Z99, Y40–Y84) | 9 | 34 | Y83, Z50, Z51, Z88, Z90, Z91, Z92, Z95, Z96 |
| 7. GBD code map describes the condition | 1 | 33 | J45 |
| 8. Not an infectious disease | 2 | 31 | J18, J22 |
| 9. The 30 with the most cases | 1 | 30 | J90 |

Criteria 2, 3, 4 and 6 are explicit rules; criteria 5, 7 and 8 are judgements recorded with
their reasons in that analysis. Three phenotypes with imperfect code maps were reviewed there
and kept (C44, N17, I95), and three are predominantly but not exclusively of one sex (K40, 92%
of cases male; M81, 83% female; E03, 79% female).

**The 30 phenotypes**, with cases in each split (section 2.5):

| Phenotype | ICD-10 | Training cases (%) | Tuning cases | Test cases |
|---|---|---|---|---|
| Essential (primary) hypertension | I10 | 68,309 (31.6) | 6,255 | 12,729 |
| Disorders of lipoprotein metabolism and other lipidaemias | E78 | 34,953 (16.2) | 3,153 | 6,378 |
| Diverticular disease of intestine | K57 | 30,506 (14.1) | 2,784 | 5,624 |
| Gastro-oesophageal reflux disease | K21 | 25,427 (11.8) | 2,387 | 4,671 |
| Gastritis and duodenitis | K29 | 24,564 (11.4) | 2,238 | 4,512 |
| Diaphragmatic hernia | K44 | 24,297 (11.3) | 2,271 | 4,524 |
| Chronic ischaemic heart disease | I25 | 23,558 (10.9) | 2,126 | 4,299 |
| Other arthrosis | M19 | 21,544 (10.0) | 1,966 | 4,006 |
| Haemorrhoids | I84, K64 | 21,296 (9.9) | 1,962 | 3,989 |
| Atrial fibrillation and flutter | I48 | 18,365 (8.5) | 1,670 | 3,346 |
| Gonarthrosis [arthrosis of knee] | M17 | 18,227 (8.4) | 1,675 | 3,476 |
| Other malignant neoplasms of skin | C44 | 17,671 (8.2) | 1,676 | 3,303 |
| Non-insulin-dependent diabetes mellitus | E11 | 17,345 (8.0) | 1,663 | 3,143 |
| Obesity | E66 | 16,800 (7.8) | 1,478 | 2,999 |
| Dorsalgia | M54 | 14,550 (6.7) | 1,310 | 2,631 |
| Benign neoplasm of colon, rectum, anus and anal canal | D12 | 14,337 (6.6) | 1,309 | 2,684 |
| Angina pectoris | I20 | 14,234 (6.6) | 1,284 | 2,609 |
| Other hypothyroidism | E03 | 13,242 (6.1) | 1,211 | 2,383 |
| Inguinal hernia | K40 | 12,541 (5.8) | 1,140 | 2,224 |
| Coxarthrosis [arthrosis of hip] | M16 | 12,286 (5.7) | 1,106 | 2,287 |
| Cholelithiasis | K80 | 12,194 (5.6) | 1,118 | 2,253 |
| Acute renal failure | N17 | 11,340 (5.3) | 1,007 | 2,043 |
| Chronic renal failure | N18 | 10,337 (4.8) | 943 | 1,911 |
| Other chronic obstructive pulmonary disease | J44 | 9,884 (4.6) | 986 | 1,906 |
| Internal derangement of knee | M23 | 9,585 (4.4) | 872 | 1,821 |
| Spondylosis | M47 | 9,492 (4.4) | 797 | 1,722 |
| Hypotension | I95 | 9,153 (4.2) | 834 | 1,639 |
| Heart failure | I50 | 8,851 (4.1) | 814 | 1,548 |
| Osteoporosis without pathological fracture | M81 | 8,604 (4.0) | 840 | 1,615 |
| Acute myocardial infarction | I21 | 8,090 (3.7) | 730 | 1,541 |

Haemorrhoids are the union of two categories (see above); every other phenotype is one
three-character category. Training prevalence runs from 3.7% (acute myocardial infarction) to
31.6% (hypertension), with a median of 6.7%. A participant of the training split has 2.5 of
the 30 phenotypes on average: 32% have none, 17% have one and 20% have five or more.

### 2.3 Case definition

A participant is a case for a phenotype if at least one record carries an ICD-10 code whose
first three characters are one of the phenotype's categories, in any of three linked sources:

1. hospital inpatient diagnoses (field 41270), in any diagnostic position of any admission;
2. the death registry, as underlying (field 40001) or contributory (field 40002) cause;
3. the cancer registry (field 40006; the extract holds the first registration only).

Ascertainment is lifetime: records of any date in the linkage period count, so cases include
conditions diagnosed before recruitment and after it. Every other participant is a non-case; no
participant is excluded for having related codes, and no phenotype is restricted to one sex.
The label matrix is built with the code of the selection analysis, and its case counts were
checked to equal that analysis's counts exactly, for all 30 phenotypes, in each of its three
populations (all participants, genotyped White British, unrelated White British).

### 2.4 Covariates

Twelve covariates enter the model, and the polygenic-score comparators are given the same
twelve: year of birth, sex (field 31) and the first ten genetic principal components (field
22009). They are standardised with the training-split mean and standard deviation.

Year of birth is used instead of age at recruitment because case status is ascertained over
the whole linkage period: with follow-up ending on a common date, a participant's age at the
end of follow-up is fixed by when they were born, not by when they happened to be recruited.
Field 34 (year of birth) is not in the extract. It was reconstructed as the date of the first
assessment (field 53) minus age at recruitment in completed years (field 21022) minus one half,
the midpoint of the year in which the birth must lie, so the error is at most half a year.

### 2.5 Splits

The 275,872 participants were split once into 215,872 for training, 20,000 for tuning and
40,000 for testing, by a seeded random permutation of the sorted participant identifiers (seed
20261111). The tuning set decides when training stops, which snapshot is kept and every
hyper-parameter; the test set is scored only after those choices are made, and the scoring
script refuses to read it without an explicit flag and logs each access.

With 20,000 tuning participants the rarest phenotype has 730 tuning cases, which puts the
standard error of its tuning AUROC near 0.011 (Hanley and McNeil, 1982, at an AUROC of 0.6);
the mean over 30 phenotypes is correspondingly steadier.

**Training sizes (the ladder), the same as the PGI arm's.** WideDeepGP is trained on exactly the
run specifications of the linear polygenic-index (PGI) comparators (`PGI/methods.md` section 2.5,
in `/mnt/sdg/robert/deepwidegp/PGI`; Robert's decision of 15 November 2026: "the same in every
way so we can compare results directly"): five nested training sets defined by the median over
the 30 diseases of the effective sample size, N_eff = 4 / (1/cases + 1/non-cases), halving from
the full training set: 13,199, 26,287, 53,498, 107,743 and 215,872 participants (median N_eff
3,372 to 53,908). Each is a prefix of one permutation of the sorted training identifiers (seed
1), so smaller sets lie inside larger ones and every disease keeps its prevalence. The tuning
and test sets are the same at every size and for both arms. The datasets are
`data/datasets/ard30/hm3plus_n13199`, `_n26287`, `_n53498`, `_n107743` and `hm3plus`.
Consequences: the two arms are compared size by size on identical participants, SNPs,
covariates, tuning and test sets, with no re-weighting; an earlier draft ladder (2,160 to 68,200
participants) is not used; and the smallest size, 13,199, is where the median disease's PGI
first becomes detectable in the test set (PGI arm's power calculation), so the bottom of the
learning curves is not a size at which nothing can be measured. Runs go from the smallest
training set to the largest, one after another in a single chained job that does not pause for
review (Robert, 15 November 2026), so that each size's results are available as soon as it
finishes while the next, twice as large and about twice as costly, runs; the chain stops only if
a run fails. With a search inside every run, the five sizes together cost a few hundred
GPU-hours, most of it at the two largest. The chain starts after the real-label check of
section 8 (`DeepWideGP-Scripts/scripts/widedeepgp-ladder/run_ladder_2026-11-15.sh`, with the
record and configuration). The first size ran on 15–16 November 2026 (summary:
`widedeepgp-ladder/rung_13199_summary.md`); the ladder was then stopped and its run folders
removed, at Robert's request, before changes to the code.

### 2.6 Limitations of the phenotypes

These are the limitations recorded by the selection analysis that bear on the labels:

- **Hospital-based ascertainment.** Conditions managed mainly in primary care or outpatient
  clinics (for example hypertension, osteoarthritis) are under-ascertained; a non-case may have
  the condition without an admission that records it.
- **Not population prevalence.** Ascertainment depends on follow-up time and birth year, few
  records predate 1997, and UK Biobank participants are healthier than the general population.
- **Partial cancer registry.** Only the first registration per participant is available.
- **ICD-9 records** (field 41271) were not used.
- **Withdrawals** are those of 25 April 2024; later withdrawals must be removed before
  publication.
- **Conservative relatedness filter.** Everyone with any identified relative is excluded,
  not one member of each related pair.

## 3. Genotypes

**SNP set.** The input is the HapMap3+ set of Privé et al. (2023): the 1,054,330 HapMap3
variants used by LDpred2 plus 389,866 variants added to improve tagging, 1,444,196 in all. The
laboratory's genotype files (UK Biobank imputed data v3 as released by Bycroft et al., 2018, on
GRCh37; bi-allelic SNPs, filtered by the laboratory on minor-allele frequency, Hardy–Weinberg
equilibrium and missingness) contain 1,258,200 of them (87.1%), matched by chromosome, position
and both alleles: 1,012,953 of the HapMap3 variants and 245,247 of the added ones. Of the
185,996 that are absent, 116,513 have a minor-allele frequency below 1% in UK Biobank (the files
hold nothing rarer than about 1%) and 69,483 failed the other filters. The model uses all
1,258,200, from 105,260 on chromosome 2 down to 17,844 on chromosome 21.

**No SNP selection.** Published DeepGP chooses its input SNPs separately for each disease, by a
GWAS p-value filter followed by clumping. Here one fixed set serves all diseases: a network
whose encoders are shared between diseases needs one input, and a fixed set also removes any
dependence of the input on the training labels.

**Encoding.** A genotype is the dosage of the alternate allele (0 to 2; imputed dosages kept at
a resolution of 0.01). Each SNP is standardised with its mean and standard deviation in the
training split; a missing genotype is set to the mean. Genotypes are stored in a compact
on-disk format and decoded on the GPU (`genostore/README.md`).

**Annotations.** For the annotated variant of the model (section 4.2), each SNP carries the 96
functional annotations of the baseline-LD model, version 2.2 (Gazal et al., 2017), as
distributed with SBayesRC (Zheng et al., 2024): coding, conserved, regulatory and
histone-mark annotations with their flanking windows, minor-allele-frequency bins, and
continuous measures such as allele age, recombination rate and background selection. The table
covers 1,250,863 of the 1,258,200 SNPs (99.4%); the others get zeros and a 97th channel marking
"no annotation". Each channel is standardised over the model's SNPs.

## 4. Model

The network maps a participant's genotypes and covariates to one logit per disease. Sections
4.1 to 4.10 describe the SNP-token mode (`--input_mode snps`), the published network widened,
and section 4.11 the LD-block mode that is the default since 17 November 2026. Sections 4.2,
4.7, 4.10 and 4.11 differ in kind from published DeepGP; section 10 lists every difference.

![Figure 1. WideDeepGP](figures/widedeepgp_architecture.png)

**Figure 1. WideDeepGP.** Each of the 22 chromosomes is a sequence of standardised SNP dosages
with its own token embedding and its own bidirectional state-space encoder; the encoder output
is averaged over the chromosome's SNPs. The covariates pass through a small perceptron. One
attention layer runs across the resulting 23 vectors, and a shared hidden layer feeds one logit
per disease. Stacked outlines mark the parts that exist once per chromosome, with weights that
are not shared between chromosomes. Tinted boxes with a heavy outline are new or changed
relative to single-phenotype DeepGP (tagged "new" or "changed"); plain boxes are as published.
Training draws uniform shuffled batches and every output learns from every batch through a
plain cross-entropy at natural prevalence (the default since 18 November 2026; the rotation of
disease-balanced batches with its sampling-rate correction and the class weight are comparison
variants, section 5). The annotation step is optional and is off in the base model. Its gain and offset are
computed from 96 functional annotations and a no-annotation flag per SNP (97 channels) by one
pair of maps shared by all chromosomes, so that box is drawn once, outside the stack. Beside the network, the
linear PRS path (section 4.10) gives every SNP and disease a weight of its own, with a ridge
penalty, and every covariate and disease a weight; its 30 logits are added to the head's, so the
network only has to add what a genome-wide PRS and the covariates miss (`--wide on`; `off`, the
default since 16 November 2026, is the network alone; `only` the path alone). With a model width
of 32 and one encoder layer the network has 465,662 trainable parameters and the path
37,746,390. The LD-block model that is the default since 17 November 2026 (section 4.11) is not
drawn here. (Vector version and the component table: `figures/widedeepgp_architecture.pdf`,
`figures/widedeepgp_architecture_data.csv`; drawn by `scripts/make_architecture_figure.py`.)

### 4.1 Input and token embedding

Each chromosome is a sequence of standardised dosages, one token per SNP, in genomic order. A
token's embedding is the sum of two parts, followed by dropout:

- a **value embedding**: a one-dimensional convolution over the dosage sequence with a kernel
  of three SNPs, from one channel to *d* (the model width), without bias and with circular
  padding, so each token also sees its two neighbours;
- a fixed **sinusoidal position embedding** of the SNP's index along the chromosome, with
  wavelengths in geometric progression (base 10⁶).

### 4.2 Annotation gain and offset (annotated variant)

With annotations switched on, the value embedding of SNP *s* is modified before the position
embedding is added:

    token_s = value_s × (1 + G a_s) + H a_s + position_s

where *a_s* is the SNP's standardised annotation vector and *G* and *H* are linear maps from
the annotations to the model width. The gain lets an annotation scale how strongly a SNP's
genotype enters the network, which mirrors how SBayesRC uses annotations (they change a SNP's
expected effect size); the offset lets it mark the SNP irrespective of genotype. Annotations
describe the SNP, not the person, so the gain and offset are computed once per chromosome and
shared by everyone in the batch. One pair of maps serves all chromosomes, since the meaning of
an annotation does not depend on the chromosome. Both maps start at zero, so the annotated
model starts as the plain model and weights trained without annotations can be loaded into it.

### 4.3 Sequence encoder

Each chromosome has its own encoder of *L* layers. A layer normalises its input (layer
normalisation), passes it through a bidirectional selective state-space block and adds the
result to a residual stream; a final root-mean-square normalisation follows the last layer.

The block is the bidirectional Mamba block of Vision Mamba (Zhu et al., 2024), built on Mamba
(Gu and Dao, 2023), as in published DeepGP. The input is projected to twice the model width and
split into a signal and a gate. The signal passes through a depthwise convolution (kernel 4)
and a SiLU non-linearity and then drives a selective state-space recurrence with 16 states per
channel, whose transition, input and output parameters depend on the token. The recurrence is
run once along the chromosome and once in reverse, and the two outputs are summed, gated and
projected back to the model width. In the variant used ("v1"), the two directions share the
convolution and the projections and differ only in their state matrices. Computation and
memory grow linearly with the number of SNPs, which is what allows a whole chromosome to be one
sequence.

**Mamba-3 (evaluated, not adopted).** Mamba-3 (Lahoti et al., 2026) is implemented as an
alternative block. Upstream Mamba-3 runs in one direction only, so two blocks are used per
layer, one reading the chromosome forwards and one backwards, each with its own weights, and
their outputs are summed. It is in single-input mode with a state size of 16, the state size of
the current block (Mamba-3's default of 128 would add parameters that outweigh the rest of a
block this narrow). At a model width of 32 the pair has 14,980 parameters per chromosome and
layer against the current block's 11,008. Two properties of the upstream implementation matter
here: its kernel computes the recurrence in bf16 whatever the precision of the rest of the
network, so a "32-bit" run with this block is 32-bit everywhere except inside the recurrence;
and its backward pass is incompatible with the checkpointing variant used for the current
block, so its encoders are checkpointed with the re-entrant variant, which gives the same
outputs and gradients. The decision rule was fixed in advance: Mamba-3 replaces the current
block only if it runs at chromosome length and its mean tuning AUROC in the pilot exceeds the
current block's by more than the seed-to-seed spread at equal GPU-hours; otherwise it is
dropped. It ran at chromosome length, with one limit (a call with more than about 122
participants at 105,260 SNPs fails inside the upstream kernel), needed about twice the memory
per participant without checkpointing, and in the pilot gave the same AUROC as the current
block at 1.6 times the training time. It was dropped; the code remains behind `--mixer mamba3`.

### 4.4 Pooling

The encoder's output is averaged over the SNPs of the chromosome, giving one vector of width
*d* per chromosome. Under mixed precision the average is taken in 32-bit floating point
(section 6).

### 4.5 Covariate encoder

The covariates pass through a two-layer perceptron (hidden width 256, ReLU, dropout 0.1) and a
root-mean-square normalisation, giving a 23rd vector of width *d*.

### 4.6 Attention across chromosomes

One transformer encoder layer (multi-head self-attention, a feed-forward layer of width 4*d*,
layer normalisation, dropout 0.1) runs over the 23 vectors, so that the representation of each
chromosome can depend on the others and on the covariates.

### 4.7 Output head

The 23 vectors are concatenated and passed through a linear layer to a shared hidden layer of
256 units (ReLU, dropout 0.1) and a final linear layer with one output per disease. The outputs
are logits; the sigmoid is applied in the loss. Diseases therefore share everything up to the
hidden layer and differ only in the last linear map: 257 parameters per disease.

### 4.8 Parameter counts

Trainable parameters with 30 outputs and 12 covariates, counted from the constructed network.
The number does not depend on the number of SNPs.

| Width *d* | Layers | One sequence block | 22 chromosome encoders | Covariate encoder | Attention layer | Head | Total | Annotation maps |
|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 11,008 | 244,992 | 11,584 | 12,704 | 196,382 | **465,662** | +6,272 |
| 32 | 2 | 11,008 | 487,168 | 11,584 | 12,704 | 196,382 | **707,838** | +6,272 |
| 64 | 1 | 34,816 | 771,584 | 19,840 | 49,984 | 384,798 | **1,226,206** | +12,544 |
| 64 | 2 | 34,816 | 1,537,536 | 19,840 | 49,984 | 384,798 | **1,992,158** | +12,544 |
| 128 | 2 | 120,832 | 5,327,872 | 36,352 | 198,272 | 761,630 | **6,324,126** | +25,088 |
| 256 | 2 | 446,464 | 19,666,944 | 69,376 | 789,760 | 1,515,294 | **22,041,374** | +50,176 |

The chromosome encoders include each chromosome's token embedding (3*d* parameters) and final
normalisation (*d*). The head is the linear layer from the 23 concatenated vectors to the
hidden layer (23·*d*·256 + 256) plus 257 parameters per disease. The annotated variant adds two
maps from the 97 annotation channels to the model width, 2·(97·*d* + *d*) parameters, shared by
all chromosomes.

For reference, the published single-phenotype configuration (*d* = 32, one layer, eight
covariates, one output) has 457,185 parameters. Thirty outputs instead of one add 7,453, and
twelve covariates instead of eight add 1,024.

### 4.9 What tells one SNP from another, and what the pilot showed

The encoders share their weights across the SNPs of a chromosome and their output is averaged
over the chromosome. A SNP-specific effect, with its own size and sign, can therefore enter
only through what distinguishes one token from another: the position code, and the annotation
vector where one is used.

The position code has 32 sinusoidal features at a model width of 32. A linear read-out of it (a
least-squares fit of the effect profile on the code and a constant, with the cut-off that
`numpy.linalg.lstsq` uses with rcond=None for directions that rounding cannot resolve; numerical
rank 10 to 31) reproduces every SNP's own effect when a chromosome has ten SNPs, as in the
synthetic test data of this repository, but only 11 to 17% of the true effect profiles of the
pilot's simulated traits on its GWAS-selected SNPs (26 to 267 per chromosome; a figure that
depends on the cut-off: about 9 to 14% at float32 precision, roughly 20 to 38% if every
direction is kept), 0.3 to 0.4% on all of the pilot's 157,279 SNPs (2,231 to 13,158 per
chromosome), and about 0.05% of an arbitrary effect profile at the 57,000 SNPs per chromosome of
the full dataset (DeepWideGP-Scripts, branch `widening`,
`scripts/widedeepgp-pilot/position_code_capacity.py`). The encoder is not linear, so this is a
heuristic, not a bound. The pilot bears it out:

- On the real labels (157,279 SNPs, 40,000 training participants, six epoch-equivalents, two
  seeds per scheme) every training scheme ended slightly below a logistic model on the
  covariates (the default by 0.0014 in mean tuning AUROC). Permuting the genotypes between
  participants changed nothing detectable in the eight planned schemes (-0.0003 to +0.0003 over
  their 16 runs). In a ninth, added during the pilot, each SNP carried the association
  *z*-scores of the 30 diseases from a GWAS of 165,872 other participants of the training split;
  there permuting cost 0.0004 and 0.0005 of mean AUROC in the two seeds, about a third of the
  0.0014 that a clumped polygenic score from the same GWAS adds to the covariates.
- On traits simulated on the same genotypes, with 20 to 20,000 causal SNPs and a known ceiling
  of 0.78 to 0.85, the network stayed at an AUROC of 0.51 where a clumped polygenic score or
  GBLUP fitted to the same participants reached 0.72 to 0.81. Mamba-3 did the same, and so did
  an annotation that marked the causal SNPs without saying which way each acts.
- Given only the 2,476 SNPs that a GWAS of the training participants selects, the kind of input
  published DeepGP receives, the network learned slowly (0.54 to 0.65 after six
  epoch-equivalents; continued to 30, it was at 0.54 to 0.64 at the end, after a best pass of
  0.55 to 0.68 at 13.5, chosen on the tuning set and so optimistic).
- Given every SNP's signed association *z*-score for each trait as annotation channels, it
  learned on all 157,279 SNPs (0.58 to 0.69 at the end of training). The annotation gain
  multiplies each SNP's dosage, so a signed statistic gives the network what it otherwise
  lacks: a size and a direction for every SNP. The same statistics kept only at clumped lead
  SNPs did not work: a few hundred marked SNPs are averaged away over a chromosome.
- The 96 functional annotations, which carry no direction, made no difference on the real
  labels and little on the simulated traits.

The consequence for the design is that the choice made in section 3, one unselected SNP set
with no per-disease information, leaves the network without the means to learn SNP effects, at
least at the scale tested. Of the ways SNP-specific information could enter (association
statistics as annotation, GWAS-based selection of SNPs, or parameters of its own for each SNP),
the third was chosen on 12 November 2026, keeping all SNPs and no GWAS: the linear PRS path of
section 4.10. Since 17 November 2026 the per-SNP parameters are the first layer of the network
itself (section 4.11), and the path is an option.

### 4.10 Linear PRS path

*An option since 16 November 2026 (`--wide on` or `only`), off by default and never part of the
reported model, available in SNP mode only: blocks mode (section 4.11) has per-SNP parameters of
its own and refuses the path. Written 12–13 November 2026; the description below is of the path
when it is on, as the runs of 13 to 16 November had it.*

The linear PRS path gives the model a parameter of its own for every SNP and disease, beside
the network. For disease *j*

    logit_j = deep_j + b_j + M^(-1/2) Σ_s x_s θ_sj + Σ_k u_k v_kj

where x_s is the standardised dosage of SNP *s* (the input the encoders read), M the number of
SNPs (1,258,200), θ one weight per SNP and disease (37.7 million in all), u the twelve
covariates, v one weight per covariate and disease, b an intercept per disease, and deep_j the
network's output. The model therefore contains a logistic regression on a genome-wide PRS and
the covariates, and the network only has to add what that misses. No GWAS and no SNP selection
are involved: the weights are learned with the rest of the model, from the training
participants alone. The path is computed in 32-bit arithmetic, also when the encoders run in
bf16. Its PRS and covariate weights start at zero and its intercepts at each disease's weighted
base rate, logit(π′_j) (section 5.2); when the path is on, the network's last layer also starts
at zero, so that at the first step the model is the base-rate model and the network adds only
what training gives it. Older weights load with the path at zero.

Three settings of `--wide`: `off`, the default since 16 November 2026 (the network alone: the
model of the pilot in SNP mode, and the only setting of blocks mode); `on` (network and path; the
default from 13 to 16 November 2026); `only`, the path alone, with no network (a linear comparator
inside WideDeepGP, trained by the same code; it also runs without a GPU). The path costs 151 MB of
weights (about 0.6 GB of GPU memory with the optimiser's state) and one matrix product per step,
which is negligible next to the encoders. Its penalty and learning rate are described in
sections 5.2 and 6, and how its settings are chosen in section 8.

**Several prior variances in one pass** (`--prs_grid`, with `--wide only`). The linear model can
be trained for K prior variances τ_1 … τ_K at once: the path then has K × n outputs (column
k·n + j is disease *j* under τ_k), each with its own weights, penalty (τ_k in place of τ) and
learning rate (the rule of section 6 at τ_k, applied by scaling each column's AdamW update by
τ_k/τ), and the training objective is the sum of the K models' objectives. Since the diseases
and the grid points share no parameter, every column follows exactly the path that a model
with that τ alone would follow on the same batches (checked to 10⁻⁴ in the tests), so K models
cost one pass over the genotypes, which is what a step of the linear model costs. At every
validation pass each disease is scored at every grid point, and each disease keeps the weights
of its best grid point and pass so far. The run's model is then the linear model that takes,
for every disease, its own best prior variance and pass: the per-disease tuning that the
polygenic-score comparators also receive (a threshold, a penalty or a heritability chosen per
disease on the tuning set). It is written as a run of its own (`per_disease/` inside the grid
run) and scored again on the tuning set, which reproduces the AUROC each disease was chosen by.
Before this, one τ was searched for all diseases together; on the simulated traits (section
8) the trait with 20 causal SNPs and heritability 0.3 trailed the others relative to GBLUP, which
had each trait's own heritability.

### 4.11 LD-block mode (the default since 17 November 2026)

Blocks mode (`--input_mode blocks`) replaces the token embedding, the per-chromosome encoders,
their pooling and the attention across chromosomes (sections 4.1 to 4.4 and 4.6) by a first layer
with a parameter of its own for every SNP, pooled by LD block, and one mixer over the blocks. The
decision and its reasons are in the status note at the top; the design document
(DeepWideGP-Scripts `scripts/widedeepgp-blocks/methods.md`, sections 1 to 4) gives the full
account and the experiments. For one participant, in order:

**The per-SNP table (first layer; `widedeepgp/blockencoder.py`).** Every SNP *s* has a learned
vector T_s of d_e numbers (`--embed_dim`, 16 by default; 4 and 64 on the table axis of the
scaling study). With `--snp_code dosage` the participant's standardised dosage x_s scales the
vector, and the pooled vector of block *b* is the sum Σ_{s∈b} x_s T_s. With `--snp_code genotype`
every SNP has three vectors, for 0, 1 and 2 copies, and the raw dosage x (not standardised: the
data module's unscaled path, as for `--snp_embed vocab`) weights them by the hat functions
max(0, 1 − x), 1 − |x − 1| and max(0, x − 1), so a whole-number dosage selects one vector and a
fractional one interpolates between two; dominance at a SNP is then exact in the first layer, at
three times the table. Blocks are contiguous runs of SNPs in store order, so the pooling is an
exact segment sum computed as one matrix product per block, of the batch's dosages of the run
with the run's rows of the table (one table per chromosome, SNP-major; the per-SNP products are
never materialised). The table starts at zero.

**The block tokens.** A learned identity vector per block and per chromosome (the position
encoding: with a fixed set of a few hundred to a few thousand tokens, position is identity, and
no sinusoid is used) is added to the pooled vector, which then passes through one small MLP shared
by all blocks (d_e → 4 d_e → d, GELU) and dropout (`--dropout`): one token of width *d* per block.
The covariates pass through the covariate encoder of section 4.5 and a layer normalisation: the
covariate token.

**The mixer.** One sequence of all block tokens and the covariate token (431 + 1 tokens with the
Privé partition) runs through `--e_layers` layers of the mixer: the published bidirectional Mamba
block of section 4.3 (`--mixer bimamba1`) or a pre-norm transformer encoder layer
(`--mixer transformer`: `torch.nn.TransformerEncoderLayer` with `--n_heads` heads, a feed-forward
layer of width 4*d*, GELU and dropout `--dropout`), then a final layer normalisation (of the
residual sum, for the Mamba block). `--d_model` and `--e_layers` set the mixer's size.

**The head.** The mean over the tokens, concatenated with the covariate vector, feeds the head of
section 4.7 (`combine`: a shared hidden layer, then one logit per disease; `--final_pool mean` is the
only pooling blocks mode accepts). Every prediction passes through the block MLP, the mixer and the
head; with d_e = 1 the pooled block values are a ridge PRS's per-block sums
(`tests/test_ml_blocks.py::test_the_first_layer_contains_a_ridge_prs`), so the first layer contains a
ridge, but the model has no linear path from the table to the logits. **A finding of 17 November
2026 that the reader of the results must know.** During the implementation a linear readout of the
table (the sum of the block vectors over all SNPs mapped to the 30 logits and added to the head's)
was tried, because on the synthetic six-trait dataset (4,000 participants, 132 SNPs) the model as
designed learned the three traits with shared causal SNPs only after a phase transition at the fourth
pass and never the two traits with causal SNPs of their own (AUROC 0.48 and 0.39 against ceilings
of 0.78 and 0.97, in every variant tried: a tenth of the table learning rate, two mixer layers,
shuffled batches without class weights, a narrower head), and with one SNP per block it learned
nothing at all. The cause is that a table starting at zero takes its gradient direction from a
random, fast-moving downstream network, so its Adam steps do not accumulate; with the readout every
trait was learned after one pass (mean AUROC 0.83 over the heritable traits against a ceiling of
0.86). Robert had the readout removed the same day: the model was never expected to work at a few
thousand participants, the question is whether it works at 200,000, and the E0 gate and the
top-rung pre-test are to answer it with the architecture as designed.

**Precision and memory.** Under `--precision bf16` the first layer, the covariate encoder and the
mixer run in bf16; the mean over the tokens, the head and the loss run in 32-bit, as the pooling and head of section 6 do, and the table's
master weights stay in float32. `--grad_ckpt` checkpoints each mixer layer (outputs and
gradients unchanged; `tests/test_ml_blocks_gpu.py` checks it on the GPU, and requires the largest
planned transformer, d = 256 with 8 layers, over 2,001 tokens to train a batch of 128 in under
40 GiB).

**The partition and its digest.** `--partition` names a `partition.npz`, or its folder, under
`data/reference/partitions/` (the layout of the design document, section 5: the store's
chromosome names and variant counts, a sha256 of its variant IDs, and the global block id of
every variant per chromosome), or `uniform:<k>` (k SNPs per block) for tests. The data module
checks the chromosomes, the variant counts and the variant digest against the store, keeps the
blocks of the dataset's regions renumbered from zero, and refuses blocks that are not contiguous
runs in store order or that cross a chromosome. Three nested partitions exist, built on 17
November 2026 (design document, section 5): `prive431` (Privé's 431 HapMap3+ blocks, 119 to
15,194 of our SNPs each), `ldsplit2000` and `ldsplit5000`. The partition as the model sees it, the
block index of every SNP of every sequence, is digested (sha256, 16 hexadecimal characters) and
saved with every checkpoint's hyper-parameters as `partition_digest`; a model refuses to be
built, loaded (`load_model`), started from (`--init_from`) or resumed against another partition.
The partition's name, digest and block count are in `config.json`.

**Parameter counts.** `config.json` holds, under `blocks`, the parameters by module and their
split into embedding parameters (the table and the identity vectors, which Kaplan et al.'s N
excludes) and non-embedding parameters (the block MLP, the mixer with its final normalisation,
the covariate encoder, and the head: N), beside the number of tokens. Counted
from constructed networks with the 1,258,200 SNPs, the `prive431` partition, 30 outputs, 12
covariates and the head width the chains use, `--head_hidden` 4 d (so that the head does not outweigh a
small mixer; the parser's default, 256, would give S1 a head of 16,638):

| Size (transformer, *d* × layers) | Table (d_e = 16) | Identity | Block MLP | Mixer | Covariate encoder | Head | N (non-embedding) | Total |
|---|---|---|---|---|---|---|---|---|
| S1, 16 × 1 | 20,131,200 | 7,248 | 2,128 | 3,312 | 7,472 | 4,062 | 16,974 | 20,155,422 |
| S2, 32 × 2 | 20,131,200 | 7,248 | 3,168 | 25,472 | 11,616 | 12,190 | 52,446 | 20,190,894 |
| S3, 64 × 4 | 20,131,200 | 7,248 | 5,248 | 200,064 | 19,904 | 40,734 | 265,950 | 20,404,398 |
| S4, 128 × 6 | 20,131,200 | 7,248 | 9,408 | 1,189,888 | 36,480 | 146,974 | 1,382,750 | 21,521,198 |
| S5, 256 × 8 | 20,131,200 | 7,248 | 17,728 | 6,318,592 | 69,632 | 556,062 | 6,962,014 | 27,100,462 |

A transformer layer has 12 d² + 13 d parameters and the final normalisation 2 d. The Mamba block
of section 4.3 has 11,008 parameters per layer at d = 32 and 34,816 at d = 64 (section 4.8), so a
Mamba mixer matched to S3's N takes six layers (209,024; N 274,910). The table has M d_e entries
(3 M d_e with the genotype code: 60,393,600 at d_e = 16) and the identity vectors (431 + 22) d_e;
on the table axis at S3, d_e = 4 gives 5,032,800 table entries and N = 261,870, and d_e = 64 gives
80,524,800 and N = 293,790 (the block MLP grows with d_e).

**Training.** Everything is trained by AdamW on the loss of section 5, from dosages, from
scratch, in two parameter groups (section 6). The table has its own group with three settings of
its own, decided on 17 November 2026 after the first top-rung runs (below): a learning rate

    lr_table = κ √B / √M

(`--table_kappa` κ = 0.03 by default, searched over 0.01 to 0.03 for a table that starts at zero and
over 3 × 10⁻⁴ to 10⁻² for one loaded with `--init_encoder`, whose entries must not be overwritten
in a few steps; B the effective batch size, M the number of SNPs; the rule of the linear PRS path
without its prior-variance and class-weight factors, since each table entry's gradient is its
dosage times the gradient of its block's token, of order 1/√M); a per-step weight decay
`--table_decay` (1 by default, searched over 0.3 to 3 from zero and over 10⁻³ to 3 × 10⁻² from a
loaded table, which decay 1 would shrink by e every 1/lr_table steps),
AdamW's decoupled decay, so that every entry shrinks by lr_table × decay of itself at each step
unless the data keep pushing it; and a large Adam epsilon `--table_eps` (10⁻⁷ by default). The
epsilon is the point: Adam divides each entry's step by the root mean square of its gradient, which
gives a SNP that does nothing the same step size as a SNP that matters, and 20 million such steps
accumulate into noise. With ε about eight times a null entry's gradient (measured 1.3 × 10⁻⁸ per
entry at batch 256 on the top rung, S3; the 99.9th percentile 8.5 × 10⁻⁸), an entry whose gradient
is small and inconsistent moves at a fraction of the rate, and only an entry with a consistent
gradient well above that scale moves at the full rate. Everything else trains at `--lr` with
weight decay `--l2`. Both groups follow the same schedule, a linear warm-up of `--warmup_steps`
(200) optimiser steps and a cosine decay to zero over the run's budget (`--lr_schedule cosine`),
the default of `main_multilabel.py` since 17 November 2026. There is no penalty in the loss in
blocks mode (the first version had Σ ‖T_s‖² / (2 N τ) with τ = `--table_var`; it was a million
times too weak to hold the table and is gone). The run's log has `train/lr_table` and
`train/grad_norm_table` beside the network's. Snapshots are written by the model
(`WideDeepGP.snapshot`) with the table in bfloat16, which `load_model` reads back into float32.

**The training recipe and its sources (17 November 2026, night).** After the first top-rung runs
(next paragraph), Robert fixed the training recipe to the one Sigurdsson et al. (2023) used for their
genome-local networks on 383,000 UK Biobank participants, the closest published system to this one,
and `main_multilabel.py` carries it as its defaults: Adam learning rate 10⁻⁴ for the network with
decoupled weight decay 10⁻³ (theirs: 1 × 10⁻⁴ and 1 × 10⁻³), dropout 0.5 (theirs, between layers),
batch 64 (theirs), **40% of the SNPs set to missing at random on every training step** with the rest
rescaled by 1 / 0.6 (`--input_mask 0.4`; theirs 40%, which they report was needed to stop the networks
overfitting), a plateau schedule that multiplies every learning rate by 0.2 after ten validation passes
without improvement (`--lr_schedule plateau`; theirs ×0.2 after ten validation steps), and early
stopping after sixteen passes without improvement that may act only after a burn-in of 2,000
optimiser steps (`--stop_patience 16 --stop_burnin 2000`; theirs: patience 16 to 20 with a burn-in of
1,000 to 2,000 iterations). **The ridge warm start is the default starting point of every run of the
scaling study**: the table starts from the rank-d_e decomposition of the rung's closed-form block
ridge (section 4.11 above; `make_warm_start.py` in DeepWideGP-Scripts), so the tokens carry
disease signal from the first step and the network learns to read them, and training then
proceeds end to end. Nothing linear remains in the architecture; the warm start chooses where
training begins. **Two conditions, found on 18 November 2026 after the first warm-started runs
ended at the covariate floor with their tables orthogonal to the warm start (design document
section 12, item 14):** the table must be scaled so that the block tokens it forms are of the same
order as the identity embeddings they are added to (`make_warm_start.py` scales them to unit RMS
over training participants; the raw ridge weights give tokens of RMS 10⁻⁴ against embeddings of
order 1), and the table's optimiser must not erase it (a warm-started single run uses κ = 10⁻³
and no decay; the search ranges above). Under this recipe the table's epsilon was re-measured (batch 64, masking): a null
entry's gradient has a root mean square of 3.3 × 10⁻⁸, so the default `--table_eps` is 2.5 × 10⁻⁷.
What the search tunes under this recipe is the table's pace and decay only (`--tune_space table`;
section 8). The objective under this recipe is the plain cross-entropy at natural prevalence of
section 5 (uniform batches, no class weight; the default since 18 November 2026). The other
published systems were read for the same purpose and point the same way:
Bellot et al. (2018) and van Hilten et al. (2021) trained networks only on pre-selected or
gene-mapped variants, and the recent transformers either pre-select SNPs by GWAS (G2PT, Lee et al.,
preprint) or prune to 13,615 SNPs before pretraining (SNPBERT, preprint); none trains a free table
over a million SNPs, and none beats well-tuned linear scores by much on polygenic diseases.

**Why these settings: the first top-rung runs (17 November 2026, afternoon).** With the first
version (Adam with its usual epsilon, κ = 0.1, the loss penalty at τ = 1, the pruner acting after
two validation passes and early stopping after eight), S1, S3 and S5 trained once each at the top
rung with fixed settings reached mean tuning AUROCs of 0.6610, 0.6624 and 0.6619 against 0.6627
for a logistic model on the covariates alone, and a 12-trial search at S1 reached 0.6626: the model
learned nothing from the genotypes. The logged curves showed two failures. At S3 the training loss
fell for the first third of the run and then rose while the table's sum of squares grew a
thousandfold and its gradient norm thirty-fold: Adam noise accumulating in the 20 million entries
and degrading the network. At S5 the table's gradient norm collapsed a hundredfold and the table
shrank towards zero: the network settled on the covariates. The large epsilon and the per-step decay
address the first failure; dividing each block's sum by the square root of its SNP count (above)
keeps the first layer's output on one scale whatever the block size, which the deeper mixers need;
κ at 0.03, the pruner held off until eight passes (`--tune_min_validations 8`) and early stopping
after sixteen (`--stop_patience 16`) give a slow-starting table time, at the cost of searches two
to three times longer. Whether the model then learns at 215,872 participants is the question the
relaunched chain answers first.

**Hyper-parameter search, selection and evaluation** are those of sections 7 to 9, with
`--table_kappa` and `--table_decay` added to the searched settings.

**What the mode cannot do.** Interactions between distant loci are representable in the mixer but
not identifiable at this training size; within-block interactions beyond the pooled vector's
passage through the block MLP are not modelled; the ridge path, the offset and the annotation
vectors are refused in blocks mode (`--wide off`, no `--annot`). The frozen cross-fitted first
layer of the design document's ablations (fixed, cross-fitted block ridge scores as the tokens'
input, against end-to-end training) is not implemented: `--init_encoder` with `--freeze_encoder`
fixes a table fitted in-sample (the warm start) or without labels (pretraining), which is not the
same ablation.

## 5. Training objective

### 5.1 Batches

**Default since 18 November 2026: uniform shuffled batches** (`--sampler shuffled`): shuffled
passes over the training participants, cut into batches of 64, so that every disease appears in a
batch at its population rate (9% of batches hold no case of the rarest phenotype, acute
myocardial infarction at 3.7%, and a batch holds 2.4 of its cases on average). This is how the
published DeepGP, Dibaeinia et al. (2025) and the linear comparators draw their data; the reasons
for the change are given at the end of 5.2.

**The rotation (the default until 18 November 2026; `--sampler rotation`, kept as a comparison
variant).** With the published model's batch of 21 a batch drawn at random often held no case of
the rarer diseases (45% of batches, for the rarest), so batches were drawn by a **rotation**:

- each batch has a *target* disease; half its participants are drawn from the target's cases
  and half from its non-cases (when the batch size *B* is odd, *B*/2 is rounded to an even
  number of cases and the rest are non-cases);
- one rotation makes every disease the target once, in an order reshuffled at every rotation;
- cases and non-cases of each disease are each visited in shuffled passes, so the cases of an
  uncommon disease recur many times while its non-cases come round once.

An *epoch-equivalent* is as many draws as there are training participants (215,872).

A batch balanced on its target is not balanced for the other diseases. They appear at their
population rate, raised by however much they co-occur with the target. Over a whole rotation,
cases of the rarest phenotype (acute myocardial infarction, 3.7% of the training split) are
7.8% of the participants drawn; the median across phenotypes rises from 6.7% to 11.3%, and
hypertension from 31.6% to 44.8%. No sampler could balance all 30 at once: a participant has
2.5 of the 30 phenotypes on average, and a batch that was half cases for every phenotype would
need 15.

All 30 outputs nevertheless learn from every batch, because a pass through the encoders is the
whole cost of a step and discarding 29 of the 30 labels would waste it. Under the rotation two
consequences follow, and the loss handles both: the balance between cases and non-cases has to come from
weights, and the people in a batch are not a random sample of the cohort.

### 5.2 Loss

For a batch of *B* participants,

    loss = (1/n) Σ_j [ Σ_k m_kj · c_j(y_kj) · BCE(z_kj, y_kj) / ω_k ] / (B · Z_j)

    c_j(1) = min( ((1 − π_j)/π_j)^γ , w_max ),   c_j(0) = 1
    ω_k    = (1/n) Σ_i [ ρ · y_ki/π_i + (1 − ρ) · (1 − y_ki)/(1 − π_i) ]
    Z_j    = π_j · c_j(1) + (1 − π_j)

where *k* indexes participants and *j* the *n* diseases, *z* is the logit, *y* the label, BCE
the binary cross-entropy, *m* an indicator that the label applies to the participant (always 1
for these 30 phenotypes) and π_j the prevalence of disease *j* in the training split; ρ is the
share of a batch drawn from its target's cases, ½ when *B* is even.

- **ω_k is the sampling rate of participant *k*** under the rotation (1 for everyone under
  shuffled batches, where no correction is applied): how often the rotation draws them relative
  to drawing people uniformly, computed from the numbers of cases and non-cases that a batch
  actually holds. At an even batch size a participant with none of the 30 phenotypes has
  ω = 0.55, the median participant 0.80, and the 1% drawn most often more than 3.1; the mean
  is 1 by construction. Dividing by ω is importance weighting: in
  expectation over a rotation, the batch sum equals the same average over the training cohort
  whatever the sampler. The rotation then changes how often an example is seen, not the
  function the network is asked to fit. Without it, the batch balanced on one disease is a
  case-control sample for that disease and a distorted sample for every disease that co-occurs
  with it, the situation of secondary phenotypes in case-control studies (Lin and Zeng, 2009;
  Monsees et al., 2009). The correction has a price in efficiency: weighted this way, the
  draws of a rotation are worth 80% as many uniform draws (Kish's effective sample size,
  1/(mean ω · mean 1/ω); Kish, 1965).
- **c_j is the class weight**, which sets how much a case counts relative to a non-case.
  γ = 0, the default since 18 November 2026, leaves the population rate: c_j(1) = 1 and every
  disease's term is its plain mean cross-entropy. γ = 1 would weight every disease to parity,
  except that the cap (w_max = 20 by default) holds the seven phenotypes with more than 20
  non-cases per case at a weight of 20, below parity; the square root, γ = ½, the default until
  18 November 2026 and now the comparison variant, keeps the weights of these phenotypes at 5.1
  or less where parity would need up to 25.7. Large weights on the one or two cases a small batch
  holds make the gradient noisy.
- **Z_j and the factor 1/n** put every disease on the same scale (each term is a weighted
  average with expected total weight one) and average the diseases equally, so that a frequent
  disease does not dominate the sum.

Because the objective is a class-weighted cross-entropy over the training cohort, a logit on
the population scale is obtained by subtracting log c_j(1), which is zero at the default γ = 0.
With one disease, uniform batches and γ = 0 the loss is exactly that of published DeepGP, so the
default objective is the published one extended to 30 diseases and averaged over them.

**Ridge penalty of the PRS weights** (section 4.10). The PRS weights carry a ridge penalty,
added to the loss rather than applied as decoupled weight decay, so that the trained PRS
approximates the posterior mode under a normal prior: θ_sj ~ N(0, τ), so that each SNP's effect
on the logit scale has prior variance τ/M and the whole PRS prior variance τ (`--prs_var`; with
`--wide on`, chosen per disease from a grid in the run's linear stage, sections 4.10 and 8). For a loss averaged over the N training participants the
posterior mode carries the penalty Σ_s θ_sj² / (2Nτ). The class weights, however, make a
disease's loss as curved as if it held more information than it does: at the base-rate
solution, the weighted loss's curvature is π′(1 − π′), with π′ = π c(1)/Z the weighted share of
cases, against π(1 − π) for the likelihood. The penalty is therefore multiplied by

    r_j = π′_j (1 − π′_j) / (π_j (1 − π_j)) = c_j(1) / Z_j²,

which restores the shrinkage a disease's real information calls for: from 1.11 for
hypertension (31.6%) to 3.82 for myocardial infarction (3.7%), geometric mean 2.53. The penalty
is averaged over diseases, as the loss is:

    penalty = (1/n) Σ_j r_j ‖θ_j‖² / (2Nτ).

The sampling-rate correction does not change it (the expected loss is the cohort's); the
covariate weights and intercepts are not penalised. N is the run's own number of training
participants, so the shrinkage grows as the training set shrinks. The test
`test_the_penalty_correction_is_the_curvature_ratio` checks the curvature ratio numerically at
three prevalences. The per-SNP table of blocks mode (section 4.11) carries no penalty in the loss:
it is held by its optimiser's weight decay, shared by all diseases, so no disease's class weight
applies to it.

**Why plain cross-entropy at natural prevalence (Robert, 18 November 2026).** The rotation and
the square-root weight were built for batches of 21. At batch 64 and prevalences of 3.7% and
above, 9% of uniform batches hold no case of the rarest disease and the mean batch holds 2.4, so
the problem they solved is gone, and the correction they need costs a fifth of the draws (Kish's
effective sample size, 80%). The class weight is a departure from maximum likelihood, the
objective of the linear comparators and of the published model: for a mostly additive signal it
does not remove bias, it adds variance. It also makes a rare disease's loss up to 3.82 times as
curved as its information warrants (r_j above, 1.11 to 3.82 across the 30 diseases) against a
table weight decay that is one number for all diseases, so the rare diseases' directions in the
table are shrunk less than the common ones' where the data call for more; and the ridge warm start
of section 4.11 is fitted at population weights, so under the weighted loss the table is pulled
away from its starting point from the first step. The published systems point the same way:
Dibaeinia et al. (2025) trained PRSformer on 3.8 million people with the plain sum of
cross-entropies at natural prevalence (celiac disease at 0.6%) and found focal loss,
task-uncertainty weighting and an averaged cross-entropy all worse; Sigurdsson et al. (2023)
used inverse-frequency weighted sampling without testing it against uniform batches; Kelemen et
al. (2025) oversampled cases to 50:50 and their networks lost to the LD-aware scores; and the
studies that undersampled controls to 1:1 (Zhu et al., 2026; Yang et al., 2025) halved their
effective sample size and weakened the linear baselines they compared with. The rotation with
γ = ½ is kept as one paired ablation cell of the scaling study (transformer S3 at the top rung
from the warm start; design document section 12, item 12); the runs before 18 November 2026 (the
v2 and v3 reference runs) used it.

### 5.3 Variants kept for comparison

Four switches exist so that the choices above can be tested rather than assumed: the rotation of
disease-balanced batches (`--sampler rotation`, the default until 18 November 2026) with or
without its sampling-rate correction (`--no_sampling_correction`); the class weight (`--gamma`,
`--w_max`); and a "target-only" loss in which only the disease a batch was balanced on learns
from it (`--loss_outputs target`, rotation only).

## 6. Optimisation

- **Optimiser.** AdamW (Loshchilov and Hutter, 2019). Until 17 November 2026 the default was a
  constant learning rate, as in published DeepGP; on 17 November `main_multilabel.py` moved to a
  linear warm-up of 200 optimiser steps and a cosine decay, and the same night to the plateau
  schedule of Sigurdsson et al. (2023), ×0.2 after ten validation passes without improvement
  (`--lr_schedule plateau`, section 4.11); `cosine` remains an option:
  a linear warm-up of `--warmup_steps` optimiser steps and a cosine decay to zero over the run's budget
  (section 4.11), and the SNP-mode recipes (the synthetic check and the classic comparator) pass
  `--lr_schedule constant --warmup_steps 0` to keep the published constant rate.
- **Batch size and accumulation.** The batch is bounded by GPU memory, which grows with batch
  size times number of SNPs. Gradients can be accumulated over several batches per update; under
  the rotation sampler (section 5.1, a comparison variant) the rotation order is reshuffled every
  rotation so that the same diseases do not always share an update.
- **Mixed precision.** The encoders can run in 16-bit (bf16) arithmetic, and since the pilot
  this is the setting for full-size runs. Pooling, the head and the loss always run in 32-bit:
  participants differ by under 2% of a pooled chromosome vector, which is finer than bf16
  resolves. In a forward probe with random weights and inputs at 87,521 SNPs, averaging in bf16
  distorted the between-participant part of the pooled vector by about 10%, and averaging the
  same activations in 32-bit by 0.3%. In the pilot, bf16 encoders gave the same mean tuning
  AUROC as 32-bit ones (0.6594 against 0.6593 over two seeds) in 0.81 times the training time
  and 0.62 times the peak GPU memory, at up to 13,158 SNPs per sequence. At full length (105,260
  SNPs on chromosome 2), 200 steps from the same seed and batches gave the same training loss to
  within 0.2% at every logged step and the same mean tuning AUROC (0.6110 against 0.6108), in
  0.74 times the time per step (0.76 in the capacity trials, which had a GPU to themselves) and
  0.59 times the peak memory; a whole run has not been compared.
- **Activation checkpointing.** Each chromosome's encoder (SNP mode) or each mixer layer (blocks
  mode) can be run without storing its intermediate activations, which are recomputed during the
  backward pass. Outputs and gradients are unchanged; in SNP mode memory is then set by the
  longest chromosome instead of the genome, at the cost of one extra forward pass.
- **Seeds.** Every run uses one seed, 1: it sets the initial weights, dropout and the order
  of the batches, and the sampler of the hyper-parameter search. The pilot, its simulated-trait
  control and its hyper-parameter search ran with it (the pilot's arms also once with seed 2,
  to measure seed-to-seed spread). The published single-disease code keeps its default of 33.
  The train/tune/test split was drawn once, with seed 20261111 (section 2).
- **Learning rate of the PRS weights.** With Adam each weight moves by about its learning
  rate per step whatever the size of its gradient, so for 37.7 million PRS weights the rate
  sets how much noise they keep around the penalised optimum, and the network's rate (about
  10⁻³) would leave the PRS mostly noise. The rate is therefore set by
  lr_θ = κ τ √B / (r̄ √M), with B the effective batch size and r̄ the geometric mean of the
  r_j; under this rule the noise depends on κ alone, and the weights settle in about 1/κ
  epoch-equivalents whatever the training size, the number of SNPs or the batch size (derived
  from Adam's per-weight step, and checked by simulation in the design review). κ = 0.1
  (`--wide_kappa`; settled in about 10 epoch-equivalents on the simulated traits; the runs of 13
  to 16 November 2026 used it. Since 17 November, when the path became an option,
  `main_multilabel.py` no longer sets it and the parser's default of 0.03 stands, so a run with
  the path on passes `--wide_kappa` explicitly); the linear
  stage of a run's search lets the rates fall on a cosine to zero over its budget, which removes
  most of the noise that remains at a constant rate. The PRS weights, the covariate
  weights with the intercepts, and the network form three parameter groups of one AdamW
  optimiser: the network keeps its learning rate and weight decay, the other two have no
  decoupled weight decay (the PRS weights have their ridge penalty instead). With a grid
  (section 4.10) the objective is summed over grid points, not averaged, so that every grid
  point's weights receive the gradient a model of its own would: AdamW divides by the square
  root of the gradient's running second moment plus a small constant (10⁻⁸), and gradients
  shrunk by the number of grid points would come close to that constant for the smallest prior
  variances.

**Measured at full size** (one 48 GB GPU; model width 32, one layer; 1,258,200 SNPs; the
analysis `widedeepgp-pilot` in DeepWideGP-Scripts, branch `widening`). The largest batch is 21
participants in 32-bit, 36 with bf16 encoders, 69 with checkpointing and 130 with both;
annotation vectors lower the 32-bit limit to 19. One epoch-equivalent takes 4.7 GPU-hours in
32-bit and 3.6 with bf16 encoders, and checkpointing costs about a quarter of the speed. The
network is fastest at batches of 8 to 16, so larger batches give fewer and larger updates, not
more throughput. A validation pass over the 20,000 tuning participants takes 7 to 10 minutes,
about 3% of an epoch-equivalent, so four passes per epoch-equivalent add about a tenth to the
training time. The per-SNP means and standard deviations of the training split are computed in
one pass over its genotypes (ten minutes) and cached with the dataset.

## 7. Model selection and snapshots

- **Validation.** Four times per epoch-equivalent, but at most once per 20,000 participants
  seen and at least once per epoch-equivalent, every output is scored on the whole tuning set:
  AUROC, average precision, and cross-entropy on the population scale. (A pass over the 20,000
  tuning participants costs 7 to 10 minutes at full size whatever the training size, so small
  training sets would otherwise spend most of their time validating.) Metrics are computed
  over all tuning participants at once, never per batch.
- **Stopping and the model kept.** Since 17 November 2026 early stopping may act only after
  `--stop_burnin` optimiser steps (2,000 by default; Sigurdsson et al., 2023), and the default
  patience is sixteen passes. Training stops when the mean AUROC across diseases has not
  improved for eight validation passes, or after 30 epoch-equivalents and at least 128,000
  participants seen (about 2,000 updates of 64). The checkpoint with the best mean tuning AUROC
  is the trial's model; section 8 says which trial is the run's.
- **Snapshots.** The weights are also saved at steps evenly spaced on a logarithmic axis (six
  per tenfold increase; eight in the pilot) and at every epoch-equivalent, at most 40 per trial
  (the regular ones are thinned first), with the number of updates, participants seen and
  training time of each. Since 17 November 2026 the model writes its snapshots itself
  (`WideDeepGP.snapshot`: the weights, with the per-SNP table of blocks mode in bfloat16, the
  hyper-parameters, the step and the participants seen), and `load_model` reads a snapshot or a
  checkpoint alike. After a run's search only the chosen trial (and, with `--wide on`, the first
  stage's per-disease model) keeps its weights; every trial keeps its records. Every snapshot can
  be scored afterwards, which gives performance as a function of training compute (as Kaplan et
  al., 2020, report loss against compute), and allows the best snapshot to be read off for each
  disease separately as a secondary analysis. The tuning set is scored at every validation pass
  as training runs; the test set is never scored during training: once it is unlocked (section
  9), the snapshots are scored on it, which gives test performance against compute without the
  test set ever informing a choice. The compute of a snapshot in floating-point operations
  follows from the model's size, the number of SNPs (in blocks mode also the number of tokens)
  and the participants seen, which are all recorded, so it is computed afterwards rather than
  logged.
- **Fixed tables are not saved.** Position tables and annotation matrices are functions of the
  SNP list and are rebuilt from the dataset; a saved model holds its weights and its
  configuration, with a digest of its positions and of its annotation matrices, and refuses to
  load against tables that differ from those it was trained with (an annotated model saved
  before the annotation digest was added, such as the pilot's annotation and GWAS-annotation
  runs, loads with a warning). In blocks mode the partition is likewise rebuilt from its file and
  checked by digest (section 4.11).
- **Continuing a search.** Rerunning a run's command continues its search: finished trials
  are kept; a trial that a crash left running is queued again with the same settings and
  carries on from its last checkpoint in its own folder (one that had written none starts again
  with the same settings), which matters at full size, where one trial can run for days; and a
  search continued with other arguments than it was started with is refused (only the number of
  trials may grow). The pruner sees a resumed trial's validation passes by their number in the
  run, so it is compared at the right rung. No run has a time limit (Robert, 15 November 2026).
- **Resuming, and starting from saved weights.** `--resume` continues a run from its last
  checkpoint (`last.ckpt`, or the highest `last-vN.ckpt` where Lightning wrote one, as in a run
  directory the pilot's code copied and resumed); a copy of a run directory (made while nothing
  writes to it), or the same directory reached by another path, is resumed as if in place,
  keeping its best checkpoint and best score. It refuses a checkpoint trained on other
  positions, annotations, block partition, diseases or training labels, or with other model,
  loss or optimiser settings, and a run whose `config.json` records other batches, seed, schedule or input
  scaling: the loss's constants and the optimiser's state are restored from the checkpoint and
  would silently override new values, and the resumed run's `config.json` replaces the first
  one. Only the planned length may change (the epoch-equivalent budget and the early-stopping
  patience), besides how the run is executed (activation checkpointing, data-loading workers,
  the validation batch size). `--init_from` starts a new run from saved weights; the output
  layer starts afresh when the source was trained for other diseases: a different number, or,
  when the source records its diseases (every WideDeepGP snapshot and checkpoint does), other
  names or another order. A single-disease model saved by `main_genome.py` records none, so a
  one-disease run loads its output layer as it is. In blocks mode the source must have been
  trained on the same partition.
- **A resumed run's repeated validation pass.** A resumed run repeats the validation pass at
  which its checkpoint was written, and Lightning restores that pass part-way through, so only
  its last batches are scored (found on 15 November 2026: a resumed grid run took every
  disease's choice, and its best checkpoint, from one batch of the tuning set). Every pass
  therefore checks that it scored the whole tuning set and otherwise scores all of it before
  its metrics are used, so a repeated pass reproduces the original exactly and can be neither a
  new best nor a new choice. When the repeated pass is also the run's best, results.json takes
  its metrics from that scoring (until 16 November 2026 the tuning set was scored a second time
  for them, 7 to 10 minutes at full size). The pilot's one resumed run is unaffected: its best checkpoint is a
  complete pass that its metrics record.

## 8. Hyper-parameter search

Every run tunes itself (Robert's decision of 12 November 2026): the best settings change with
the size of the training set, which the scaling study varies. A run, for one dataset and
training set, is by default one search of the network alone (stage `deep`; the default since 17
November 2026, when the linear PRS path became an option): 12 trials proposed by Optuna's
tree-structured Parzen estimator (Akiba et al., 2019), each an ordinary training run, over the
network's learning rate (10⁻⁴ to 3 × 10⁻³), weight decay (10⁻⁶ to 10⁻²), embedding dropout (0 to
0.4) and, in blocks mode, the table's pace κ (`--table_kappa`, 0.01 to 0.03 from zero, 3 × 10⁻⁴ to
10⁻² from a loaded table) and weight decay (`--table_decay`, 0.3 to 3 from zero, 10⁻³ to 3 × 10⁻²
from a loaded table; section 4.11), all but the dropout log-uniform; with `--tune_space table`
the table's two settings alone, the rest held at the recipe of section 4.11, which is what the
scaling study's chain does (six trials per cell). Asynchronous
successive halving (Jamieson and Talwalkar, 2016; Li et al., 2020) stops the trials whose mean
tuning AUROC trails at each rung, after at least `--tune_min_validations` passes (2 by default; the
chains of the scaling study use 8), and the
run's model is the trial with the best mean tuning AUROC. A cell of the scaling study runs its
search once with seed 1 and retrains the chosen settings with seeds 2 and 3 (`--tune_trials 0`
with the chosen `--lr`, `--l2`, `--dropout`, `--table_kappa` and `--table_decay`; the retrain reproduces the trial's
model settings exactly but for the seed, `tests/test_ml_blocks.py`), and reports the mean of
the three with the seed spread (Robert, 17 November 2026; the PGI arm stays at one seed).

**With the linear PRS path on** (`--wide on`; the default from 13 to 16 November 2026, SNP mode
only), the search has two stages:

1. **The linear model alone** (`--wide only`), one training run over a grid of prior variances
   (section 4.10; by default τ = 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and 30, half-decade steps
   over the range a disease's PRS can take, from a liability-scale heritability of a few percent
   to well over one half, and below, where the PRS is negligible and a disease whose SNPs give
   the model nothing usable falls back to its covariates; the first grid stopped at 0.03, which
   6 of the 30 diseases chose on the pilot's proxy region). Each disease keeps its best τ and
   validation pass, by tuning AUROC. Batches
   of 256, one validation per epoch-equivalent but at most one per 20,000 participants, the run's
   budget (30 epoch-equivalents and at least 128,000 participants), the learning rates on a
   cosine to zero over that budget, and early stopping after ten passes without improvement.
   Nothing else is searched: κ and the learning rate of the covariate weights and intercepts
   (10⁻², so that they reach their optimum within the 500 updates of the smallest training
   sets) only set how fast the same optimum is reached.
2. **The network on top** (`--wide on`; 12 trials by default), each an ordinary training run
   proposed by Optuna's tree-structured Parzen estimator (Akiba et al., 2019). Every trial
   starts from the first stage's per-disease model with its PRS weights fixed (the covariate
   weights and intercepts keep learning) and searches the network's learning rate, weight decay
   and embedding dropout. Asynchronous successive halving (Jamieson and Talwalkar, 2016; Li et
   al., 2020) stops the trials whose mean tuning AUROC trails at each rung.

With the path on, the run's model is the second stage's best trial, the network on top of the
linear model, whatever its mean tuning AUROC relative to the linear model alone (Robert's
decision of 15 November 2026): WideDeepGP was reported as the whole model, never as the
ridge-regression PGI inside it, which would duplicate the PGI arm, and the first stage's
per-disease model is recorded beside it in the run's `results.json` as an ablation
(`linear_only`: WideDeepGP without its network, with the difference between the two), so such a
run also shows what the network adds at its training size. (Until 15 November the linear model
was the run's model when the network did not beat it; the simulated-trait check below ran under
that rule.) With `--wide only` the search is the first stage alone. In every case trials use the
run's seed; each worker's sampler is seeded with the run's seed plus the worker's index. Trials
run in parallel, one per GPU, on the free GPUs minus one (`--tune_gpus auto`), or one after
another without a GPU. Each GPU's worker takes the next trial whenever it is free, counting and
claiming under a lock so that exactly the stage's number of trials is started (since 16 November
2026: before, each worker had a fixed share, and on the first training size one slow trial kept
a worker busy for about six hours while the other three GPUs stood idle). Model width, depth,
the mixer, the table's width and SNP code, the partition and the use of annotations are
experimental factors and are never searched.

**Checks on the pilot's proxy** (gates, with the linear path on; methods in
`DeepWideGP-Scripts/scripts/widedeepgp-gates/`; its results files and run folders were removed
on 16 November 2026, the numbers below are the record). On five traits simulated on the proxy's real
genotypes, the linear stage reached a mean tuning AUROC of 0.7187 against 0.7132 for GBLUP given
each trait's true heritability, and the network on top added nothing (its best trial 0.7187,
below the linear model alone in the fifth decimal; under the rule of that day the linear model
was kept, under the rule decided on 15 November the network on top is the run's model). On the 30
diseases, the linear model at one prior variance for all reached 0.6616 against 0.6607 for the
covariates alone (0.6641 with prior variance and pass chosen per disease on the tuning set,
which is optimistic); the pilot's network alone had reached 0.6593. The network-on-top check on
the 30 diseases has not been run (section 7: its first attempt was stopped when the resume
problem was found).

**Decided (15 November 2026).** Robert asked whether the linear stage made WideDeepGP partly a
simple linear PGI, overlapping the PGI arm, the study's linear comparator: as first built, a run
reported the linear model whenever the network did not improve on it. He decided that the
reported result is always the linear model with the network on top, whichever does better; the
linear model alone is kept only as a labelled ablation (above). **Superseded on 16 November
2026, implemented on 17 November:** the path is off by default and the model is the LD-block
network of section 4.11 (status note at the top); the rule above applies only to runs with
`--wide on`.

The search inside every run replaces the plan's search on a proxy followed by a comparison at
full size. In the pilot, on the network alone, a search on the real labels of the proxy was
uninformative: every training scheme ended slightly below a logistic model on the covariates
(section 4.9); and an eleven-trial search on the simulated-trait control (one trial completed,
nine pruned, one stopped by hand), on GWAS-selected SNPs, found no setting that learned faster
than the pilot's settings.

## 9. Evaluation outputs

For any snapshot and split the scoring script writes, per disease: AUROC, average precision and
population-scale cross-entropy; the same for a logistic model on the twelve covariates alone,
fitted on the training split; and the difference between the two AUROCs. Per-participant scores
are written only on request and only beside the dataset. These are WideDeepGP's own checks.

**The comparison with the PGI arm** follows the protocol written for both arms
(`PGI/methods.md` sections 6 and 8): every model's final prediction for a disease is a logistic
regression, fitted on the tuning set, of disease status on the model's score (for WideDeepGP its
logit for that disease) plus the covariate basis: since 18 November 2026 the 12 covariates
entered linearly (year of birth, sex and the ten principal components; before, a natural cubic
spline of year of birth with four degrees of freedom by sex plus the ten principal components,
kept only as a robustness check of the final comparison; `PGI/methods.md` 6.1 and 13); the
covariates-only reference is the same regression without a score. The test set is opened once,
for both arms together, after every choice at every training size is recorded; the metrics,
intervals and learning curves are those of that document.

**The classic DeepGP as the single-disease comparator** (Robert's decisions of 15 November
2026). The headline model is WideDeepGP. The only single-disease comparator is the classic
DeepGP (the published single-disease model on GWAS-selected SNPs), for one disease, the one
with the highest effective sample size of the 30, hypertension (I10; N_eff 186,775 in the full
training set), at each of the five training sizes; the single-disease WideDeepGP runs of the
original plan (E11, I25, I48, M17, K80) are dropped. At each size:

- **SNPs**, as the published pipeline selects them: from the PGI arm's REGENIE GWAS of I10 on
  that size's training participants, the SNPs with *p* < 0.05, LD-clumped with plink2 at
  r² 0.1 within 250 kb (the published clumping parameters were not available; these are the
  standard ones and the strictest of the PGI arm's C+T options), LD from the PGI arm's LD panel
  (20,000 training participants, the panel its C+T clumps on); details in
  `DeepWideGP-Scripts/scripts/classic-deepgp-comparator/methods.md`.
- **Model and training as published**: model width 32, one encoder layer, four attention
  heads, attention pooling, a two-layer head with 256 hidden units, 32-bit arithmetic, batches
  of 64 drawn at random, AdamW with learning rate 7 × 10⁻⁴ and weight decay 10⁻⁴, dropout 0.2,
  and an unweighted cross-entropy loss; with the twelve covariates of every other model.
- **Run under the study's protocol**: seed 1; the same per-run search as WideDeepGP's network
  (12 Optuna trials over learning rate, weight decay and dropout, the published settings being
  the first), in SNP mode with the published constant learning rate (`--input_mode snps
  --lr_schedule constant --warmup_steps 0`, passed explicitly since the defaults changed on 17
  November 2026), early stopping on the tuning set, the tuning set scored as a whole; the test set
  only at the joint unlock. The published script itself is not used as it stands: it fixes the
  seed (33) and the data folder, scores the test set at the end of every run and writes
  per-person predictions into the repository. Its network and training are reproduced by the
  model code (the same output to within 10⁻⁶ and the same initial weights from the same seed;
  section 12), with one disease, no class weights (γ = 0) and shuffled batches.

Consequences: for I10 the comparison holds three kinds of model at every size (the seven PGIs,
the classic DeepGP on GWAS-selected SNPs, and WideDeepGP on all HapMap3+ SNPs); the selection
uses only training participants of that size, so no tuning or test label enters it; each size
needs its own SNP set and genotype dataset; and with one disease it says nothing about the
other 29. The runs follow each WideDeepGP size in the ladder's chain.

**Model size** (Robert, 15 November 2026). Each ladder varies one factor and holds the others
fixed. The training-size ladder above keeps the model at width 32 and one layer; model size is
varied later, several times, at the largest training size only. **Superseded on 16 and 17
November 2026** by the Kaplan-style design of the design document (sections 2 and 4): five mixer
sizes at every training size, the table's width as a second size axis at the middle size, and
the block count as a one-factor side study; the classic comparator keeps width 32 and one
layer.

## 10. What changed from single-phenotype DeepGP

| Component | Single phenotype (published) | Multi-phenotype (this model) |
|---|---|---|
| SNPs | Chosen per disease: GWAS p-value filter, then clumping | One fixed set for all diseases: 1,258,200 HapMap3+ SNPs |
| SNP effects | Learned by the network from the selected SNPs | A learned vector of d_e = 16 numbers per SNP as the network's first layer, pooled by LD block and read out linearly beside the head (section 4.11; the default since 17 November 2026). From 13 to 16 November 2026 a linear PRS path beside the network (one weight per SNP and disease, ridge penalty; section 4.10), reported as the whole model with the linear part alone as an ablation; now an option, off by default, in SNP mode only |
| SNP annotations | None | Optional per-SNP annotation vectors as a gain and offset (96 functional annotations; not adopted after the pilot, see section 4.9; SNP mode only) |
| Labels | One per participant | One per disease per participant |
| Covariates | Eight | Year of birth, sex, ten principal components |
| Sequence block | Bidirectional Mamba | The same, over SNP tokens (SNP mode) or over LD-block tokens, or a transformer encoder layer over the block tokens (blocks mode, section 4.11); Mamba-3 in a bidirectional wrapper was evaluated and not adopted |
| Output | One sigmoid probability | 30 logits from the same hidden layer |
| Batches | Shuffled | Shuffled (the default since 18 November 2026); a rotation of disease-balanced batches as a comparison variant |
| Loss | Binary cross-entropy | Binary cross-entropy per disease, averaged across diseases (the default since 18 November 2026); a class weight and a sampling-rate correction as comparison variants |
| Validation metric | Mean of per-batch AUROCs | AUROC of each disease over the whole tuning set; their mean for stopping |
| Validation schedule | Every 300 batches | Four times per epoch-equivalent, but at most once per 20,000 participants and at least once per epoch-equivalent |
| Saved models | The best | The best, the last, and snapshots along training |
| Hyper-parameters | Set by hand | Chosen inside every run by Optuna: the network's learning rate, weight decay and dropout and, in blocks mode, the table's prior variance (12 trials; section 8). With the linear path on: first the linear model's prior variance per disease from a grid, then the network on top |
| Precision and memory | 32-bit; all chromosomes' activations held | bf16 encoders with 32-bit pooling, head and loss (adopted for full-size runs after the pilot); optional checkpointing per chromosome (SNP mode) or per mixer layer (blocks mode) |

**Unchanged in SNP mode:** the dosage input and its standardisation, the convolutional and
positional embedding, the per-chromosome encoders and their block, mean pooling, the covariate
encoder, the attention layer across chromosomes, the shared hidden layer, and the AdamW
optimiser. With one disease, the same weights give the same output as the published network to
within 10⁻⁶, and the same random seed gives identical initial weights. Blocks mode (section 4.11)
replaces the embedding, the per-chromosome encoders and the attention layer by the per-SNP
table, the block MLP and one mixer over the blocks, and keeps the dosage input, the covariate
encoder, the form of the head, the loss, the sampler and the optimiser.

## 11. Departures from the published implementation, and known quirks

The single-phenotype code (`models/`, `layers/`, `utils.py`, `main_genome.py`) is left as
published, apart from fixes for the installed software versions, labels coded 0/1 instead of
1/2 (April 2026) and the option to read the genotype store. Four of its properties are
deliberately reproduced, so that the comparison with the published model is like for like:

- The published model's weight-initialisation call runs before its layers exist and has no
  effect; weights keep the default initialisation of their layers. WideDeepGP does the same.
- The published configuration does not pass `--bi_direct`, which selects the "v1" bidirectional
  variant described in section 4.3, not a one-directional model.
- `--dropout` sets the embedding dropout only; the dropout of the covariate encoder, attention
  layer and head is fixed at 0.1.
- The convolution of the value embedding uses circular padding, so the first and last SNP of a
  chromosome are treated as neighbours.

The alternative value embedding, `--snp_embed vocab` (a table of 201 vectors, one per hundredth
of a dosage, in place of the convolution), reads raw dosages: SNPs are not standardised for it.
WideDeepGP indexes the table by the genotype store's own code, the dosage times 100 rounded; a
missing dosage takes the vector of dosage 0, where the convolution sees the mean. The published
module's forward rounds to two decimals and then truncates in float32, which puts 6 of the 201
levels (0.53, 0.59, 1.05, 1.06, 1.17 and 1.18) one code low. Because the linear PRS path
(section 4.10) assumes standardised SNPs, vocab runs only with `--wide off`. Until 16 November
2026 WideDeepGP passed the dosages to the table unconverted, which fails at the first forward
pass, so no run has used this option.

Two published options do not run: the parser's default pooling (`--final_pool concat`) and the
SNP-only mode (`--dm snps`), which names a class that is not in the repository. WideDeepGP does
not offer them either: its entry points replace `concat` with `atten`, the pooling of the
published configuration (`mean` and `last` are also supported), and training stops with an
error when a model without covariates is asked for (`--dm snps`, or `--use_cov`, which switches
the covariates off) or the dataset names no covariates.

## 12. Tests

`tests/test_ml_wide.py` checks the linear PRS path and the search inside every run: the
path's arithmetic (a PRS plus a logistic model on the covariates, intercepts at the weighted base
rate); the penalty and its curvature correction (against the Hessian from automatic
differentiation, within 5%); the parameter groups and the learning-rate rule; that with the path
at zero the network's output is unchanged, and that bf16 leaves the path in 32-bit (both on a
GPU); older weights and a new set of diseases; the refused combinations; the validation floors
and the snapshot cap; that every grid point trains exactly as a model of its own (six AdamW
steps through the model's own optimiser step, to 10⁻⁴); a search end to end (each disease's
choice is the best of metrics.csv over passes and prior variances, rescoring reproduces it, the
grid's weights are deleted and the chosen model kept, the scoring script scores it, a rerun
continues, a rerun with other arguments is refused, a moved run folder continues); trials left
running by a crash are replaced; which GPUs a search takes; and that a resumed run scores the
whole tuning set at every pass. On 13 November 2026 every GPU suite passed on commit 81ded30
(test_ml_wide 21 tests, test_ml_model 9, the slow training tests, and the capacity probe on the
current and the pilot's code), except one test of the standalone tuner, which timed out under
the new 128,000-participant budget floor; with the small-data floors of the other tests it
passed (197741a).

`tests/test_ml_*.py` check, among other things, that: only named label and covariate columns are
read; the rotation sampler draws each participant at exactly the rate ω that the loss divides
by, for odd batch sizes too; the expected batch loss under the rotation equals the cohort loss
with the correction and differs without it; published weights give the published output;
checkpointing leaves outputs and gradients unchanged; a resumed run continues the same batches,
and a checkpoint trained differently is refused; and saved models hold no fixed tables. Learning
checks train for a few minutes each: one disease on the published smoke-test data must land in
the published AUROC band, and on a synthetic six-trait dataset the heritable traits must be
learned, the rare one included (2% prevalence), while a trait with no genetic signal that
co-occurs with a heritable trait must stay within an AUROC of 0.40 to 0.60; with annotation
vectors that mark the causal SNPs the model must reach the higher bands of annotated runs. The
bands are computed from five training seeds (`scripts/multilabel_synthetic_seeds.csv`): the
lowest seed minus the larger of 0.04 and 2.5 seed standard deviations. `tests/test_ml_mamba3.py`
checks the Mamba-3 wrapper in its own environment: one block reads in one direction only, the
pair reads both ways, and reversing the input while exchanging the two blocks' weights reverses
the output. With the Mamba-3 block the six-trait synthetic run passed the same bands as with the
current block; this was the last of the five gates Mamba-3 had to pass before the pilot (record:
`/mnt/sdg/robert/deepwidegp/runs/gate_evidence/multi6_mamba3_gate5`).

Added on 16 November 2026: `tests/test_ml_vocab.py` checks that `--snp_embed vocab` looks up
each of the store's 201 levels at its own code, equals the published module wherever the
published conversion is right, refuses the linear PRS path, and trains (on a GPU), and that the
default embedding is still the published one. `tests/test_genostore_identifiers.py` checks that
no participant ID or value reaches an error, chained exceptions included (unknown participants
are counted, a non-number cell is not quoted, other label values are counted), and that
per-person scores are refused in a git work tree or a run directory. In `tests/test_ml_train.py`,
on the CPU, a resumed run scores only what it lacks: the pass it repeats once, an earlier best
step once more, with a grid as its validation passes did, and results.json matches metrics.csv
in each case.

Added on 17 November 2026 for blocks mode (section 4.11). `tests/test_ml_blocks.py`, on the CPU
with the transformer mixer: a `uniform:<k>` partition is contiguous blocks numbered across the
chromosomes; a partition file is checked against the store (variant counts, variant digest,
contiguity, each in turn refused), a dataset region keeps its blocks renumbered from zero, and
blocks mode without `--partition` is refused; the pooling equals numpy segment sums of the
dosage-scaled vectors at d_e = 1, 16 and 64; the genotype code's hat weights, that three
collinear vectors (0, v, 2v) reproduce the dosage code and that it reads raw dosages; with the
head's last layer at zero the logits are the ridge PRS of section 4.11 (one PRS with a scale per
disease at d_e = 1, each disease's own combination of 16 scores at d_e = 16); the output shape,
the state-dict names and the parameter counts by module (the sum equals the model's); the
transformer block keeps the block interface; the refusals (`--wide on`, annotations, another
pooling, another or no partition, an unknown SNP code); `--resume` and `--init_from` refuse
another partition; the table's learning rate follows its rule and forms the second optimiser
group without weight decay, its penalty has the stated value and a frozen table has neither
group nor penalty; an `--init_encoder` file is checked (manifest, shapes, tables present),
loaded and may be frozen; a run trains, writes `config.json`'s `blocks` section (whose embedding
and non-embedding counts sum to the model's parameters) with the cosine recipe, snapshots the
table in bfloat16 and loads it back in float32, is scored with `--save_scores`, resumes on the
same data and refuses a resume with another `--embed_dim`; a one-trial search runs the `deep`
stage alone, records the new scalars in `search.json`, searches the table's pace and decay, and a retrain of
the chosen settings with seed 2 reproduces the trial's model settings (only the seed, the run's
name and the GPU list differ); masked pretraining lowers its loss on the synthetic store, writes
the file owner-only with the expected entries, a run starts from it with `--freeze_encoder` and
the file is refused for another partition; `main_multilabel.py`'s defaults are blocks mode,
`--wide off`, the warm-up and cosine recipe and `--final_pool mean` (`atten` in SNP mode). Its
learning check trains the blocks model for three passes on a synthetic six-trait dataset of
3,000 participants and asks only for mechanics: three recorded passes, a training loss that does
not rise and the null trait within 0.3 to 0.7 (whether the first layer learns per-SNP effects is
the E0 gate's question, section 4.11). `tests/test_ml_blocks_gpu.py`, on a GPU: both mixers train a step
in bf16 with finite, non-zero table gradients and float32 master weights; activation
checkpointing leaves the loss and the gradients of the table and the mixer unchanged (to 10⁻³);
the largest planned model (d = 256, 8 transformer layers, or 14 Mamba layers) over 2,001 tokens
at a batch of 128 peaks below 40 GiB; and the mixer parameter counts of the planned sizes are
printed, with the transformer's S3 within 5% of 12 L d².

## 13. Software and reproduction

Python 3.10, PyTorch 2.10, PyTorch Lightning 2.0.8, `mamba_ssm` 1.1.1 from the Vision-Mamba
fork, Optuna 5.0; Mamba-3 runs in a second environment with upstream `mamba_ssm`
(`scripts/setup_mamba3_env.sh`). The genotype conversion (`genostore.convert`, run from the
DeepWideGP directory) needs `pgenlib`, which neither environment has; it ran under Python 3.11
with pgenlib 0.95.2 in `/mnt/sdg/robert/miniforge3/envs/gsem-py`. Each run's `config.json`
records every setting together with the code directory, its git commit (none for a copy outside
git), whether any tracked .py file differs from that commit (`git_dirty`; the differences
themselves, and untracked files, are not saved) and the Python interpreter that ran it.

    # SNP list, labels, covariates, splits, annotations (DeepWideGP-Scripts, branch widening)
    Rscript scripts/build-ard30-dataset/make_snp_list.R
    <gsem-py>/bin/python -m genostore.convert --pgen-dir <pgen dir> --pattern "UKB_chr{chrom}" --chroms $(seq 1 22) \
        --extract <data>/reference/hm3plus_in_ukb_pgen.tsv --out <data>/genostore/hm3plus   # in DeepWideGP
    python scripts/build-ard30-dataset/build_dataset.py
    python scripts/build-ard30-dataset/describe_dataset.py     # cases by split, sampling rates

    # partitions (DeepWideGP-Scripts, scripts/widedeepgp-blocks: make_partition.R, check_partition.py,
    # run_make_partition.sh; design document section 5) -> <data>/reference/partitions/{prive431,ldsplit2000,ldsplit5000}

    # train, score, tune (DeepWideGP): the LD-block model (section 4.11), and the SNP-token mode
    python main_multilabel.py --data_dir <data>/datasets --label ard30 --rd_mode hm3plus --snp_stats train \
        --partition <data>/reference/partitions/prive431 --mixer transformer --d_model 64 --e_layers 4 --n_heads 4 \
        --embed_dim 16 --snp_code dosage --precision bf16 --batch_size 256 --exp_name <run>
    python main_multilabel.py --data_dir <data>/datasets --label ard30 --rd_mode hm3plus --snp_stats train \
        --input_mode snps --lr_schedule constant --warmup_steps 0 --final_pool atten --d_model 32 --e_layers 1 \
        --n_heads 4 --precision bf16 --batch_size 16 --accumulate 4 --exp_name <run>
    python score_snapshots.py --run runs/<experiment>/<run> --split val
    python tune_optuna.py --study <name> --n_trials 20 -- <the training arguments>
    CUDA_VISIBLE_DEVICES=<gpu> python pretrain_blocks.py --data_dir <data>/datasets --label ard30 --rd_mode hm3plus \
        --partition <data>/reference/partitions/prive431 --embed_dim 16 --out <dataset>/blocks/pretrained_prive431.pt

    # tests
    python -m pytest tests/test_ml_labels.py tests/test_ml_sampler.py tests/test_ml_loss.py tests/test_ml_blocks.py -q
    CUDA_VISIBLE_DEVICES=<gpu> python -m pytest tests/test_ml_model.py tests/test_ml_train.py tests/test_ml_blocks_gpu.py -q
    CUDA_VISIBLE_DEVICES=<gpu> bash scripts/multilabel_synthetic.sh

## References

- Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter
  optimization framework. *KDD* 2019.
- Bellot P, de los Campos G, Pérez-Enciso M. Can deep learning improve genomic prediction of
  complex human traits? *Genetics* 2018;210:809–819.
- Bycroft C, Freeman C, Petkova D, et al. The UK Biobank resource with deep phenotyping and
  genomic data. *Nature* 2018;562:203–209.
- Dibaeinia P, German C, Shringarpure S, Auton A, Khan AA. PRSformer: disease prediction from
  million-scale individual genotypes. bioRxiv 2025. doi:10.1101/2025.10.26.684578.
- Gazal S, Finucane HK, Furlotte NA, et al. Linkage disequilibrium-dependent architecture of
  human complex traits shows action of negative selection. *Nat Genet* 2017;49:1421–1427.
- Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces.
  arXiv:2312.00752, 2023.
- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating
  characteristic (ROC) curve. *Radiology* 1982;143:29–36.
- Jamieson K, Talwalkar A. Non-stochastic best arm identification and hyperparameter
  optimization. *AISTATS* 2016.
- Kelemen M, Xu Y, Jiang T, et al. Performance of deep-learning-based approaches to improve
  polygenic scores. *Nat Commun* 2025;16:5122.
- Kish L. *Survey sampling*. New York: Wiley, 1965.
- Lahoti A, et al. Mamba-3: improved sequence modeling using state space principles.
  arXiv:2603.15569, 2026.
- Li L, Jamieson K, Rostamizadeh A, et al. A system for massively parallel hyperparameter
  tuning. *MLSys* 2020.
- Lin DY, Zeng D. Proper analysis of secondary phenotype data in case-control association
  studies. *Genet Epidemiol* 2009;33:256–265.
- Loshchilov I, Hutter F. Decoupled weight decay regularization. *ICLR* 2019.
- Monsees GM, Tamimi RM, Kraft P. Genome-wide association scans for secondary traits using
  case-control samples. *Genet Epidemiol* 2009;33:717–728.
- Privé F, Albiñana C, Arbel J, Pasaniuc B, Vilhjálmsson BJ. Inferring disease architecture and
  predictive ability with LDpred2-auto. *Am J Hum Genet* 2023;110:2042–2055.
- Sigurdsson AI, Louloudis I, Banasik K, et al. Deep integrative models for large-scale human genomics.
  *Nucleic Acids Research* 2023;51:e67.
- Sudlow C, Gallacher J, Allen N, et al. UK Biobank: an open access resource for identifying the
  causes of a wide range of complex diseases of middle and old age. *PLoS Med* 2015;12:e1001779.
- van Hilten A, Kushner SA, Kayser M, et al. GenNet framework: interpretable deep learning for predicting
  phenotypes from genetic data. *Communications Biology* 2021;4:1094.
- Yang Y, Guo K, Zhang Y, et al. MetaGeno: a chromosome-wise multi-task genomic framework for
  ischaemic stroke risk prediction. *Brief Bioinform* 2025;26:bbaf348.
- Zheng Z, Liu S, Sidorenko J, et al. Leveraging functional genomic annotations and genome
  coverage to improve polygenic prediction of complex traits within and between ancestries.
  *Nat Genet* 2024;56:767–777.
- Zhu L, Liao B, Zhang Q, et al. Vision Mamba: efficient visual representation learning with
  bidirectional state space model. *ICML* 2024.
- Zhu T, Ghose U, Climente-González H, Howson J, Hu S, Nevado-Holgado A. Multimodal deep
  learning enhances genomic risk prediction for cardiometabolic diseases. *Brief Bioinform*
  2026;27:bbag550.
