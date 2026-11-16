"""Draw the WideDeepGP schematic used in methods.md.

    python scripts/make_architecture_figure.py        # writes figures/widedeepgp_architecture.{pdf,png}
                                                      # plus _data.csv (the components) and _legend.md

183 mm wide, 7 pt text. Components that are new or changed relative to single-phenotype
DeepGP are tinted, outlined more heavily and tagged, so the distinction does not rest on
colour alone. Parameter counts in the table are counted from a constructed network.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures")
NAME = "widedeepgp_architecture"
MM = 1 / 25.4
W, H = 183.0, 117.0                                  # figure size in mm
INK, INK2, MUTED, EDGE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, BLUE_TINT, PLAIN = "#2a78d6", "#cde2fb", "#f6f5f1"
Y_CHR = 91                                           # bottom edge of the per-chromosome row (mm)

# name, x, y, width, height (mm), status, title, lines, copies drawn behind (one per chromosome)
BOXES = [
    ("genotypes", 4, Y_CHR, 26, 16, "changed", "Genotypes", ["one chromosome,", "standardised dosage"], 2),
    ("embedding", 35.5, Y_CHR, 26, 16, "same", "Token embedding", ["convolution over", "3 SNPs + position"], 2),
    ("encoder", 98.5, Y_CHR, 26, 16, "same", "Sequence encoder", ["bidirectional Mamba,", "one or more layers"], 2),
    ("pool", 130.5, Y_CHR, 21, 16, "same", "Mean pool", ["over the SNPs", "(32-bit)"], 2),
    # one pair of maps serves every chromosome, so the box is drawn once, below the per-chromosome row
    ("annotation", 49, 72, 62, 16, "new", "Annotation gain and offset",
     ["from 96 functional annotations and a no-annotation", "flag (97 channels per SNP); one pair of maps,",
      "shared by all chromosomes (optional)"], 0),
    ("covariates", 4, 54, 26, 16, "changed", "Covariates", ["year of birth, sex,", "10 genetic PCs"], 0),
    ("cov_encoder", 130.5, 54, 21, 16, "same", "Covariate MLP", ["two layers"], 0),
    ("attention", 157.5, 54, 22.5, Y_CHR + 16 - 54, "same", "Attention",
     ["across the 23", "vectors: 22", "chromosomes", "and the", "covariates"], 0),
    # beside the network: one weight per SNP and disease, reading every SNP of every chromosome
    ("prs", 35.5, 32, 89, 16, "new", "Linear PRS path",
     ["one weight per SNP and disease (1,258,200 × 30), ridge penalty;",
      "one per covariate and disease; 32-bit; 30 logits added to the head's",
      "(off: the network alone; only: the path alone, a linear model)"], 0),
    ("batches", 4, 8, 57.5, 18, "new", "Batches: a rotation over the diseases",
     ["each batch is half cases and half non-cases of", "one target disease; every disease is the target", "in turn"], 0),
    ("loss", 67, 8, 57.5, 18, "new", "Loss: all 30 outputs learn from every batch",
     ["cross-entropy divided by each person's sampling", "rate, weighted by class, averaged per disease", "and then across diseases"], 0),
    ("head", 130.5, 8, 49.5, 18, "changed", "Output head", ["shared hidden layer (256 units),", "then one logit per disease (30)"], 0),
]

# what the figure shows, as a table (d_model 32, one layer, 30 outputs, 12 covariates)
COMPONENTS = [
    # component, input, output, trainable parameters, relative to single-phenotype DeepGP
    ("Genotypes", "22 chromosomes, 1,258,200 SNPs", "[B, L_c, 1] per chromosome", 0,
     "changed: one fixed SNP set (HapMap3+) for all diseases instead of SNPs chosen per disease"),
    ("Token embedding", "[B, L_c, 1]", "[B, L_c, d]", 2112, "same (96 parameters per chromosome)"),
    ("Annotation gain and offset",
     "[L_c, 97] per chromosome: 96 functional annotations and a no-annotation flag per SNP",
     "scales and shifts [B, L_c, d]", 6272,
     "new, optional; one pair of maps shared by all chromosomes; not in the total below"),
    ("Sequence encoder", "[B, L_c, d]", "[B, L_c, d]", 242880,
     "same (11,008 per chromosome and layer, plus a final normalisation of 32)"),
    ("Mean pool", "[B, L_c, d]", "[B, d]", 0, "same; taken in 32-bit arithmetic under mixed precision"),
    ("Covariates", "year of birth, sex, 10 principal components", "[B, 12]", 0, "changed: 12 covariates instead of 8"),
    ("Covariate MLP", "[B, 12]", "[B, d]", 11584, "same architecture"),
    ("Attention across vectors", "[B, 23, d]", "[B, 23, d]", 12704, "same"),
    ("Output head", "[B, 23 d]", "[B, 30] logits", 196382,
     "changed: 30 outputs instead of 1 (257 parameters per disease); logits instead of probabilities"),
    ("Linear PRS path", "[B, 1,258,200] standardised dosages and [B, 12] covariates",
     "[B, 30] logits, added to the head's", 37746390,
     "new: 1,258,200 x 30 SNP weights with a ridge penalty, 12 x 30 covariate weights and 30 intercepts "
     "(--wide on; off: the network alone; only: the path alone)"),
    ("Batches", "training split", "B participants, a target disease", 0, "new: rotation of disease-balanced batches"),
    ("Loss", "[B, 30] logits, [B, 30] labels", "scalar", 0,
     "new: sampling-rate correction, class weights, per-disease normalisation"),
    ("Total network (without annotations)", "", "", 465662, ""),
    ("Total with the linear PRS path", "", "", 38212052, ""),
]

LEGEND = """**Figure 1. WideDeepGP.** Each of the 22 chromosomes is a sequence of standardised SNP dosages
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
shared by all chromosomes, so that box is drawn once, outside the stack. Beside the network, the
linear PRS path gives every SNP and disease a weight of its own, with a ridge penalty, and every
covariate and disease a weight; its 30 logits are added to the head's, so the network only has
to add what a genome-wide PRS and the covariates miss (`--wide on`, the default; `off` is the
network alone, `only` the path alone). With a model width of 32 and one encoder layer the
network has 465,662 trainable parameters and the path 37,746,390 (table:
`widedeepgp_architecture_data.csv`).
"""


def box(ax, x, y, w, h, status, title, lines, copies):
    changed = status != "same"
    face, edge, lw = (BLUE_TINT, BLUE, 1.1) if changed else (PLAIN, EDGE, 0.7)
    for k in range(copies, 0, -1):                   # outlines behind: one encoder per chromosome
        ax.add_patch(FancyBboxPatch((x + 0.9 * k, y + 0.9 * k), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc="white", ec=edge, lw=0.6, zorder=2 - 0.1 * k))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=face, ec=edge, lw=lw, zorder=3))
    top = y + h - 2.1
    ax.text(x + 1.6, top, title, fontweight="bold", color=INK, va="top", ha="left", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 1.6, top - 3.2 * (i + 1), line, color=INK2, va="top", ha="left", zorder=4)
    if changed:
        ax.text(x + w - 1.4, y + 1.3, status, fontstyle="italic", color=INK2, va="bottom", ha="right", zorder=4)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2, shrinkA=0, shrinkB=0,
                                 zorder=5))


def line(ax, pts):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=INK2, lw=0.8, solid_capstyle="round", zorder=5)


def draw():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "savefig.dpi": 450, "figure.dpi": 150})
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    b = {name: (x, y, w, h) for name, x, y, w, h, *_ in BOXES}
    for _, *spec in BOXES:
        box(ax, *spec)

    mid = lambda n: b[n][1] + b[n][3] / 2
    right = lambda n: b[n][0] + b[n][2]
    # the chromosome branch, left to right
    for a_, c_ in (("genotypes", "embedding"), ("embedding", "encoder"), ("encoder", "pool")):
        arrow(ax, (right(a_) + 1.9, mid(a_)), (b[c_][0], mid(c_)))
    arrow(ax, (right("pool") + 1.9, mid("pool")), (b["attention"][0], mid("pool")))
    # the shared annotation gain and offset act on the tokens between embedding and encoder
    x_ann = b["annotation"][0] + b["annotation"][2] / 2
    arrow(ax, (x_ann, b["annotation"][1] + b["annotation"][3]), (x_ann, mid("embedding")))
    # the covariate branch
    arrow(ax, (right("covariates"), mid("covariates")), (b["cov_encoder"][0], mid("cov_encoder")))
    arrow(ax, (right("cov_encoder"), mid("cov_encoder")), (b["attention"][0], mid("cov_encoder")))
    # attention -> head -> loss <- batches
    xa = b["attention"][0] + b["attention"][2] / 2
    arrow(ax, (xa, b["attention"][1]), (xa, b["head"][1] + b["head"][3]))
    arrow(ax, (b["head"][0], mid("head")), (right("loss"), mid("loss")))
    arrow(ax, (right("batches"), mid("batches")), (b["loss"][0], mid("loss")))
    # the batch supplies genotypes and covariates: a rail up the left margin
    x_rail = 1.6
    line(ax, [(b["batches"][0], mid("batches")), (x_rail, mid("batches")), (x_rail, mid("genotypes"))])
    arrow(ax, (x_rail, mid("covariates")), (b["covariates"][0], mid("covariates")))
    arrow(ax, (x_rail, mid("genotypes")), (b["genotypes"][0], mid("genotypes")))
    # the linear PRS path reads every SNP and the covariates from the batch; its logits join the head's
    arrow(ax, (x_rail, mid("prs")), (b["prs"][0], mid("prs")))
    ax.text((x_rail + b["prs"][0]) / 2 + 0.8, mid("prs") + 0.9, "all SNPs,", color=MUTED, ha="center", va="bottom")
    ax.text((x_rail + b["prs"][0]) / 2 + 0.8, mid("prs") - 0.9, "covariates", color=MUTED, ha="center", va="top")
    x_join = b["head"][0] + 9
    line(ax, [(right("prs"), mid("prs")), (x_join, mid("prs"))])
    arrow(ax, (x_join, mid("prs")), (x_join, b["head"][1] + b["head"][3]))
    ax.text(x_join + 1.2, (mid("prs") + b["head"][1] + b["head"][3]) / 2, "+ logits", color=MUTED,
            ha="left", va="center")

    # labels on the long arrows
    ax.text((right("batches") + b["loss"][0]) / 2, mid("batches") + 1.2, "labels", color=MUTED, ha="center",
            va="bottom", rotation=90)
    ax.text((right("loss") + b["head"][0]) / 2, mid("head") + 1.2, "logits", color=MUTED, ha="center",
            va="bottom", rotation=90)
    # the bracket over the per-chromosome parts
    x0, x1, yb = b["genotypes"][0], right("pool") + 1.8, Y_CHR + 16 + 4.6
    line(ax, [(x0, yb - 1.2), (x0, yb), (x1, yb), (x1, yb - 1.2)])
    ax.text((x0 + x1) / 2, yb + 0.8, "22 chromosomes, each with its own embedding and encoder (weights not shared)",
            color=INK, ha="center", va="bottom")
    # key
    for x, (face, edge, lw, label) in zip((4, 60), ((PLAIN, EDGE, 0.7, "as in single-phenotype DeepGP"),
                                                    (BLUE_TINT, BLUE, 1.1, "new or changed for many phenotypes"))):
        ax.add_patch(FancyBboxPatch((x, 1.6), 5, 3.2, boxstyle="round,pad=0,rounding_size=0.6", fc=face, ec=edge, lw=lw))
        ax.text(x + 6.5, 3.2, label, color=INK2, va="center", ha="left")
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = draw()
    fig.savefig(os.path.join(OUT, NAME + ".png"))
    fig.savefig(os.path.join(OUT, NAME + ".pdf"))
    plt.close(fig)
    with open(os.path.join(OUT, NAME + "_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "input", "output", "trainable_parameters_d32_1layer", "relative_to_single_phenotype_deepgp"])
        w.writerows(COMPONENTS)
    with open(os.path.join(OUT, NAME + "_legend.md"), "w") as f:
        f.write(LEGEND)
    print("wrote", os.path.join(OUT, NAME + ".png"))


if __name__ == "__main__":
    main()
