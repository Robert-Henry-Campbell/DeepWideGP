read all of the md files in /agent_notes/agent_notes.md to aquaint yourself with context, and read the README.md file. Then, as you write code or make changes in this session, write a summary of the changes you made to the codebase and enviornment into agent_notes/agent_notes.md. Only write text in cases where you have changed some code. Do not overwrite the existing text in agent_notes.md; add to them using dates/timestamps. 

## Write-ups and figures

- **Every analysis gets a `methods.md` and a `results.md`** in its own folder, kept current. Each
  describes the analysis as it now stands, not its history.
- **Both are short and scientific.** They hold only what a scientist needs to understand and judge
  the answer to the analysis's question, at the length of a paper's Methods or Results subsection.
  They leave out the implementation: code, file paths, flags, functions, software set-up, compute
  and run logistics, which belong in the code and the commit messages. Name software and versions
  only as a paper's Methods would.
  - `methods.md`: the question; the data (source, sample size, who was included or excluded); the
    design (what is compared with what, and the outcome measures); the statistical methods, with
    any setting that changes the result; and each decision, its reason and what it changes.
  - `results.md`: the answer to the question in one or two sentences; the findings behind it, in
    aggregate tables of the estimates with 95% confidence intervals, sample sizes and, where a
    test was run, P values; each figure by name, with a line on what it shows; and what the
    results do not show.
- **Every figure is three files with one name, side by side in the analysis folder:**
  `<figure>.png` (the figure), `<figure>.data.csv` (the values plotted) and `<figure>.legend.txt`
  (the legend). For example: `auroc_by_size.png`, `auroc_by_size.data.csv`,
  `auroc_by_size.legend.txt`. A figure is not done until all three exist.
- **Figures are Nature size and style.** 89 mm wide (single column) or 183 mm wide (double column),
  at most 170 mm tall, drawn at that size and never scaled afterwards. Text 7 pt, Arial or
  Helvetica; panel letters 8 pt bold lower case (a, b, c). No title on the figure; the legend
  carries it. PNG at 300 dpi or more.
- **The CSV holds every value plotted, enough to redraw each panel.** Write it from the same data in
  the same step as the PNG, so the two always match: one row per plotted value, a `panel` column
  when there are several panels, and column names that give the quantity and its unit. It holds
  only what may leave the project under its data rules (for UK Biobank: aggregates only, counts of
  1 to 9 written `<10`).
- **Each legend is Nature style and under 250 words;** count the words before saving.
  - It opens with a short title sentence for the whole figure, then describes each panel in order
    (a, b, c), saying in words what each colour, symbol and line shows.
  - It defines every error bar and statistic: the centre value, the interval (such as a 95% CI),
    n for each group and what n counts, and the test behind each P value. It defines unusual units
    and abbreviations.
  - It gives no methods detail; that goes in `methods.md`.
