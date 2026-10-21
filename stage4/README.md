# Stage 4: Accessibility-Based Reranking

**Question:** can accessibility select better OpenFold predictions, or is improvement limited by the candidates available?

## Method

Reuse the same 25 candidates for each of 85 proteins; no model weights change. Select the highest score `S = z(confidence) - lambda * z(accessibility MAE)`, standardized within each protein. Lambda = 0.1 was chosen on 15 validation proteins using structural-rank change and the one-SE rule. RMSD to the experimental PDB structure evaluates selection; reference accessibility is calculated from that structure, not independently measured.

Compare OpenFold's top choice, combined scoring, accessibility alone, shuffled-accessibility scoring, and the oracle (lowest-RMSD candidate available). Tied winners have their evaluation outcomes averaged.

## Results: Test A/B (40 Proteins)

| Outcome | Definition | Count |
|---|---|---:|
| A-like | Available gain >0.5 angstroms; combined rule recovers at least 50% | 2 |
| B-like | Available gain >0.5 angstroms; less than 50% recovered | 4 |
| C-like, oracle >=0.5 angstroms | Available gain <=0.5 angstroms; best candidate still misses project benchmark | 23 |
| C-like, oracle <0.5 angstroms | Available gain <=0.5 angstroms; best candidate below benchmark, not necessarily correct | 11 |

Available gain = OpenFold RMSD minus oracle RMSD. The 0.5-angstrom gain threshold is a convention, separate from the expert-informed 0.5-angstrom absolute-RMSD benchmark. Smaller improvements can still matter.

The combined rule improves / leaves unchanged / worsens 3 / 35 / 2 proteins (unchanged = within 0.01 angstroms). Mean RMSD change is -0.164 angstroms (95% bootstrap interval -0.468 to +0.002); median change is zero. It beats shuffled scoring (p = 0.0001), but average improvement over OpenFold remains uncertain.

## Conclusion

C-like is the majority at the stated threshold. For 23/40 proteins, limited reranking opportunity coexists with an oracle RMSD >=0.5 angstroms: better candidates are needed to reach the project benchmark. Selection also misses opportunities in four B-like proteins.

Together with Stage 3, this motivates testing accessibility-guided generation; it does not establish that guidance will work or outperform other sampling methods. A/B/C classification was added after test evaluation and is exploratory. Test A/B were previously inspected in Stage 3; no lambda was tuned on them. Counts depend on thresholds: at a 0.1-angstrom available-gain threshold, A/B/C counts are 2/19/19 rather than 2/4/34. Full sensitivity is in [outcome_analysis.csv](results/outcome_analysis.csv).

## Deliverables

[plots/rerank_figure.png](plots/rerank_figure.png): A selects lambda; B/C compare RMSD changes on development/validation and Test A/B (dots = proteins, bars = medians; negative = better); D counts Test A/B outcome categories. B/C use symmetric-log axes.

`results/` holds per-protein results, aggregate statistics, outcome sensitivity and the frozen rule (including the lambda curve). `rerank_candidates.py` implements selection, evaluation and outcome reporting using the Stage 3 `accessfold` package; run its `--help` for arguments. Keep full working outputs and evaluation logs locally; tests, duplicate SVGs and generated supporting READMEs are not committed.
