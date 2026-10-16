# Stage 3 Figures

**Question:** does lower accessibility error identify structurally better candidates, and is that information lost when residue positions are shuffled?

All figures combine 2,125 candidates from 85 proteins: development (30 proteins, blue), validation (15, orange), Test A (20, green) and Test B (20, purple). Each protein has 25 candidates. No RMSD-range threshold separates the proteins.

## Measures

- **Accessibility MAE:** mean absolute difference between candidate and reference relative accessibility, averaged over four probe radii; lower is better.
- **Accessibility 1 - r:** one minus the Pearson correlation between candidate and reference residue profiles, averaged over probe radii; lower means closer exposure patterns.
- **Structural error:** C-alpha RMSD against the experimental structure, in angstroms; lower is better.
- **Ranking agreement:** within each protein, Spearman rho between accessibility error and RMSD across its candidates. Positive rho supports the hypothesis. This is distinct from the residue-profile Pearson correlation used in `1 - r`.

## Reading the Figures

`accessibility_vs_accuracy/` contains the overview and five pages of raw per-protein scatter plots. In overview panel A, both axes are ranks within each protein (0 = best, 1 = worst); the solid line averages structural ranks in accessibility-rank bins. The dotted diagonal represents identical rankings, not a fitted trend. Overlapping dots can hide candidates. Panel B shows one rho per protein; brackets give its highest minus lowest candidate RMSD. Raw plots have independent axis limits: compare their numbers, not just visual slopes.

`controls/` contains an overview and per-protein selection figure for each accessibility error:

- **Overview A:** coloured dots are real rho; grey bars span 95% of shuffled rho values and ticks mark their means. Rows are sorted by real minus mean shuffled rho (delta rho), largest advantage first. A dot right of its tick indicates better ranking agreement than the average shuffle, not necessarily a statistically significant result for that protein.
- **Overview B:** dots show median rho across proteins; horizontal lines are 95% bootstrap intervals over proteins. The grey band is the shuffled null range of that summary, not a bootstrap interval. Compare the real result with this band, not only with zero.
- **Overview C:** mean structural-rank percentile of the candidate selected by minimum accessibility error. Lower is better: 0 = best, 1 = worst; random selection averages 0.5 with untied structural scores. Compare the real selection with the shuffle band and the supplied OpenFold ranking.
- **Per-protein selection:** coloured dot = accessibility-selected candidate; cross = OpenFold top-ranked candidate; grey bar/tick = shuffled-selection range/mean. Brackets show selected RMSD minus that protein's mean candidate RMSD; negative is better than average. Rank percentiles do not show the absolute size of RMSD gains.

Shuffling/reversal/wrong-protein profiles disrupt positional information. Keeping fewer residues tests sparsity. Buried/exposed controls test whether coarse labels retain useful information instead of needing precise accessibility values. All aggregate summaries remain pooled across datasets; colours alone do not establish a separate test-set result.

## Results and Interpretation

| Across all 85 proteins | MAE | 1 - r |
|---|---:|---:|
| Real median rho | +0.31 | +0.29 |
| Shuffle median rho, approximately | 0.00 | 0.00 |
| Real mean selection percentile | 0.35 | 0.39 |
| Shuffle mean selection percentile | 0.51 | 0.51 |
| Win rate versus OpenFold top-ranked, ties count half | 68% | 59% |

Real-versus-shuffle comparisons give p = 0.0001 for ranking agreement with either error, and p = 0.0001 / 0.0003 for MAE / 1 - r selection, as reported by the run. These are aggregate shuffle comparisons, not tests of superiority over OpenFold. The MAE scatter analysis has median rho +0.307 (95% bootstrap interval +0.262 to +0.410), with positive rho in 67 of 85 proteins.

**Answer:** the results support position-specific accessibility as a useful candidate-ranking signal, with exceptions and modest aggregate effects. Reference accessibility is calculated from the experimental structure: this is a best-case information-content check, not proof that measured accessibility or accessibility-guided diffusion will improve predictions.

## Regenerate

From the repository root, install `numpy`, `scipy`, `matplotlib` and the local package (`pip install -e stage3/truth_accessibility`). Source candidate tables are already in `stage3/scoring/results/`; no rescoring is needed for the initial figures.

```bash
python stage3/plots/plot_accessibility_vs_accuracy.py \
  --accessibility-csv \
    stage3/scoring/results/development_validation/accessibility_candidate_scores.csv \
    stage3/scoring/results/test/candidate_scores.csv \
  --structure-csv \
    stage3/scoring/results/development_validation/structure_candidate_scores.csv \
    stage3/scoring/results/test/candidate_scores.csv \
  --x ac_error --y rmsd_ca --strict --out <figures_out>

python stage3/plots/plot_controls.py \
  --controls-dir <controls_out> --error-metric mae corr \
  --metric rmsd_ca --out <controls_figures_out>
```

Controls plotting reads the locally retained output of [run_controls.py](../controls/README.md), including its shuffle null. Only PNG deliverables and production scripts are stored here; generated SVGs, diagnostic tables, settings, logs and test scripts are omitted. Future datasets should regenerate and replace the figures rather than add duplicate versions.
