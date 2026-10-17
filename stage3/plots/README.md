# Stage 3 Figures

**Question:** does lower accessibility error identify structurally better candidates, and is that information lost when residue positions are shuffled?

All figures combine 2,125 candidates from 85 proteins: development (30 proteins, blue), validation (15, orange), Test A (20, green) and Test B (20, purple). Each protein has 25 candidates. No RMSD-range threshold separates the proteins.

## Measures

- **Accessibility MAE:** mean absolute difference between candidate and reference relative accessibility, averaged over four probe radii; lower is better.
- **Accessibility 1 - r:** one minus the Pearson correlation between candidate and reference residue profiles, averaged over probe radii; lower means closer exposure patterns.
- **Structural error:** C-alpha RMSD against the experimental structure, in angstroms; lower is better.
- **Ranking agreement:** within each protein, Spearman rho between accessibility error and RMSD across its candidates. Positive rho supports the hypothesis. This is distinct from the residue-profile Pearson correlation used in `1 - r`.

## Reading the Figures

`accessibility_vs_accuracy/fig1_accessibility_vs_accuracy.png` answers the main question in two panels:

- **Panel A:** within each protein, group candidates from lowest to highest accessibility error into five rank groups. Each dot is that protein/group's mean structural-rank percentile; the boxes summarise proteins. Lower is better (0 = best candidate, 1 = worst); 0.5 is the expected average under unrelated rankings. In this dataset every group contains five candidates; average-rank ties can make group sizes unequal in future datasets.
- **Panel B:** each dot is one protein's Spearman rho, grouped by dataset. For each protein, the underlying candidate-level x is accessibility error and y is C-alpha RMSD. The tick is the median rho across proteins, not their average; the bar is its 95% bootstrap interval. Each protein appears in both the all-protein row and its dataset row.
- The caption's raw effect size is the median, across proteins, of group 5 mean RMSD minus group 1 mean RMSD. Its positive value (+0.0624 angstroms) supports the expected direction but indicates a small typical absolute difference; it is not a top-1 gain over OpenFold.

`controls/` contains two figures: `controls_mae_rmsd_ca_overview.png` and `controls_corr_rmsd_ca_overview.png`. Reference means relative SASA computed from experimental PDB coordinates, not independently measured accessibility. Pearson r compares candidate and reference SASA across residues; `1 - r` is an accessibility error, not a correlation with RMSD.

- **Panel A, ranking agreement:** one rho per protein, correlating accessibility error (MAE or `1 - r`) with RMSD across its 25 candidates; dots show the median of these rhos. Positive is the expected direction. Horizontal lines are 95% bootstrap intervals over proteins. The three selection-only reference rows are blank here because they do not define an accessibility error across candidates.
- **Panel B, selection:** choose the lowest-accessibility-error candidate for each protein, then average its structural-rank percentile across proteins. Lower is better: 0 = best available candidate, 1 = worst, 0.5 = expected random rank. The supplied ranking selects OpenFold's top-ranked candidate; the oracle selects the lowest actual RMSD using the known PDB structure.
- In both panels the grey band is the 95% shuffle-null range of the aggregate statistic, not a bootstrap interval; the dashed line is its mean. Compare unshuffled SASA against this null. Ranks show ordering, not RMSD differences in angstroms.

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
    stage3/scoring/results/development_validation/candidate_scores.csv \
    stage3/scoring/results/test/candidate_scores.csv \
  --structure-csv \
    stage3/scoring/results/development_validation/candidate_scores.csv \
    stage3/scoring/results/test/candidate_scores.csv \
  --x ac_error --y rmsd_ca --strict --out <figures_out>

python stage3/plots/plot_controls.py \
  --controls-dir <controls_out> --error-metric mae corr \
  --metric rmsd_ca --out <controls_figures_out>
```

Controls plotting reads the locally retained output of [run_controls.py](../controls/README.md), including its shuffle null. Only PNG deliverables and production scripts are stored here; generated SVGs, diagnostic tables, settings, logs and test scripts are omitted. Future datasets should regenerate and replace the figures rather than add duplicate versions.

Only three summary PNGs are retained. Optional `--raw-pages` (accessibility script) and `--per-protein` (controls script) generate detailed inspection plots locally; they are not part of the committed deliverables.
