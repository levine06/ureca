# Stage 3: The information-content experiment

Stage 3 asks: **among OpenFold3 predictions for the same protein, do structurally more accurate candidates usually agree better with the true synthetic accessibility?** This checks whether accessibility contains useful structural information before building a more complicated model. The reference accessibility is computed from the experimental structure, not measured dye data. For a protein with true structure `X_true`:

1. Calculate the true synthetic accessibility: `y = f(X_true)`.
2. Generate candidate OpenFold3 structures: `X_1, ..., X_K`.
3. Calculate the predicted accessibility of each candidate: `ŷ^(k) = f(X_k)`.
4. Calculate an accessibility error for each candidate.
5. Calculate the true structural accuracy of each candidate.

The accessibility error is then plotted against structural error (C-alpha RMSD) for every candidate; lower values are better on both axes. Comparisons are made within each protein so differences between easy and difficult proteins do not drive the result. Controls repeat the comparison after shuffling reference accessibility values between residues, testing whether the relationship depends on their correct positions.

## Components

| Step | What | Where |
|---|---|---|
| 1 | True synthetic accessibility | [`truth_accessibility/`](truth_accessibility/README.md) |
| 2 | Candidate generation | [`candidate_generation/`](candidate_generation/README.md) |
| 3-5 | Candidate accessibility, accessibility error, structural accuracy | [`scoring/`](scoring/README.md) (result tables in [`scoring/results/`](scoring/README.md#results)) |
| 6 | Shuffling and degraded-information controls | [`controls/`](controls/README.md) |
| 7 | Combined figures and interpretation | [`plots/`](plots/README.md) |

## Current Answer

**Yes, generally, although the relationship is moderate and not universal.** Across development, validation, Test A and Test B (85 proteins, 25 candidates each), the median within-protein correlation between accessibility MAE and RMSD is +0.307 (95% bootstrap interval +0.262 to +0.410); 67 of 85 proteins show a positive correlation. Thus, candidates agreeing better with synthetic accessibility tend to be structurally more accurate.

After shuffling, the median correlation is approximately zero, and the real reference outperforms the shuffle null. The correlation-based accessibility error gives the same overall conclusion. This supports the hypothesis that correctly positioned accessibility values carry useful structural information.

**Decision:** proceed to testing accessibility-guided modelling on the strength of this best-case information-content check. This is not yet evidence that diffusion guidance or noisy experimental accessibility improves predictions. See the [figure guide](plots/README.md) for the panel explanations and supporting selection results.
