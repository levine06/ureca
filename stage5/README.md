# Stage 5: Differentiable accessibility

A PyTorch estimate of per-residue relative accessibility from C-alpha coordinates, with gradients
with respect to those coordinates. It is the building block for Stage 6 (guided diffusion).
OpenFold3 is not modified.

## Model (`differentiable_accessibility.py`)

$$\rho_i = \sum_{j \ne i} \sigma\!\left(\frac{r_0 - d_{ij}}{\tau}\right) \qquad \text{neighbour density, } (N,3) \to (N,)$$

$$\hat a_i = \sigma\!\left(\frac{b - \rho_i}{s}\right) \qquad \text{accessibility in } (0,1)$$

$$L_{\text{accessibility}} = \frac{1}{N}\sum_{i=1}^{N}\left(\hat a_i - a_i^{\text{ref}}\right)^2$$

where $d_{ij}$ is the C-alpha distance between residues $i$ and $j$ and $\sigma$ is the sigmoid.

$\tau$ is fixed at 1 Å. $r_0$, $b$, $s$ were fitted (Adam on the MSE) on the 30 **development**
proteins only: $r_0 = 8.65$ Å, $b = 9.02$, $s = 3.38$ (`results/calibrated_params.json`).

## Validation (`validate_accessibility.py`)

Reference: Stage 3 relative SASA (probe 1.4 A, `stage3/truth_accessibility/results`), clipped to [0, 1],
on valid (unmasked) residues. All resolved C-alphas contribute neighbours. Buried = relative SASA < 0.2
(the Stage 3 convention). Per-protein metrics are in `results/validation_metrics.csv`; the table gives the
mean over proteins (+/- sd).

| Metric | Development (30) | Validation (15) |
| --- | --- | --- |
| Pearson | 0.838 +/- 0.026 | 0.837 +/- 0.025 |
| Spearman | 0.839 +/- 0.024 | 0.839 +/- 0.031 |
| MAE | 0.099 +/- 0.009 | 0.101 +/- 0.012 |
| Buried/exposed accuracy | 0.830 +/- 0.030 | 0.838 +/- 0.031 |
| Balanced accuracy | 0.822 +/- 0.030 | 0.833 +/- 0.035 |

Pooling all residues of a split into one vector (5879 development and 3103 validation residues, the
"across thousands of residues" evaluation) gives the same picture; validation: Pearson 0.839, Spearman 0.844,
MAE 0.097, accuracy 0.843, balanced accuracy 0.842 (`pooled` rows in `results/validation_metrics.csv`).

Development and validation agree, so the three fitted parameters are not overfit. The research plan sets
no thresholds; the working acceptance criterion is validation Pearson/Spearman > 0.8, MAE < 0.15 and
balanced accuracy > 0.8, all met. The Stage 3 test splits were not used.

Plots: `plots/accessibility_comparison.png`, `plots/buried_exposed_accuracy.png`. How to read each figure: [plots/README.md](plots/README.md).

## Gradient test (`test_gradients.py`)

Checks (asserted): output shape `(N,)`, `coords.grad` shape `(N, 3)` and finite, accessibility unchanged
under a random rotation plus translation (max difference ~3e-6).

Gradient descent (SGD, lr 5, 300 steps) on 1AYD chain A, residue 27 (the buried residue the model scores
lowest): target accessibility 0.65, estimated accessibility rises **0.032 -> 0.650**. The residue moves
~8 A, moving outward: its neighbours within $r_0$ drop from 16 to 5 and its distance from the chain centroid
grows from 6.1 to 12.5 A (the PCA arrow in the figure looks short because most of the move is out of that
2D plane). Neighbours shift by up to ~1.5 A (RMSD 0.95 A over the chain). The result is not meant to be a
realistic structure (no chain connectivity or clash terms), only a demonstration that the gradient works.
Outputs: `results/gradient_test.csv`, `plots/gradient_test.png` (see [plots/README.md](plots/README.md)).

## Candidate and noise tests (`test_candidates_noise.py`)

Question: is the estimate still useful on OpenFold candidates and on noisy coordinates, as in diffusion? Uses the
45 development and validation proteins (1,125 candidates from Stage 3); the held-out test splits are not read.
Outputs: `results/candidate_test.csv` (per protein), `results/noise_test.csv`, `plots/candidate_noise_test.png`.

**Candidate test.** For each protein, the Spearman rho between a candidate's accessibility error (MAE against the
reference relative SASA, probe 1.4 A) and its C-alpha RMSD over its 25 candidates; positive means lower accessibility
error goes with lower RMSD. Median over proteins, 95% bootstrap interval over proteins:

| Accessibility error | Median rho [95% CI] | Proteins with rho > 0 |
| --- | --- | --- |
| `accessfold` against reference (Stage 3 signal) | +0.28 [+0.15, +0.36] | 35/45 |
| Differentiable estimate against reference | +0.25 [+0.13, +0.33] | 34/45 |
| Differentiable estimate against its own native estimate | +0.43 [+0.29, +0.54] | 41/45 |
| Differentiable, shuffled reference (control) | +0.02 [-0.06, +0.08] | 26/45 |

On the same candidates, the estimate agrees with `accessfold` (mean Pearson 0.837, as on native structures).
Selecting the candidate with the lowest error gives a median structural rank percentile (0 = best, 0.5 = chance) of
0.54 for OpenFold's top-ranked candidate, 0.17 (mean 0.35) for `accessfold` and 0.42 (mean 0.43) for the differentiable
estimate. The differentiable selector's gain over OpenFold (mean -0.08, 95% CI [-0.21, +0.04]) is not statistically
clear, unlike `accessfold`'s (mean -0.16, [-0.28, -0.03]).

**Noise test.** Independent Gaussian noise is added to the native C-alphas (as in diffusion; 5 draws per protein).
Equivalent RMSD is sigma times sqrt(3). "Gradient cosine" is the cosine similarity between the loss gradient on the
noisy coordinates and on the clean ones (loss against the reference):

| Equivalent RMSD (A) | Pearson with reference | Pearson with clean estimate | Gradient cosine |
| --- | --- | --- | --- |
| 0 | 0.84 | 1.00 | 1.00 |
| 0.9 | 0.82 | 0.97 | 0.92 |
| 1.7 | 0.77 | 0.91 | 0.74 |
| 3.5 | 0.64 | 0.75 | 0.43 |
| 5.2 | 0.53 | 0.61 | 0.27 |
| 13.9 | 0.24 | 0.27 | 0.06 |

**Conclusion.** The Stage 3 accessibility-structure signal survives in the differentiable estimate, and the shuffled
control is at zero. It is a weaker selector than `accessfold`, though. The gradient direction is reliable within about
2 A of the native structure and mostly lost beyond 3-4 A, so Stage 6 guidance should act late in diffusion or on the
denoised structure estimate, not on highly noisy samples.

Caveats: candidates are close to the native structure (median RMSD range 0.67-1.00 A per protein), so the test says
little about badly wrong structures; there are only 25 candidates per protein; candidates contain every sequence
position, while the fit used resolved residues only.

## Limitations

- C-alpha neighbour density ignores side-chain orientation, so it cannot reproduce exact SASA; it gives
  roughly 0.84 correlation and a mean absolute error of about 0.10 in relative SASA.
- It is calibrated for probe radius 1.4 A only, and for compact single chains: larger probes (the Stage 2
  sensitivity radii) would need a separate fit.
- Unresolved residues are absent from the coordinates and do not contribute neighbours.
- The estimate degrades with coordinate noise (see the noise test), so it needs low-noise or denoised coordinates.
- The optimiser moves atoms freely, so structures after descent are unphysical.
- If this proves too coarse for Stage 6, the plan's optional step is a small neural network.

## Run

Needs torch, numpy, scipy, pandas, matplotlib, scikit-learn; run from this folder. `test_candidates_noise.py` also needs the Stage 3 candidates and scoring runs (`stage3/scoring/runs/dev_val`).

```bash
python validate_accessibility.py && python test_gradients.py && python test_candidates_noise.py
```
