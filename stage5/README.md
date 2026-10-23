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

Plots: `plots/accessibility_comparison.png`, `plots/buried_exposed_accuracy.png`.

## Gradient test (`test_gradients.py`)

Checks (asserted): output shape `(N,)`, `coords.grad` shape `(N, 3)` and finite, accessibility unchanged
under a random rotation plus translation (max difference ~3e-6).

Gradient descent (SGD, lr 5, 300 steps) on 1AYD chain A, residue 27 (the buried residue the model scores
lowest): target accessibility 0.65, estimated accessibility rises **0.032 -> 0.650**. The residue moves
~8 A, moving outward: its neighbours within $r_0$ drop from 16 to 5 and its distance from the chain centroid
grows from 6.1 to 12.5 A (the PCA arrow in the figure looks short because most of the move is out of that
2D plane). Neighbours shift by up to ~1.5 A (RMSD 0.95 A over the chain). The result is not meant to be a
realistic structure (no chain connectivity or clash terms), only a demonstration that the gradient works.
Outputs: `results/gradient_test.csv`, `plots/gradient_test.png`.

## Limitations

- C-alpha neighbour density ignores side-chain orientation, so it cannot reproduce exact SASA; it gives
  roughly 0.84 correlation and a mean absolute error of about 0.10 in relative SASA.
- It is calibrated for probe radius 1.4 A only, and for compact single chains: larger probes (the Stage 2
  sensitivity radii) would need a separate fit.
- Unresolved residues are absent from the coordinates and do not contribute neighbours.
- The optimiser moves atoms freely, so structures after descent are unphysical.
- If this proves too coarse for Stage 6, the plan's optional step is a small neural network.

## Run

Needs torch, numpy, scipy, pandas, matplotlib, scikit-learn; run from this folder.

```bash
python validate_accessibility.py && python test_gradients.py
```
