# Stage 6: Guided OpenFold3 diffusion

Built in steps, each checked before the next. OpenFold3 is not modified.

## Step 1: log the accessibility loss during diffusion (no change to structures)

`log_access_loss.py` attaches a forward hook to OpenFold3's `DiffusionModule`. At every diffusion step and for every sample it takes the C-alpha atoms of the noisy input $x_t$ and of the denoised estimate $\hat x_0$ (the module's output), and computes the Stage 5 loss

$$L_{\text{access}} = \frac{1}{N}\sum_{i=1}^{N}(\hat y_i - y_i)^2$$

with $\hat y$ from `stage5/differentiable_accessibility.py` and $y$ the Stage 3 reference relative SASA (probe 1.4 Å, clipped to [0, 1]; scored on usable residues, resolved C-alphas as neighbours, as in Stage 5). The hook returns nothing, so the output is unchanged and the sampling loop is not copied or edited. Residue mapping: each residue is one token with one atom named `CA`.

```bash
python stage6/log_access_loss.py --entry 1L66_A --seeds 42 --num-diffusion-samples 5 --out stage6/runs/step1   # openfold3 env, CPU
```

Output: `results/access_loss_1L66_A.csv` (seed, step, sigma, sample, `loss_x0`, `loss_xt`).

### Check on 1L66_A (seed 42, 5 samples, 200 steps, `cpu_inference.yml`)

- 1,000 loss values (200 steps × 5 samples), all finite; C-alpha count equals token count equals reference length (asserted).
- **Structures unchanged:** the same run with and without the hook (`--no-hook`) gives identical C-alpha coordinates (max difference 0.0 Å for all 5 samples).
- Loss of the denoised estimate $\hat x_0$: 0.0155 at step 0, 0.0153 at step 50, 0.0159 at the last step. It is already at the level of the experimental structure from the first step.
- Loss of the noisy input $x_t$: 0.485 until about step 100 (sigma > 100 Å), 0.10 at step 150 (sigma 4.3), 0.016 at step 170 (sigma 0.4). The accessibility estimate is only meaningful on $x_t$ late, and from the start on $\hat x_0$.
- The structures are **not** identical to the Stage 3 candidates with the same seed and sample index (C-alpha differences up to ~1.1 Å), so a rerun is not a bit-exact copy of Stage 3. A guided run therefore cannot be treated as noise-paired with the Stage 3 candidates.

## Step 2: gradient of the loss with respect to the denoised estimate (still no change to structures)

`log_access_loss.py --gradient` keeps the Step 1 logging at every step. In the final 30% of the 200 steps (steps 140-199) it also differentiates $L_{\text{access}}$ with respect to the C-alpha coordinates of $\hat x_0$ for each sample, tests the gradient and records the result in `results/gradient_check_1L66_A.csv`. Nothing is returned to OpenFold3, so predictions are untouched. Guidance itself will be applied only in this window in Step 3; earlier steps are logged for diagnosis only.

```bash
python stage6/log_access_loss.py --entry 1L66_A --seeds 42 --num-diffusion-samples 5 --gradient --out stage6/runs/step2
```

### Is the hook compatible with autograd? (`results/hook_info.json`)

Not directly. Lightning predicts under `torch.inference_mode()`: inside the hook `inference_mode` is on, `grad_mode` is off, and the denoised output is an inference tensor with `requires_grad=False` and no `grad_fn`. So OpenFold3's $\hat x_0$ is detached. That is fine, because only the accessibility function needs differentiating, not the network. The hook copies the C-alpha of $\hat x_0$ inside `torch.inference_mode(False)` and `torch.enable_grad()`, makes a fresh leaf tensor and differentiates the Stage 5 loss with respect to it. The original tensor is not altered.

### Mapping checks (1L66_A: 164 residues, 1,304 atoms)

- Each residue is one token with one atom named `CA`. The C-alpha token indices are 0 to 163 in order, and the residue types of the tokens match the Stage 1 sequence (asserted).
- The reference vector has length 164, equal to the sequence length. 162 residues resolved in the experiment are neighbours and are scored; 2 are unresolved and excluded (Stage 5 convention).

### Gradient tests (300 cases: 5 samples x 60 steps)

- Shape `(164, 3)` and finite in all 300.
- Residues that are not neighbours have an exactly zero gradient (max 0.0), so the gradient reaches only residues that enter the loss.
- Gradient RMS is 1.6-1.7e-4 per residue (loss per Angstrom), steady over the window.
- **Direction.** Moving every C-alpha against the gradient by a small RMS step lowers the loss in 300/300 cases at 0.01, 0.03 and 0.1 A. The median change matches the first-order prediction $g\cdot\delta$ (actual/predicted 0.99, 0.98, 0.94). A random move of the same size lowers the loss in only 51%, 49% and 42% of cases, and the changes are tiny or of mixed sign (mean about 1e-7 to 2e-5).
- A 0.1 A RMS move lowers the loss by about 2.5e-3, around 16% of its value (about 0.016). The loss responds strongly to small coordinate changes, so guidance steps must be small.
- **Predictions preserved.** The run with the gradient enabled and a control run with no hook give identical C-alpha coordinates for all 5 samples (max difference 0.0 A).

Caveat: one protein, one seed. This shows the gradient is correct for the Stage 5 surrogate, not that following it improves the real structure.

## Next steps

3. Apply the gradient to the denoised estimate inside the update, in the final 30% of steps only.
4. Guidance schedule and safeguards.
**Question:** does accessibility guidance improve generation beyond normal OpenFold and accessibility reranking?

## A/B Baselines (Reused Stage 4 Results)

| Condition | Files | Selection |
|---|---|---|
| A: normal OpenFold | `A_normal_openfold/candidate_rankings.csv`, `selected_candidates.csv` | Highest original `rank_score` |
| B: normal + reranking | `B_normal_reranked/candidate_rankings.csv`, `selection_comparison.csv`, `comparison_summary.csv` | Highest `z(confidence) - 0.1 * z(accessibility MAE)` |

There are 85 proteins, 25 original candidates each. A's table includes all original confidence scores/ranks and winner flags. `is_rank_1` identifies the supplied rank-1 candidate; `is_A_winner` includes all candidates tied within the evaluation tolerance. B records all combined scores, ranks and winner flags. Both point to the SAME original CIFs under `A_normal_openfold/structures/<split>/<protein>/seed_*/` (local, git-ignored). No structures are regenerated or modified for B.

Selection tables list tied winner IDs; RMSD outcomes are their averages, not a new structure. RMSD is evaluation-only. `B_minus_A < 0` means reranking improves selection; oracle RMSD is the best of the 25. Tables were checked against the saved Stage 4 outcomes for all 85 proteins.

[Comparison figure](B_normal_reranked/plots/stage6_baseline.png): one dot per protein, black bar = median with bootstrap interval. Negative changes favour B; the vertical axis is symmetric-log. Test A/B results: 3 better, 35 unchanged (within 0.01 angstroms), 2 worse; median change 0. The mean is -0.164 angstroms with a 95% interval spanning zero, so improvement is not established for the typical protein or conclusively for the average.

Lambda was selected on validation only, not Test A/B; these are reused baseline results, not new guidance experiments. Reference accessibility is PDB-derived synthetic SASA. Test A/B were inspected in earlier stages. See the [frozen rule](../stage4/results/frozen_rule.json) and [Stage 4](../stage4/README.md) for the source protocol.

## Remaining Conditions

Local `runs/` placeholders are labelled A-E: **C** generates guided candidates; **D** reranks those same C candidates; **E** generates samples with shuffled accessibility guidance. C-E have not been run. Dataset splits (`development`, `validation`, `test_a`, `test_b`) are separate from experimental condition letters.

The checks above found that reruns differ from the original Stage 3 candidates even with the same seed. These reused A/B results are therefore baseline reference evidence, not noise-paired controls for new guided runs. Generate matched unguided controls alongside guided runs for that comparison, matching code revision, checkpoint, inputs/features, templates, seeds, candidate counts and sampling settings. A/B share generation costs, but accessibility scoring adds overhead. Equal candidate counts alone do not establish equal runtime budgets.

## Replot

With the Stage 3 `accessfold` package, NumPy, SciPy and Matplotlib available, run from the repository root:

```bash
python stage6/plot_stage6_baseline.py \
  --results stage4/results/rerank_per_protein.csv \
  --expect-lambda 0.1 --out <local_plot_folder>
```

Keep CIFs, confidence JSONs, full run outputs and SVG duplicates locally; commit the compact ranking tables and PNG figure only.
