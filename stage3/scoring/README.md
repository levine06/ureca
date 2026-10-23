# Stage 3, steps 3-5: Scoring candidates

This folder scores the OpenFold3 candidates from [candidate generation](../candidate_generation/README.md) against the experimental structures and the [truth accessibility](../truth_accessibility/README.md) values. It covers steps 3 to 5 of the [Stage 3 experiment](../README.md).

For every candidate, `score_candidates.py` computes:

- **Candidate accessibility** `ŷ^(k) = f(X_k)`, using the same package, probe radii and number of sampling points as the truth.
- **Accessibility error** against the saved truth profile, on exactly the residues in the truth `mask`.
- **Structural error** against the experimental chain: Cα RMSD (primary), lDDT-Cα and TM-score.

The calculations live in the `accessfold` package in `truth_accessibility/src/accessfold/` (`scoring.py`, `structure_metrics.py`, `controls.py`, `structures/predicted.py`). This folder holds the two command-line scripts.

## Files

| Path | Contents |
|---|---|
| `prepare_openfold3_outputs.py` | checks the OpenFold3 outputs against Stage 1 and writes `ranking.csv`; scores nothing |
| `score_candidates.py` | computes the accessibility and structural errors for every candidate |
| `results/development_validation/` | committed result tables for the 45 development and validation proteins (see [Results](#results)); also holds the git-ignored per-residue table `accessibility_candidate_residues.csv.gz` |
| `results/test/` | committed result tables for the 40 test proteins (see [Results](#results)) |
| `runs/` | working output of each scoring run (per-protein `.npz` files and the tables the script writes); git-ignored |
| `logs/` | PBS job logs; git-ignored |
| [`../plots/`](../plots/README.md) | combined accessibility-versus-accuracy figures, controls figures and plotting scripts |

## What the scripts do

**`prepare_openfold3_outputs.py`** is a pre-flight check that scores nothing. It walks the OpenFold3 output folder, matches each `<query>_seed_<n>_sample_<i>_model.cif` to a Stage 1 entry, and validates it with the same loader the scorer uses (chain present, residue names equal to the Stage 1 sequence, numbering 1..N). It reads `sample_ranking_score` and the other confidence values from the sibling `_confidences_aggregated.json`, writes them to `ranking.csv`, and prints the `--candidates` pattern to use next. Higher ranking score is assumed to be better. It exits with status 1 if a model fails validation or a Stage 1 entry has no predictions.

**`score_candidates.py`** scores every candidate of every entry against the experimental structure and the saved truth file. It does the orchestration only; the calculations are in `accessfold`.
- Structural accuracy: Cα RMSD (Å, lower is better), lDDT-Cα and TM-score (0 to 1, higher is better), plus the RMSD between every pair of candidates. Residues are paired by sequence position, so no alignment is needed.
- Accessibility: candidate accessibility and its error against the truth profile on the truth-mask residues (MAE, RMSE and correlation error, per probe radius and averaged over radii).
- Mode flags, mutually exclusive: `--structure-only` (structural accuracy only), `--accessibility-only` (candidate accessibility and accessibility errors only), or neither for both. Use separate output folders for each mode.
- `--verify-truth` checks the saved truth accessibility against the structure cache and cannot be combined with `--structure-only`. Keep `--workers` no larger than the allocated CPU count.
- Optionally joins the model's own ranking (`--ranking-csv`) to fill the top-ranked columns.
- Every accessibility run saves all four per-residue profiles at every probe radius as `abs_<r>`, `rel_<r>`, `sc_abs_<r>` and `sc_rel_<r>` (candidate, residue) in each entry NPZ. Absolute values are in A^2 and relative values are unitless; missing or excluded values are blank, and glycine side-chain values are undefined, not zero.
- `candidate_scores.csv` holds MAE, RMSE, correlation error and valid-residue counts for each profile and radius. `--kind` selects only the headline comparison (`rel` by default), not which profiles are saved. Profiles use the matched experimental atom environment by default. Relative-only outputs from older runs must be rerun, because the current scoring schema prevents their reuse.

## Results

Both splits have the same two tables, from one combined run of `score_candidates.py` (structure and accessibility together):

| Split | Folder | Proteins | Candidates | Working output (git-ignored) |
|---|---|---|---|---|
| development + validation | `results/development_validation/` | 45 | 1,125 | `runs/dev_val/` |
| test_a + test_b | `results/test/` | 40 | 1,000 | `runs/test/` |

- `candidate_scores.csv`: one row per candidate (25 per protein).
  - Structure: `rmsd_ca` (Å, lower is better), `lddt_ca` and `tm_score` (0 to 1, higher is better), `n_reference` and `n_common` (residues compared), `coverage_ca` and `coverage_mask`.
  - Ranking: `rank_score` and `rank`, the model's own ranking from `sample_ranking_score` in `candidate_generation/candidates_manifest.csv` (rank 1 is the highest score).
  - Accessibility: `ac_error` (mean absolute error of the relative whole-residue accessibility, averaged over the four probe radii; lower is better), `ac_rmse`, `ac_corr_error` (1 − Pearson r) and `n_ac_residues`, followed by the MAE, RMSE, correlation error and residue count for each profile (`abs`, `rel`, `sc_abs`, `sc_rel`) at each radius (1.4, 2.5, 4.0 and 6.0 Å). `ac_kind` names the profile used for the headline columns (`rel`).
- `entry_summary.csv`: one row per protein, pooling its 25 candidates.
  - RMSD minimum, median, maximum and spread; the median and maximum RMSD between candidates (`pair_rmsd_*`); and the number of candidates within 1, 2 and 4 Å (`n_within_*A`).
  - `rmsd_top1`, `rmsd_best_top{1,5,20}` and `top1_minus_best`: the RMSD of the model's top-ranked candidate, of the best candidate among its top 1, 5 and 20, and the gap between the top-ranked and the best.
  - `rmsd_acc_selected`: the RMSD of the candidate with the lowest `ac_error`.
  - `rho_ac_vs_{rmsd,lddt,tm}` and `rho_accorr_vs_{rmsd,lddt,tm}`: the Spearman correlation, over the protein's 25 candidates, between the accessibility error and the structural error (RMSD, 1 − lDDT, 1 − TM-score). `ac` uses `ac_error` and `accorr` uses `ac_corr_error`. A positive value means candidates that match the experimental accessibility better also have a better structure.
  - `topN_unavailable_reason` is empty when the ranking is complete for the protein.

Per-protein NPZ files are written to the `runs/` folders and are not committed. The development and validation per-residue accessibility table, `accessibility_candidate_residues.csv.gz`, is kept beside its result tables but is git-ignored.

Three proteins were generated under a chain id other than `A`, so they are scored separately by pointing at their actual candidate folder and chain:

| Entry | Candidate folder | `--candidate-chain` |
|---|---|---|
| `2H36_A` (development) | `2H36_X` | `X` |
| `6F4C_A` (validation) | `6F4C_B` | `B` |
| `8EXF_B` (test_b) | `8EXF_B` | `B` |

Score each with `--only <entry>` and its own `--candidates` pattern, then rebuild the tables for all entries with `--collect-only`.

## Workflow

Run from the repository root, in an environment where `accessfold` is installed (`pip install -e "stage3/truth_accessibility[biotite]"`). The truth files come from `truth_accessibility/compute_truth_accessibility.py`.

```bash
# 1) check the predictions and write the ranking table
python stage3/scoring/prepare_openfold3_outputs.py \
  --csv stage1/datasets/development.csv stage1/datasets/validation.csv \
  --predictions <openfold3_output_dir> --out ranking.csv

# 2) score every candidate
python stage3/scoring/score_candidates.py \
  --csv stage1/datasets/development.csv stage1/datasets/validation.csv \
  --truth-dir <truth_out> --cache-dir <cif_cache> \
  --candidates "<openfold3_output_dir>/{entry}/seed_*/*_model.cif*" --candidate-chain A \
  --ranking-csv ranking.csv --out scores_out --workers 8
```

`prepare_openfold3_outputs.py` prints the `--candidates` pattern to use. `--truth-dir` is the folder of truth `.npz` files (the committed ones are in `truth_accessibility/results/`), and `--cache-dir` is the folder of downloaded experimental structures.

## What the checks do

- Candidates are verified, not assumed: residue names must match the Stage 1 sequence, and a candidate must cover every residue resolved in the experiment. Anything less is recorded in the entry's `failed_candidates` (the entry itself goes to `failures.csv` if too few candidates remain) and never scored on an easier subset.
- By default each candidate is first restricted to the atoms the experiment resolved (`truth_atoms`), so missing experimental atoms do not count as a structural difference. Other options are `truth_residues` and `all_atoms` (see `accessfold/scoring.py`).
- Re-running is safe. A result is reused only if the fingerprint of the candidate files, truth file, structure cache and settings still matches. `--shard i/n` splits the work and `--collect-only` rebuilds the tables.

Outputs in `--out`: one `<entry>.npz` per protein, `candidate_scores.csv` (one row per candidate), `entry_summary.csv` (one row per entry) and `failures.csv`. See the docstring at the top of each script for the full list of options.
