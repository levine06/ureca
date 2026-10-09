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

- Candidates are verified, not assumed: residue names must match the Stage 1 sequence, and a candidate must cover every residue resolved in the experiment. Anything less is reported in `failures.csv` and never scored on an easier subset.
- By default each candidate is first restricted to the atoms the experiment resolved (`truth_atoms`), so missing experimental atoms do not count as a structural difference. Other options are `truth_residues` and `all_atoms` (see `accessfold/scoring.py`).
- Re-running is safe. A result is reused only if the fingerprint of the candidate files, truth file, structure cache and settings still matches. `--shard i/n` splits the work and `--collect-only` rebuilds the tables.

Outputs in `--out`: one `<entry>.npz` per protein, `candidate_scores.csv` (one row per candidate), `entry_summary.csv` (one row per entry) and `failures.csv`. See the docstring at the top of each script for the full list of options.

Add `--accessibility-only` for candidate accessibility and accessibility errors,
or `--structure-only` for structural accuracy. Without either flag the combined
workflow is unchanged. Use separate output folders for each mode.
