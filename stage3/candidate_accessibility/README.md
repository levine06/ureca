# Stage 3, step 3: Candidate accessibility

This is step 3 of the [Stage 3 experiment](../README.md): calculating the predicted accessibility `ŷ^(k) = f(X_k)` of every OpenFold3 candidate from [candidate generation](../candidate_generation/README.md).

`compute_candidate_accessibility.py` uses the same calculation as the [truth pipeline](../truth_accessibility/README.md), so truth and candidate values are directly comparable. Step 4 (accessibility error) compares its output with the saved truth `.npz` files.

## What it does

For every candidate in `candidate_generation/candidates_manifest.csv`:

1. Loads the candidate `.cif` (chain `A`) with `load_chain_from_mmcif`, using the protein's sequence from its truth file.
2. Computes relative and absolute SASA, for the whole residue and the side chain only, with `truth_arrays`. The probe radii (1.4, 2.5, 4.0 and 6.0 Å) and the number of sampling points (1000) are read from the protein's truth `.npz`, so they always match.
3. Keeps only the residues in the saved truth `mask` that are also valid in the candidate. The same residues are used on both sides of the comparison.

It stops with an error if a candidate's length or residue names differ from the truth entry.

## Running

Use the `openfold3` conda environment. If `accessfold` is not installed, the script falls back to `truth_accessibility/src`.

```bash
# one protein (test)
python stage3/candidate_accessibility/compute_candidate_accessibility.py --only 1L66_A

# everything
python stage3/candidate_accessibility/compute_candidate_accessibility.py --workers 4

# one shard of four
python stage3/candidate_accessibility/compute_candidate_accessibility.py --shard 0/4 --workers 4
```

Run it inside an allocation on the cluster. `--workers` must not exceed the CPUs you were given.

| Option | Default | Notes |
|---|---|---|
| `--only 1L66_A ...` | all | restrict to named proteins |
| `--workers` | 1 | proteins processed in parallel, one process each |
| `--shard I/N` | `0/1` | process every N-th protein starting at I |
| `--manifest` | `candidate_generation/candidates_manifest.csv` | |
| `--truth-dir` | `truth_accessibility/results` | folder with the truth `.npz` files |
| `--out` | `candidate_accessibility/results` | |
| `--overwrite` | off | recompute proteins that already have an output file |

Re-running is safe: proteins that already have an output file are skipped. A protein that fails is reported and does not stop the others; failures are listed in `results/failures.csv`.

One protein takes about 2 minutes on a single worker (25 candidates), so all 45 proteins take roughly 1.5 hours with one worker.

## Output

One `results/<PDB>_<CHAIN>.npz` per protein (git-ignored). Arrays are stacked over the protein's K candidates, with N sequence positions:

| Array | Shape | Meaning |
|---|---|---|
| `candidate_id`, `seed`, `sample` | K | identify each row; match the manifest |
| `residue_names` | N | residue types |
| `mask` | K × N | truth `mask` AND candidate-valid residues |
| `rel_<r>`, `abs_<r>` | K × N | relative and absolute accessibility at probe radius `r` (`1.4`, `2.5`, `4`, `6`) |
| `sc_rel_<r>`, `sc_abs_<r>` | K × N | the same, side chain only |
| `meta` | scalar | JSON with the settings, the truth file used and the package version |

Values are NaN outside `mask`. The array names match the truth `.npz` files, so a candidate row `rel_1.4[k]` can be compared directly with the truth `rel_1.4`.
