# Stage 3, step 1: True synthetic accessibility

This is step 1 of the [Stage 3 experiment](../README.md): calculating the reference accessibility from the experimental structures. "True" means calculated from experimental coordinates, not measured accessibility in solution.

See also [candidate generation](../candidate_generation/README.md) and
[candidate scoring](../scoring/README.md).

## Workflow

| Task | Script |
| --- | --- |
| Calculate true synthetic accessibility | `compute_truth_accessibility.py` |
| Generate candidate OpenFold3 structures | `../candidate_generation/generate_candidates.py` |
| Calculate predicted accessibility | `../scoring/score_candidates.py --accessibility-only` |
| Calculate accessibility error | Same accessibility-only run, against experimental-reference values |
| Calculate true structural accuracy | `../scoring/score_candidates.py --structure-only` |

`../scoring/prepare_openfold3_outputs.py` checks existing candidates before scoring;
it generates no structures and calculates no accessibility or accuracy.
`src/accessfold/` contains the required engine and reference tables; see
[Stage 2](../../stage2/README.md) and [its validation](../../stage2/validation/README.md).

## Run

Run from `stage3/truth_accessibility/` in a separate Python environment.
The CSVs must contain `pdb_id`, `label_chain_id` and `sequence`.

```bash
python -m pip install -e ".[biotite]"
python compute_truth_accessibility.py \
  --csv ../../stage1/datasets/development.csv ../../stage1/datasets/validation.csv \
  --cache-dir cif_cache --out truth_out --download-only --workers 4
python compute_truth_accessibility.py \
  --csv ../../stage1/datasets/development.csv ../../stage1/datasets/validation.csv \
  --cache-dir cif_cache --out truth_out --offline --workers 4
```

For scoring candidates against these reference NPZs, see the [scoring README](../scoring/README.md).

## Outputs

Development and validation: [results](results/README.md). All 30 development and
15 validation chains pass 90% usable-coordinate coverage. 2ID7 has CYQ at position
56 and five shadow exclusions; 3RF2 has 165 coordinate-record positions but 160
usable positions. The modifications in 2BE4, 2O6W, 4DXZ and 4H4J are accepted
selenomethionine.

Test A and Test B: [results_test](results_test/README.md), computed with the same settings.

Each chain has an NPZ (`<PDB>_<chain>.npz`), and each folder has a `summary.csv` and a
`residues.csv.gz`. The scoring outputs built from these files are described in the
[scoring README](../scoring/README.md#results).

## Limitations

Unresolved, incomplete, modified and shadow-affected reference positions are
masked. Missing regions are not modelled as shielding atoms. The 5 A^2 shadow
threshold, alternative-conformation selection and uncertainty near missing
regions have not been tested for sensitivity. Inspect failure reports and QC
before analysis.
