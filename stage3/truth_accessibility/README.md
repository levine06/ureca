# Stage 3: Accessibility

Calculate reference accessibility from experimental structures and compare it
with OpenFold3 candidate accessibility. "True" means calculated from experimental
coordinates, not measured accessibility in solution. The calculation engine and
validation are described in [Stage 2](../../stage2/README.md) and
[its validation](../../stage2/validation/README.md).

See the [Stage 3 overview](../README.md), partner-owned
[candidate generation](../candidate_generation/README.md), and the separate
[candidate scoring workflow](../scoring/README.md).
Those workflows are preserved; the scoring modes below are provided in `../scoring/`.

## Workflow

| Task | Script |
| --- | --- |
| Calculate true synthetic accessibility | `compute_truth_accessibility.py` |
| Generate candidate OpenFold3 structures | Partner workflow: `../candidate_generation/generate_candidates.py` |
| Calculate predicted accessibility | `../scoring/score_candidates.py --accessibility-only` |
| Calculate accessibility error | Same accessibility-only run, against experimental-reference values |
| Calculate true structural accuracy | `../scoring/score_candidates.py --structure-only`, or your partner's accuracy workflow |

`../scoring/prepare_openfold3_outputs.py` checks existing candidates before scoring;
it generates no structures and calculates no accessibility or accuracy.
`src/accessfold/` contains the required engine and reference tables.

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

For scoring, use the saved reference NPZs, original CIF cache and candidate CIFs.
Prepare candidate files first, then use the printed candidate path.
Add `--accessibility-only` for candidate accessibility/errors or
`--structure-only` for RMSD, lDDT-Ca, TM-score and pairwise RMSD.
Without either flag, scoring runs both. Use separate output folders for each
mode, and `--workers` no larger than the allocated CPU count.
`--verify-truth` checks saved accessibility against the cache and cannot be used
with `--structure-only`.

## Outputs

Experimental outputs are in [results](results/README.md): all 30 development and
15 validation chains pass 90% usable-coordinate coverage. 2ID7 has CYQ at position
56 and five shadow exclusions; 3RF2 has 165 coordinate-record positions but 160
usable positions. The modifications in 2BE4, 2O6W, 4DXZ and 4H4J are accepted
selenomethionine. Test A and Test B have not been run.

Scoring writes `candidate_scores.csv`, `entry_summary.csv` and per-entry NPZs.
NPZs store profiles as `c[candidate,residue,radius]`, experimental values as
`y`, and identifiers/radii in JSON `meta`. Single-mode outputs omit skipped
metrics. Join them by entry and candidate identifier before combined analysis.
Ranking scores are required for top-ranked summaries.

## Limitations

Unresolved, incomplete, modified and shadow-affected reference positions are
masked. Missing regions are not modelled as shielding atoms. The 5 A^2 shadow
threshold, alternative-conformation selection and uncertainty near missing
regions have not been tested for sensitivity. Inspect failure reports and QC
before analysis.
