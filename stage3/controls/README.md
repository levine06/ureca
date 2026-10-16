# Stage 3 Controls

**Question:** does position-specific reference accessibility help rank and select accurate structures, beyond shuffled or degraded information?

`run_controls.py` uses the scoring NPZs and candidate tables. Its implementation is `../truth_accessibility/src/accessfold/controls.py`. The candidates and structural scores remain fixed; controls shuffle or reverse the reference, substitute another protein's profile, retain 50%/20% of residues, or reduce both profiles to buried/exposed labels. Shuffling moves each residue's values across probe radii together.

Per protein, it measures ranking agreement (Spearman rho) and the structural rank of the lowest-accessibility-error candidate. Reference rules are random selection, the best possible candidate, and OpenFold3's top-ranked candidate. Summaries and bootstrap intervals use proteins, not individual candidates, as the sampling unit.

## Run

Install the package from the repository root: `pip install -e stage3/truth_accessibility`. Prepare one folder with all 85 scoring NPZs and two combined tables, preserving `entry`, `candidate` and OpenFold `rank`. The test candidate table contains both accessibility and structural metrics. Do not restrict inputs to the old development/validation CSVs.

```bash
python stage3/controls/run_controls.py \
  --scores-dir <all_scores> \
  --accessibility-csv <all_scores>/candidate_scores.csv \
  --structure-csv <all_scores>/structure_candidate_scores.csv \
  --rank-csv <all_scores>/structure_candidate_scores.csv \
  --rank-column rank --strict-join --out <controls_out>
```

The run uses 1,000 shuffle/dropout repetitions, up to 20 other profiles and 10,000 bootstrap repetitions by default. Rank 1 is OpenFold3's best ranking score, not the experimentally best structure or mean pLDDT. The runner also emits an exploratory RMSD-range subset; the committed figures use all proteins without that distinction.

Working NPZs, null distributions and intermediate CSVs are not deliverables committed here. Keep the controls output locally to [regenerate the figures](../plots/README.md).
