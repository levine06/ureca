# Stage 3: Truth Accessibility

Calculate synthetic reference accessibility from experimental PDB structures in Stage 1 CSVs.
The calculation engine and reference tables are unchanged from [Stage 2](../../stage2/README.md);
existing method validation is documented [there](../../stage2/validation/README.md).
Stage 2 files are not modified. "Truth" means calculated from the experimental coordinate model,
not measured accessibility in solution. Candidate generation and RMSD evaluation are separate.

## New Loading and Masking

- `structures/mmcif.py` maps chains by `label_asym_id` and residues by `label_seq_id`.
  It uses the first model, keeps heavy atoms of the isolated chain, excludes zero-occupancy atoms,
  and selects the highest-occupancy alternative per atom (first listed on ties).
- `truth.py` excludes unresolved, incomplete, modified and sequence-mismatching residues.
  Selenomethionine is the accepted modification exception.
- Shadow masking excludes residues whose absolute SASA changes by more than 5 A^2 at any requested
  radius when modification-specific extra atoms are removed (`--shadow-threshold`).
  Present atoms of masked residues still shield neighbours; unresolved residues contribute no atoms.
- QC distinguishes coordinate-record coverage, usable-coordinate coverage and Stage 1 coverage.
  Saved `gap_distance` measures sequence distance to internal unresolved regions, excluding terminal
  gaps. Optional `--gap-flank` masks sequence neighbours of internal gaps; its default is zero.

## Batch Workflow

Run from this folder in a separate Python environment. CSVs require `pdb_id`, `label_chain_id`
and `sequence`. Replace the example paths with your actual CSV paths.

```bash
python -m pip install -e ".[biotite]"
python scripts/compute_truth_accessibility.py --csv development.csv validation.csv \
  --cache-dir cif_cache --out truth_out --download-only --workers 4
python scripts/compute_truth_accessibility.py --csv development.csv validation.csv \
  --cache-dir cif_cache --out truth_out --offline --workers 4
```

Run cluster calculations within a PBS CPU allocation; workers must not exceed allocated CPUs.
Defaults are radii 1.4, 2.5, 4.0 and 6.0 A and 1,000 sampling points.
The runner supports `--shard i/n`, `--collect-only` and resuming. Existing entry files are skipped:
use a new output folder when changing inputs or settings, or deliberately use `--overwrite`.
Collection includes every entry file in that folder, so keep different datasets/runs separate.

Outputs: per-chain `.npz` arrays, masks and metadata; `summary.csv`; `residues.csv.gz`; failure reports
when entries fail. Cluster outputs are separate from the committed code. Candidate comparisons must
use the saved reference masks and matching calculation settings.

## Dataset Check

The updated pipeline completed 30 development and 15 validation chains on the CPU cluster.
The summary confirms CYQ at position 56 and five shadow exclusions in 2ID7, and distinguishes
165 coordinate-record positions from 160 usable positions in 3RF2. The other flagged entries
(2BE4, 2O6W, 4DXZ and 4H4J) contain only selenomethionine modifications.
All 45 chains pass 90% usable-coordinate coverage. Test A and Test B have not been run.

Remaining choices are the 5 A^2 shadow threshold, alternative-conformation selection and uncertainty
near missing regions. Shadow masking does not estimate shielding by unresolved loops.
