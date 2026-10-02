# Stage 2 validation

Place this folder at stage2/validation, beside stage2/pyproject.toml. Fixture paths are resolved relative to scripts; reference_winners.json stores relative paths. Saved results can be checked without opening PyMOL.

## Install and check saved results

From the repository's stage2 folder, using your chosen Python environment:

```text
python -m pip install -e ".[test]"
python -m pytest validation/test_structures.py validation/test_step2.py -q -p no:cacheprovider
```

PyMOL uses its own Python; it does not need AccessFold for the export scripts.

## PyMOL inspection

Choose File > Run Script and browse to validation/START_HERE.py. This launcher finds sibling files regardless of checkout location or current directory. It exports standard-probe areas, creates inspection sessions, and opens ubiquitin. The legacy START_HERE.pml requires PyMOL's working directory to be the validation folder; prefer the Python launcher.

After opening a different *_inspect.pse, restore the helper through File > Run Script > inspection_controls.py. For example:

```pymol
inspect_residue 15
hide surface, protein
show spheres, protein
set sphere_scale, 1
orient protein
```

Rotate through several angles. Restore the surface with `hide spheres, protein` and `show surface, protein`. Classify the whole residue; an exposed tip does not imply high total accessibility. Record observations in manual_labels.csv before viewing scores, and preserve uncertain cases as ambiguous. The manual regression requires three buried and three exposed cases per protein and checks their ordering. Current labels include documented follow-up reclassifications after score disclosure.

The 12 archived representative images and their provenance are in screenshots/INDEX.md. They support inspection, not independent numerical validation.

## Reproduce numerical follow-ups

From stage2:

```text
python validation/prepare_step2.py
```

In PyMOL choose File > Run Script and browse to validation/STEP2_PYMOL.py. Wait for `STEP 2 EXPORT COMPLETE: 100 measurements saved to step2_pymol.csv`, then rerun the saved-result tests above. Temporary measurement objects are removed; inspection objects are retained.

Other regeneration commands, run from stage2:

```text
python validation/check_biopython.py
python validation/compare_manual.py
```

prepare.py downloads public fixtures and checks release dates; it requires internet access. Other regeneration scripts use local files. Raw hashes and dates are in manifest.json. Only development fixtures released strictly before 2021-09-30 are used. Exclude them from final evaluation and reconcile homology exclusions when Stage 1 is ready.

## Scope

VALIDATION_SUMMARY.md records settings and limits; RESULTS.md records detailed history. Standard-probe tests permit 2% total-area difference, correlation above 0.99, and maximum residue difference below 8 square Angstrom. Step 2 tolerances were fixed before its independent export. Reference winners verify stored-conformer areas and reproducibility, not exhaustive global physical maxima. Side-chain reference maxima and Stage 1 integration have not been independently validated here.
