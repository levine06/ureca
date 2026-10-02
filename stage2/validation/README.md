# Stage 2 accessibility validation

Goal: check whether manually inspected buried and exposed residues agree with the package's total-residue accessibility values.

## Results

36 observations were recorded on three pre-cutoff structures, using chain A, protein heavy atoms, Bondi radii, a 1.4 Angstrom probe, and 4000 sphere points.

| Protein | Buried relative SASA | Exposed relative SASA | Manual ordering test |
| --- | --- | --- | --- |
| Ubiquitin (1UBQ) | 0.000-0.040 | 0.403-1.383 | Passed |
| Crambin (1CRN) | 0.000-0.039 | 0.125-0.702 | Passed |
| Lysozyme (1LYZ) | 0.000-0.010 | 0.025-0.516 | Passed |

Each protein has at least three buried and three exposed examples. Ambiguous observations are excluded from the ordering test. Crambin 13 and 32 were reclassified after seeing scores; their original observations are retained in manual_labels.csv. This is a consistency check, not a blinded estimate of classification accuracy.

## Evidence

- manual_labels.csv: observations and reclassification history.
- manual_comparison.csv: absolute and relative SASA for each observation.
- screenshots/INDEX.md: 12 representative images and their provenance.
- manifest.json: release dates and hashes; all three structures predate 2021-09-30 and are development fixtures.

Lysozyme 25, 108, 98 and crambin 16, 30 looked exposed but had limited water-probe accessibility. Probe sweeps support partial or size-dependent accessibility. Crambin 30 remains accessible to large probes, so probe exclusion alone does not explain it. Original visual labels were retained. Sweep CSVs contain the measurements.

## Supporting numerical checks

12 standard-probe PyMOL comparisons and all 15 validation tests passed. Smaller-probe comparisons cover 20 measurements (maximum area difference 0.512 square Angstrom). PyMOL measurements of 80 saved total-residue reference conformers agree within 0.483%. Reference reconstruction and normalization arithmetic passed. The original package suite had 87 passes and 5 skips; all five skipped checks subsequently passed in separate runs: the Dunbrack check for LEU and SER at 1.4 Angstrom, and four FreeSASA Lee-Richards checks at probes 1.4, 2.5, 4.0, and 6.0 Angstrom. The FreeSASA run used local Ubuntu, Python 3.14.4, and pytest 9.1.1; its saved XML confirms four tests with zero failures, errors, or skips. This is not a claim that the full suite was rerun together on Linux.

The external Dunbrack library SHA256 is 71c16926a4140604e97f1c3c2625dc96a87a13c059ec565cbcacbed46265fe4a; it is not redistributed. Numerical inputs and exports are retained for reproduction. FreeSASA evidence is saved in freesasa-linux-results.xml, freesasa-linux-log.txt, and freesasa-linux-environment.txt.

## Reproduce

From stage2, in your chosen Python environment:

```text
python -m pip install -e ".[test]"
python -m pytest validation/test_structures.py validation/test_step2.py -q -p no:cacheprovider
```

In PyMOL, File > Run Script > START_HERE.py regenerates standard measurements and inspection sessions. Restore inspection_controls.py after opening a session; use `inspect_residue 15` to highlight a residue. For reference and smaller-probe regeneration, run `python validation/prepare_step2.py`, then run STEP2_PYMOL.py in PyMOL. Paths are portable. The legacy START_HERE.pml requires this folder as working directory.

On Ubuntu with python3-venv, python3-dev, and build-essential installed, run `bash validation/run_freesasa_linux.sh` for the four optional FreeSASA checks. It saves logs, environment versions, and XML results. Confirm four passed without skips.

## Boundary

Results support continued development with consistent preparation and settings. They do not validate experimental labeling, all possible physical reference maxima, or independent side-chain normalization. Terminal RSA can exceed one. Stage 1 numbering, missing atoms/residues, alternate conformations, chain context, and split/homology exclusions still need integration checks. No held-out test structures were used to tune this validation.
