# Validation status

The existing package suite completed: 87 passed, 5 skipped. Four skipped tests require optional FreeSASA; one requires the external Dunbrack rotamer library.

All 12 real-structure comparisons against Bio.PDB.SASA passed: three structures times four probe radii, 1000 sphere points, matching element-based Bondi radii. Values matched to floating-point precision. These implementations use the same style of sphere-point sampling, so this checks implementation and residue aggregation but provides less algorithmic independence than the planned PyMOL comparison. Detailed results are in biopython_checks.csv.

RCSB initial public release dates were checked before downloading: 1UBQ 1987-04-16; 1CRN 1981-07-28; 1LYZ 1977-04-12. All are strictly before 2021-09-30. Files and hashes are recorded in manifest.json. They are reserved for development; the pending Stage 1 split and homology exclusions still need reconciliation.

The user ran START_HERE.pml inside PyMOL, generating pymol_sasa.csv and inspection sessions. All 12 automated PyMOL comparisons against AccessFold passed (three proteins at four probe radii). One pytest cache-write warning did not affect the calculations.

The user manually inspected ubiquitin and labelled residues 3, 15, and 23 buried, and 48, 63, and 76 exposed. These observations are recorded in manual_labels.csv. At probe radius 1.4 Angstrom with 4000 points, their relative SASA values are respectively 0.000, 0.039, 0.002, 0.403, 0.582, and 1.383. The ordering check passes: every labelled buried residue is below every labelled exposed residue. Residue 76 is terminal; its value above one is compatible with a reference based on an internal tripeptide residue and is not clipped.

Manual observations have now been collected for all three proteins (36 residues total). Original observations and reclassification history are preserved in the notes. After reinspection, the user explicitly reclassified crambin residues 13 and 32 as buried overall with small exposed portions. Their initial ambiguous labels and the fact that scores had already been disclosed are recorded; these are follow-up observations rather than blind validation. The user subsequently reported lysozyme residue 17 as buried, providing a third buried example. The unchanged manual-label test now passes for all three proteins (one test passed): each meets the original coverage requirement and its buried group has lower relative SASA than its exposed group. The planned 12 representative screenshot views are now archived and indexed in screenshots/INDEX.md with provenance and limitations.

The observed buried/exposed groups are ordered correctly in each protein, but ordering alone is a weak check. Lysozyme residues labelled exposed include 25 (RSA 0.0253), 108 (0.0453), and 98 (0.0581), all low accessibility. Crambin exposed residues 16 (0.1250) and 30 (0.1589) are also less accessible than other exposed examples. These warrant reinspection with attention to total-residue versus side-chain exposure and the limits of local sphere views. No binary numerical threshold was predefined, so these are review flags rather than formal classification failures. Do not relabel solely to agree with calculated scores.

All 36 observations with absolute and relative SASA at 1.4 Angstrom are saved in manual_comparison.csv. No overlap occurs between the current buried and exposed groups within each protein, although this ordering check does not establish that all visually exposed labels represent high absolute or relative accessibility. The manual regression test passes. The scoped smaller-probe and total-residue reference checks described below now pass, and the planned screenshot collection is complete. Stage 1 integration remains outstanding. VALIDATION_SUMMARY.md consolidates the completed three-step workflow and scientific limits.

## Lysozyme residue 25 follow-up

The user reinspected residue 25 and proposed that it faces a narrow groove where larger probes cannot fit. A follow-up calculation with the same cleaned coordinates, Bondi radii, and 4000 sphere points supports probe-size sensitivity: absolute SASA is 107.67 square Angstrom at radius 0.0, 73.75 at 0.5, 19.38 at 1.0, 5.09 at 1.4, and 0.00 at 2.5 Angstrom. Radius zero is a geometric diagnostic, not a water-sized solvent model. Absolute areas are compared because bundled relative-SASA references are unavailable for the smaller diagnostic radii.

This is consistent with a narrow groove limiting accessibility and explains why visible exposure can coexist with low water-probe SASA. The sweep does not prove a specific entry route: the local Shrake-Rupley calculation tests sphere overlap, not whether a probe centre can travel to each point from bulk solvent. The original visual exposed label is retained with the qualified observation. No package defect is demonstrated by this result; it is not a claim that all package behaviour has been validated.

## Lysozyme residues 108 and 98 follow-up

The same diagnostic sweep was run on residues 108 and 98. Both show strong probe-size sensitivity, consistent with restricted local access. The user subsequently reinspected both in PyMOL and supplied screenshots: residue 108 was described as potentially facing a groove or valley, and residue 98 was confirmed by the user to have a narrow groove appearance. These visual observations are consistent with the sweep but do not prove a specific solvent entry route. The original exposed labels are retained as visual observations, qualified by limited water-probe accessibility.

| Probe radius (Angstrom) | Residue 25 SASA | Residue 108 SASA | Residue 98 SASA |
| --- | --- | --- | --- |
| 0.0 | 107.671 | 155.270 | 107.430 |
| 0.5 | 73.750 | 126.496 | 72.948 |
| 1.0 | 19.383 | 41.257 | 27.678 |
| 1.4 | 5.088 | 12.671 | 11.553 |
| 2.5 | 0.000 | 0.000 | 0.000 |
| 4.0 | 0.000 | 0.000 | 0.000 |
| 6.0 | 0.000 | 0.000 | 0.000 |

All areas are square Angstrom. This diagnostic sweep is from AccessFold; the smaller probe radii have not been independently cross-checked against PyMOL. The prior PyMOL checks cover 1.4, 2.5, 4.0, and 6.0 Angstrom.

## Crambin residue 16 follow-up

The user supplied a reinspection screenshot and suggested a valley limits access. AccessFold was run on the same cleaned crambin coordinates with Bondi radii and 4000 sphere points. Absolute SASA is 76.609 square Angstrom at probe radius 0.0, 49.387 at 0.5, 27.500 at 1.0, 20.651 at 1.4, 9.147 at 2.5, 0.106 at 4.0, and 0.000 at 6.0 Angstrom. The original exposed visual label is retained with the qualified observation.

The decline supports probe-size-dependent restricted accessibility, consistent with the user's valley interpretation. Residue 16 retains some area at radius 2.5, unlike the three lysozyme follow-up residues. This does not prove a specific groove or solvent entry route and does not demonstrate a package bug. Smaller diagnostic probe radii have not been independently checked with PyMOL. The user subsequently described the depression around residue 16 as visually wide and shallow; this is retained as an observation rather than a geometry inferred from SASA.

## Crambin residue 30 follow-up

The user supplied multiple reinspection screenshots and described apparent exposure with a possible valley. The original exposed label is retained. Using the same cleaned coordinates, Bondi radii, and 4000 sphere points, residue 30 absolute SASA is 92.138 square Angstrom at probe radius 0.0, 57.408 at 0.5, 31.092 at 1.0, 26.913 at 1.4, 21.125 at 2.5, 17.333 at 4.0, and 13.857 at 6.0 Angstrom. Results are also saved in crambin_30_probe_sweep.csv.

Residue 30 retains measurable area even for the 6.0 Angstrom probe, unlike residue 16 and the lysozyme follow-up residues. Some surface therefore remains locally accessible to large probes; a valley entirely excluding large probes does not explain this case. A visible exposed portion alongside substantial occlusion of the rest of the residue is compatible with the low water-probe relative SASA (0.1589), but this sweep alone does not establish its geometry. Absolute SASA need not decrease monotonically with probe radius in general; the decrease here is an observed result. No label or regression threshold has been changed to match the scores. The sweep has not demonstrated a package defect, and smaller diagnostic radii still lack an independent PyMOL cross-check.

## Step 2 normalization and independent follow-ups

prepare_step2.py reconstructed the stored maximizing Gly-X-Gly conformation for all 20 standard amino acids at each of four reference radii (80 cases), using the recorded phi/psi/chi angles. Every reconstructed conformer passed the package's steric filter. At 10000 sphere points, all central-residue absolute areas reproduced the stored total-residue maxima to floating-point precision. Relative SASA equalled absolute SASA divided by the correct residue-type and probe-radius reference in all cases. These checks establish reproducibility and arithmetic consistency, not algorithmic independence or exhaustive physical maxima. Side-chain maxima have not been independently reconstructed here because their maximizing conformations are not separately recorded.

The nine existing reference-table tests passed. An initial attempt encountered two temporary-directory permission errors; rerunning with a fresh workspace-local temporary directory resolved these environmental errors. The bundled 1.4 Angstrom maxima differ from the package's Tien 2013 theoretical comparison table by -4.58% to +1.37%, mean absolute difference 2.37%. Different radii, geometry, and conformer restrictions prevent treating that approximate agreement as exact equivalence. Literature basis: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080635 . The paper excludes chain-terminating residues from its empirical analysis, reinforcing the need to qualify terminal RSA values above one.

Independent PyMOL verification completed successfully. STEP2_PYMOL.py exported 20 measurements for the five investigated residues at probe radii 0.0, 0.5, 1.0, and 1.4 Angstrom, plus 80 central-residue areas for the reference-winner conformers, using the same Bondi radii and complete structure context. Both tests in test_step2.py passed, covering all 100 measurements. Criteria were set before the independent export: residue areas within max(3 square Angstrom, 3%); reference areas within max(3 square Angstrom, 2%). The maximum residue-area discrepancy was 0.512 square Angstrom (lysozyme 25 at probe 0.5). Maximum reference-area relative discrepancy was 0.483% (GLN at probe 6.0), with mean absolute relative discrepancy 0.150%. The zero-radius case is diagnostic. No normalization references were invented for smaller radii. Results are saved in step2_pymol.csv.

The helper initially encountered temporary-object selection errors and then a string-versus-Path error before completing an export. Direct coordinate ingestion into fresh objects and explicit Path conversion resolved these helper issues. No partial export was accepted as a passing comparison.

Step 2 is complete within its stated scope: smaller-probe absolute-area agreement, total-residue reference-winner reproducibility and independent area agreement, table consistency, and normalization arithmetic. This does not prove that the conformer search found global physical maxima, validate side-chain reference maxima independently, or calibrate accessibility against real labeling experiments. Those are separate scientific limitations. All checks use pre-cutoff development structures or synthetic tripeptides; no held-out experimental structures were added.

## Portability

Scripts now locate fixtures relative to their own files. The 80 reference records use relative paths, and generation preserves this convention. START_HERE.py provides a portable PyMOL entry point selected through File > Run Script. The legacy PML launcher requires the validation folder as working directory. Instructions target stage2/validation and use the user's chosen Python environment rather than a machine-specific virtual environment. Step 2 saved-result tests passed after copying their inputs to a separate stage2/validation directory. The revised PyMOL launchers have been syntax-checked, but have not yet been executed in a different PyMOL installation.

## Optional real Dunbrack library check

The user supplied ALL.bbdep.rotamers.lib after selecting the Simple Mode 18-standard-residue default 5%-stepdown download. Its SHA256 is 71c16926a4140604e97f1c3c2625dc96a87a13c059ec565cbcacbed46265fe4a. The library remains an external dependency and is not copied into the validation archive.

The previously skipped test_real_library_gives_maxima_close_to_staggered_grid now passed (1 selected test passed). It compares LEU and SER reference maxima using library-mean rotamers versus the staggered grid at probe 1.4 Angstrom, with a coarse 30-degree backbone grid, 500 sampling points, and 2000 refinement points. The unchanged criterion is 0.85 < library_max / staggered_max <= 1.02. This is a limited sensitivity check, not an all-residue/all-radius table rebuild or evidence that the library is needed for ordinary SASA calculations. The bundled references are unchanged. The original suite result of 87 passed and 5 skipped is historical; this additional run closes its Dunbrack skip. Four optional FreeSASA checks remain unrun in this environment.

## Optional FreeSASA installation attempt

Installation was attempted in the isolated Windows Python 3.12 validation environment using pip with --use-pep517. After a cache-permission failure, a --no-cache-dir retry reached compilation but failed because Microsoft Visual C++ 14.0 or greater is required and unavailable. FreeSASA was not installed, and its four tests were not run; this is an environment limitation rather than an observed AccessFold test failure. No system compiler was installed and no reference tables or package source were changed.

Run the optional checks from stage2 in a Python environment where FreeSASA can be installed (for example a Linux environment with the required compiler tooling):

```text
python -m pip install -e ".[test]"
python -m pip install freesasa --use-pep517 --no-cache-dir
python -c "import freesasa; print(freesasa.__file__)"
python -m pytest tests/test_physical_sanity.py -k freesasa -v -rs
```

Confirm four passed, rather than four skipped. The import check prevents treating an absent optional dependency as a completed validation. On this Windows installation, compiler tooling is required before repeating these commands. The check uses FreeSASA Lee-Richards slicing with matched element radii to provide a different numerical algorithm from point sampling.
