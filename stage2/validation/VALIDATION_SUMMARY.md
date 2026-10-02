# Stage 2 validation summary

## Outcome

The tested total-residue accessibility workflow is suitable for continued method development under the settings below. Numerical cross-checks and the current manual regression test pass. This is a scoped validation, not a guarantee for arbitrary structures, a calibration of experimental labeling, or a completed Stage 1 integration.

## Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Existing package suite | 87 passed; 5 optional tests skipped | RESULTS.md |
| Real-structure Bio.PDB comparisons | 12 comparisons passed | biopython_checks.csv |
| Real-structure PyMOL comparisons | 12 comparisons passed | pymol_sasa.csv; test_structures.py |
| Manual regression | Passed on all three proteins | manual_labels.csv; manual_comparison.csv; test_structures.py |
| Smaller-probe follow-ups | All 20 measurements passed; maximum area difference 0.512 square Angstrom | step2_accessfold.csv; step2_pymol.csv; test_step2.py |
| Reference-winner reconstruction | All 80 total-residue maxima reproduced; normalization arithmetic passed | reference_winners.json; prepare_step2.py |
| Independent reference-winner measurement | All 80 measurements passed; maximum relative difference 0.483% | step2_pymol.csv; test_step2.py |
| Existing reference-table suite | 9 tests passed | RESULTS.md |

The original five skips required optional FreeSASA or an external Dunbrack library. The real-library Dunbrack test was subsequently run with a user-supplied library and passed for LEU and SER at probe 1.4 Angstrom; provenance and limits are recorded in RESULTS.md. Four FreeSASA checks remain unrun: a Windows installation attempt reached compilation but failed for lack of Microsoft Visual C++ 14.0 or greater. This is an environmental block, not a failed accessibility comparison. Reproduction commands are in RESULTS.md. The supplementary Dunbrack check does not change the original historical suite count or bundled reference tables. Bio.PDB and AccessFold use similar sampling grids, so their agreement supplies less algorithmic independence than the PyMOL comparison.

## Tested inputs and settings

Only chain A of the first model of 1UBQ, 1CRN, and 1LYZ was used. RCSB initial release dates are 1987-04-16, 1981-07-28, and 1977-04-12 respectively, strictly before the project's 2021-09-30 cutoff. These are development fixtures and must be excluded from final evaluation. Stage 1 homology-cluster exclusions cannot be reconciled until its split is available. Downloaded file hashes and release dates are in manifest.json.

The cleaned fixtures contain protein heavy atoms, omit waters, ligands, and other chains, and resolve alternate locations through the recorded preparation procedure. Calculations therefore describe isolated-chain accessibility, not complete complexes or crystal contacts. Inspect the original structure before applying this preparation policy to a new system.

Real-structure AccessFold checks use element-based Bondi radii and 4000 sphere points. PyMOL uses identical element radii, dot_solvent enabled, and dot_density 4. Standard probes are 1.4, 2.5, 4.0, and 6.0 Angstrom. Follow-up probes are 0.0, 0.5, 1.0, and 1.4 Angstrom; zero radius is diagnostic. Reference winners are recomputed with 10000 points. Relative SASA uses the matching residue-type and probe-radius reference, with no clipping. Smaller-probe follow-ups compare absolute SASA because corresponding relative references are not bundled.

## Manual interpretation

There are 36 recorded observations across the three proteins. The regression requires at least three buried and three exposed examples per protein, then checks that every buried score is below every exposed score within that protein. It passes with the current labels. This is an ordering check, not a validated binary threshold or estimate of classification accuracy.

Crambin residues 13 and 32 were initially ambiguous and later classified buried overall by the user after score disclosure. Their original observations and reclassification history are retained. These follow-ups are not independent blind labels. Residue selection was conversational rather than random; class proportions do not estimate the proteins' overall exposure distributions.

Lysozyme 25, 108, and 98 and crambin 16 and 30 were visually exposed but had relatively low water-probe accessibility. Follow-up sweeps and reinspection support qualified descriptions of partial or probe-dependent access. Crambin 30 retains area even with a 6.0 Angstrom probe, so exclusion of large probes does not fully explain that case. Surface appearance, camera occlusion, and local sphere displays are not direct measurements of solvent accessibility. Original visual exposed labels are retained with observations, rather than changed to fit the numerical scores.

The planned representative screenshot collection is complete: two views each for ubiquitin residues 48 and 15, crambin residues 26 and 7, and lysozyme residues 17 and 2 (12 images). Files are saved in screenshots and indexed with provenance and image limitations. Residue identity relies on user identification and prepared-session context because labels are not readable. Transparency and cropping limit independent classification from the images. Session files and written observations are retained. Earlier temporary image paths checked during consolidation were unavailable. These views support the recorded manual inspection; they do not constitute independent verification of all 36 labels.

## Limits and future use

Proceed with synthetic accessibility experiments using consistent preparation, probe radius, radii, reference tables, and missing-residue policy for truth and candidates. Keep this validation set in development. Preserve parameters alongside generated measurements; inspect terminal residues and RSA above one instead of assuming RSA is a probability bounded by one.

The tests reproduce and independently measure the stored reference-winning conformers. They do not prove that the search found global physical maxima. Steric thresholds, rotamer sampling, peptide geometry, and finite sphere sampling remain modeling choices. Side-chain reference maxima were not independently reconstructed in this step. This validation does not establish that synthetic SASA predicts CpK accessibility or experimental labeling.

The SASA calculation tests local geometric occlusion; it does not explicitly establish a probe's connected path from bulk solvent to every surface point. Do not interpret a probe sweep as proof of a specific entry route.

Stage 1 integration must verify PDB-to-sequence numbering, insertion codes, unresolved residues, incomplete atom sets, alternate conformations, nonstandard residues, and the intended chain or assembly context. The current adapter's numbering policy is provisional. Recheck train/development/test membership and homology separation against the final Stage 1 manifest. No Stage 1 integration pass is claimed here.

## Reproduction

Run from stage2 after installing the package and its test dependencies into your chosen Python environment:

```powershell
python -m pytest validation/test_structures.py validation/test_step2.py -q -p no:cacheprovider
```

The files required by these tests are already saved. Regeneration scripts and PyMOL instructions are in README.md. RESULTS.md contains the detailed follow-up history.
