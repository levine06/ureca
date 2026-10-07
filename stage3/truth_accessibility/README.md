# Stage 3: Truth Accessibility

This folder calculates synthetic reference accessibility from experimental PDB structures listed in Stage 1
dataset CSVs. It contains a self-contained AccessFold package snapshot used by the Stage 3 truth pipeline;
the repository's `stage2/` folder is unchanged. "Truth" means calculated from the experimental coordinate
model, not measured accessibility in solution. Candidate generation and RMSD evaluation are separate steps.

## What it does (data flow)

    AtomicStructure  coords [A,3], elements [A], atom_names [A], residue_index [A], residue_names [N]
      -> radii_from_elements                      radii.py     Bondi radii, heavy atoms only (named, recorded)
      -> atom_sasa(coords, radii, probe, n_pts)   sasa.py      Shrake-Rupley: points on each atom's inflated sphere
                                                               (r_vdW + probe); a point is exposed if it lies outside
                                                               every other atom's inflated sphere; area = 4*pi*R^2 * fraction
      -> sum per residue (total, side chain)      reference.py
      -> divide by ReferenceTable[aa, probe]      reference.py max ASA of that residue type AT THAT probe radius
      -> AccessibilityResult                      compute.py   .y, .valid_mask, absolute, relative, sidechain_*, mask, metadata

Where the per-radius maximum comes from (`tripeptide.py`, after Tien et al. 2013, PLoS ONE 8:e80635): for each
amino acid X, build Gly-X-Gly tripeptides, sweep phi/psi of X (10 deg grid) and its side-chain chi angles, drop
sterically impossible conformers, compute the SASA of X, and take the **maximum**. Tables for probe radii
1.4 / 2.5 / 4.0 / 6.0 A ship in `src/accessfold/data/references/` and are registered as the defaults on import.

## Usage

    from accessfold import compute_accessibility, compute_accessibility_profile
    r = compute_accessibility(structure, method="relative_sasa", probe_radius=1.4)     # 2A
    r.y, r.valid_mask                                       # headline vector (NaN where masked / undefined)
    prof = compute_accessibility_profile(structure, (1.4, 2.5, 4.0, 6.0))              # 2B: same structure, 4 definitions

    compute_accessibility(s, "absolute_sasa", 3.0)          # always works
    compute_accessibility(s, "relative_sasa", 3.0)          # ValueError: no table for 3.0 A (never falls back to 1.4)
    from accessfold import build_tripeptide_reference, register_reference
    register_reference(build_tripeptide_reference(3.0), default=True)   # needs PeptideBuilder + biopython, ~10 min single-core

Structures enter through `AtomicStructure` (array level; sequence-position indexing, unresolved residues are
legal and masked). `structures/adapters.py:from_biotite` is a **provisional** converter; replace its numbering
policy when Stage 1 fixes the PDB-numbering <-> SEQRES map. `structure.with_coords(x)` swaps coordinates on a fixed
topology (candidates, diffusion intermediates).

Rebuild the tables: `python scripts/build_reference_tables.py --workers 2` (~13 min on 2 cores).
Tests: `pip install -e .[test] && pytest` (~1 min).

## Conventions that must stay identical between truth and candidates

heavy atoms only; waters/ligands/other chains removed (isolated monomer); residues indexed by sequence
position; same `probe_radius`, `n_points`, `radii_table`, `gap_flank`. `AccessibilityResult.metadata` records them.
Unresolved residues: NaN and masked; `gap_flank=k` also masks k neighbours of *internal* gaps (a big probe is
affected over a longer range than a water-sized one, so consider a larger k at large radii); the alternative for
Stage 3 is to delete the same residues from each candidate before computing.

## Deviations from Tien et al. 2013 (all recorded in each table's `extra`)

| Paper | Here | Why |
|---|---|---|
| Dunbrack rotamer library, random subsample of 10 for residues with >10 rotamers | staggered chi grid (+60/180/-60; 30 deg grid for planar chi), **all** points evaluated | the registration-gated library was not available when the tables were built; the statistic is a maximum, so only allowed chi values matter. `DunbrackLibrary` now reads the library; using it changes A_max by <= 1.7 % (see below). |
| bond lengths/angles mined from 3197 PDB structures | PeptideBuilder 1.1.0 defaults (same lab; averages over crystal structures) | user-chosen. Not verified to be the identical dataset. |
| DSSP (its radii and algorithm) | this package's Shrake-Rupley + Bondi radii | one engine for numerators and denominators |
| "biophysically allowed" conformers (criterion not given in the text I read) | hard-sphere filter: reject if d < 0.80 (r_i + r_j) for atoms >= 4 bonds apart (C...C closer than 2.72 A) | user-chosen; permissive on purpose (a maximum, not a typical value). See sensitivity below |
| exact maximum | maximum re-evaluated for the top 24 candidates at 10 000 points | a maximum over ~10^5 noisy estimates is biased upward |

## Measured behaviour (see also the test suite)

* **1.4 A vs the published table, all 20 residues (clash_scale 0.80):** -4.6 % ... +1.4 %, mean |diff| 2.4 %, Pearson 0.997, Spearman 0.995.
  Ala 130.5 (129), Gly 105.4 (104), Trp 279.5 (285), Arg 265.4 (274).
* **Steric threshold is the largest uncertainty.** Going from 0.85 to the shipped 0.80 raised A_max by +0.6 % (1.4 A) to +0.7 % (6.0 A)
  on average (at most +1.7 %). Changing `clash_scale` from 0.75 to 0.95 changes A_max by
  5.4 % on average at 1.4 A (3.1-8.1 %) and 9.0 % at 6.0 A (6.7-12.3 %). For nearly every residue the maximising backbone sits near
  (phi, psi) ~ (70, -70), at the edge of the allowed region, which is why. Grid step 10 -> 30 deg changes A_max by
  0.6 % (1.4 A) and 1.1 % (6.0 A) on average; worst cases 3.8 % (1.4 A, Val) and 5.0 % (6.0 A, Gly).
* **Orientation noise of the numerical SASA** (20-residue peptide, 20 random rotations, 1000 points): total SASA repeatable to
  0.1-0.28 %; per-residue standard deviation 0.5 A^2 (1.4 A) ... 1.8 A^2 (6.0 A) median. Per-residue noise fell about 4-5x from 250 to 2000 points
  at every radius.
* A_max(6.0)/A_max(1.4) = 3.05-3.73; all tables are strictly increasing with probe radius; Gly is smallest and Trp largest at every radius.
* `top_m` 24 vs 256 gave identical maxima for Arg, Leu, Trp (step 20, radii 1.4 and 6.0).

## Verification performed

Analytic checks (isolated sphere, two-sphere spherical-cap formula, fully buried atom); KD-tree neighbour search vs
brute force at probe 0-9 A; translation/permutation invariance; agreement with `biotite.structure.sasa` and with
`Bio.PDB.SASA.ShrakeRupley` (independent implementations) at all four radii; chi rotations hit their targets exactly and preserve bonds
for 19 residue types; batched SASA identical to the main engine; non-finite conformers rejected (PeptideBuilder emits NaN for Met at
phi=0, psi=90); scan order invariance; an independent read-only code review found no numerical bugs.
FreeSASA's Lee-Richards (1000 slices, identical Bondi radii on both sides) is used as a near-exact reference: total-area error
+0.06 / -0.24 / +0.07 / +0.16 % at 1.4 / 2.5 / 4.0 / 6.0 A with 1000 points (within 0.06 % at 4000 points); worst single-atom difference 0.6-2.5 A^2.
Install with `pip install freesasa --use-pep517` (the legacy `setup.py` path fails on Debian's patched setuptools); the test skips if it is absent.

## Stage 3 step 1: accessibility of the true structures

`python scripts/compute_truth_accessibility.py --csv development.csv validation.csv --out truth_out`
(see the script header for the download / SLURM-array / collect workflow). For each row it fetches the mmCIF file, loads the chain
`label_chain_id` with `accessfold.structures.mmcif.load_chain_from_mmcif` (sequence-position indexing from `label_seq_id`; isolated
monomer; first model; highest-occupancy altlocs; hydrogens, waters, ligands, other chains dropped; residue names taken from the CSV sequence),
and calls `compute_accessibility` at 1.4 / 2.5 / 4.0 / 6.0 A (`accessfold.truth.truth_arrays`). Saved per entry: `abs_/rel_/sc_abs_/sc_rel_<radius>`,
`mask`, `resolved`, `complete`, `gap_distance`, QC metadata. Residues with a partly modelled side chain are masked (their atoms still occlude);
`mask` must be applied to the candidate structures too. Unresolved residues are NaN.
Masking policy (v2): unresolved | partly modelled | chemically modified or different from the sequence residue (selenomethionine is the only
exception; the identity, parent and extra atoms are recorded) | "shadow" residues whose area changes by more than `--shadow-threshold` (5 A^2) at any
radius when a modification's extra atoms are removed. Coverage is reported three ways (coordinate records, usable atoms, Stage 1's number) with a warning
when they differ; zero-occupancy atoms are never used.

The updated pipeline was run on the CPU cluster for 30 development and 15 validation chains at all four radii.
The resulting summary confirms CYQ and five shadow exclusions in 2ID7, and distinguishes 165 coordinate-record
positions from 160 usable positions in 3RF2. All 45 chains pass the 90% usable-coordinate coverage threshold.
Test A and Test B have not been run. Generated cluster outputs are separate from the code in this folder.

## Not done / open

(Stage 2C, the CpK rotamer forward model, is deliberately outside this deliverable; `register_method` is the only hook for it.)

* Earlier manual PyMOL validation covered ubiquitin, crambin and lysozyme; see `../../stage2/validation/`.
  The 45-chain dataset run was checked through QC summaries, not visual inspection of every chain.
* Missing regions contribute no shielding. The 5 A^2 shadow threshold is a modelling choice, and alternate-location
  ties use the first listed coordinates. Calculations use an isolated chain and the first model; these assumptions
  must remain explicit when comparing with candidate structures.
* **Dunbrack library:** `DunbrackLibrary(path_to_ALL.bbdep.rotamers.lib)` (Simple Mode, `accessibility/dunbrack.py`) is a ready `RotamerSource`
  (mean chi only, probability floor 0.01, not bundled; ODC-BY, cite Shapovalov & Dunbrack 2011, Structure 19:844). The bundled tables still use the
  staggered grid. Full 10-degree scan of all 20 residues with the library vs the bundled tables: mean +0.06 % (1.4 A) and -0.17 % (6.0 A);
  range -1.7 % (Leu) ... +0.4 % at 1.4 A and -0.5 % ... +0.6 % at 6.0 A. Probability floor 0 / 0.01 / 0.05 (Arg, Lys, Leu, Phe, Trp, Met, Gln): identical
  except Leu at 1.4 A (201.9 / 198.1 / 197.5) and Gln at 1.4 A with floor 0.05 (-0.3 %). Rebuild with `build_tripeptide_references(..., source=DunbrackLibrary(path))`
  if you prefer library-derived tables; `tests/test_dunbrack.py` has an opt-in real-file test (`ACCESSFOLD_DUNBRACK_LIB=...`).
* `clash_scale` = 0.80 is a documented modelling choice, not a validated one; a lower value gives a looser (larger) A_max, a higher one a tighter A_max, and real residues can then exceed relative accessibility 1.
* Known limitations of the generator: Arg chi4 on the sp3 grid (finer sweep +0.6 % at 6 A); Pro keeps a rigid ring; flanking Gly backbone fixed at phi=-120, psi=140; no OXT.
