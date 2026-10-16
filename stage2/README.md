# Stage 2: Accessibility Package

`accessfold` calculates conventional solvent-accessible surface area (SASA)
per residue using Shrake-Rupley sampling and Bondi atomic radii, and
normalizes it to a relative accessibility using Gly-X-Gly maximum-area tables.

## Contents

- [Deliverables](#deliverables)
- [Usage](#usage)
- [Package layout](#package-layout)
- [How accessibility is computed](#how-accessibility-is-computed)
  - [1. Input structure](#1-input-structure)
  - [2. Atomic radii](#2-atomic-radii)
  - [3. Per-atom SASA (Shrake-Rupley)](#3-per-atom-sasa-shrake-rupley)
  - [4. Per-residue and side-chain sums](#4-per-residue-and-side-chain-sums)
  - [5. Relative SASA](#5-relative-sasa)
  - [6. Masking and the result object](#6-masking-and-the-result-object)
  - [7. Probe-radius profiles](#7-probe-radius-profiles)
- [How the reference (maximum ASA) tables are built](#how-the-reference-maximum-asa-tables-are-built)
- [Bundled reference values](#bundled-reference-values)
- [Settings that must match](#settings-that-must-match)
- [Validation](#validation)
- [Reference conventions and limitations](#reference-conventions-and-limitations)

## Deliverables

- `src/accessfold/`: the Python package and its structure adapters.
- `src/accessfold/data/references/`: bundled Gly-X-Gly normalization tables
  for probe radii 1.4, 2.5, 4.0 and 6.0 Angstrom.
- `validation/`: recorded real-structure validation measurements,
  manual inspection labels, structures, PyMOL sessions and screenshot evidence.

Correctness checks were run during development, including checks on manually
inspected exposed and buried residues. Their recorded evidence is retained;
test scripts and regeneration helpers are archived outside the repository.

## Usage

Install from this folder:

```bash
python -m pip install -e .
```

```python
from accessfold import compute_accessibility, compute_accessibility_profile

result = compute_accessibility(
    structure, method="relative_sasa", probe_radius=1.4
)
absolute = result.absolute
relative = result.relative
sidechain_absolute = result.sidechain_absolute
sidechain_relative = result.sidechain_relative

profiles = compute_accessibility_profile(structure, (1.4, 2.5, 4.0, 6.0))
```

Absolute areas are in square Angstrom; relative values are unitless ratios
to the corresponding residue-type maximum at the same probe radius.
Glycine side-chain accessibility is undefined (NaN).
The multi-radius profiles are a sensitivity analysis, not assigned dye radii.

Inputs use `AtomicStructure`, with residues indexed by sequence position.
The optional Biotite adapter is in `structures/adapters.py`; install
`.[biotite]` to use it. Prepare the intended chain and atom environment
consistently before calculation. The calculation uses heavy atoms only.
Unresolved residues have no coordinates and are masked; their missing atoms
cannot occlude neighbouring residues. `gap_flank` optionally masks neighbours
of internal gaps. Metadata records calculation settings.

Options accepted by `compute_accessibility` (all keyword arguments):

| Option | Default | Meaning |
| --- | --- | --- |
| `method` | `"relative_sasa"` | `"relative_sasa"` (headline vector is relative SASA, needs a reference table) or `"absolute_sasa"` (headline vector is Angstrom^2) |
| `probe_radius` | `1.4` | Probe sphere radius in Angstrom (1.4 = water) |
| `n_points` | `1000` | Sample points per atom sphere |
| `radii_table` | `"bondi"` | Named atomic radii set |
| `reference` | bundled table for `probe_radius` | A `ReferenceTable` to normalize with |
| `gap_flank` | `0` | Also mask residues within this many positions of an internal gap |
| `clip` | `None` | If set, relative values are capped at this number |

## Package layout

| File | Role |
| --- | --- |
| `structures/atomic.py` | `AtomicStructure`: array container (coordinates, elements, atom names, residue index, residue names), backbone/side-chain split, `with_coords` |
| `structures/adapters.py` | Optional `from_biotite` converter (provisional numbering policy) |
| `accessibility/radii.py` | Radii tables and element-to-radius lookup |
| `accessibility/sasa.py` | `atom_sasa`: the Shrake-Rupley engine on plain NumPy arrays |
| `accessibility/reference.py` | `ReferenceTable`, the reference registry, `residue_sasa` (atom to residue sums), empirical reference builder |
| `accessibility/compute.py` | `compute_accessibility`, `compute_accessibility_profile`, masking, `AccessibilityResult`, method registry |
| `accessibility/tripeptide.py` | Gly-X-Gly conformer scan that produces the maximum-ASA tables |
| `accessibility/dunbrack.py` | Optional Dunbrack rotamer source for the tripeptide scan |
| `data/references/*.json` | The four bundled tables, loaded and registered on import |

## How accessibility is computed

The pipeline for one structure at one probe radius is:

```
AtomicStructure -> radii per atom -> per-atom SASA -> sum per residue
                -> divide by Gly-X-Gly maximum for that residue type -> relative SASA
```

### 1. Input structure

`AtomicStructure` holds one chain as flat per-atom arrays:

- `coords [A, 3]`, `elements [A]`, `atom_names [A]`, and `residue_index [A]`,
  the sequence position each atom belongs to;
- `residue_names [N]`, a three-letter code for every position of the full
  sequence, including positions with no atoms.

Atoms named `N`, `CA`, `C`, `O` or `OXT` are backbone; every other atom is
side chain (`CB` and beyond). Glycine therefore has no side-chain atoms.
A residue with no atoms is *unresolved*. Coordinates and topology are kept
separate so truth structures, predicted candidates and intermediate diffusion
coordinates can share one topology via `with_coords`. Non-finite coordinates
are rejected at construction.

### 2. Atomic radii

Each atom gets a van der Waals radius from its element (`radii.py`). The only
bundled set is Bondi (1964), heavy atoms only: C 1.70, N 1.55, O 1.52,
S 1.80, Se 1.90, P 1.80 Angstrom (H is listed but hydrogens are not used).
An element missing from the table raises an error. The radii set is a
modelling choice recorded in every result and reference table.

### 3. Per-atom SASA (Shrake-Rupley)

`atom_sasa` in `sasa.py` implements the Shrake and Rupley (1973) numerical
method. For atom `i` with radius `r_i` and probe radius `p`:

1. Define the *expanded radius* `R_i = r_i + p`. The accessible surface is the
   surface traced by the probe centre, which is the sphere of radius `R_i`
   around each atom.
2. Place `n_points` quasi-uniform points on the unit sphere with a
   golden-spiral (Fibonacci) lattice: for `k = 0 .. n-1`,
   `z = 1 - 2(k + 0.5)/n`, azimuth `phi = k * pi * (3 - sqrt(5))`, and
   `rho = sqrt(1 - z^2)`. The same lattice is used for every atom, scaled by
   `R_i` and translated to the atom centre.
3. A test point is *buried* if it lies strictly inside the expanded sphere of
   any *other* atom `j`, that is `|point - c_j| < R_j`. Every atom in the
   structure occludes every other, including atoms of the same residue.
4. The atom's area is the exposed fraction of its points times the full
   expanded sphere area:

   ```
   area_i = 4 * pi * R_i^2 * (points not buried) / n_points
   ```

For speed, a `cKDTree` finds candidate neighbours within `R_i + max(R)`, and
only atoms whose spheres actually overlap (`d < R_i + R_j`) are tested. An
isolated atom gets its full sphere area. Squared distances between sample
points and neighbour centres are computed with a matrix product.

Accuracy notes:

- Area resolution is roughly `4 * pi * R^2 / n_points` per atom. The default
  is 1000 points, which keeps discretization noise near the percent level.
  The validation runs used 4000 points.
- A fixed lattice is not exactly rotation invariant. Always compare truth and
  candidate structures at the **same** `n_points`.

### 4. Per-residue and side-chain sums

`residue_sasa` in `reference.py` converts atom areas to residue areas by
summing over `residue_index` (`np.bincount`):

- `total[i]` = sum of areas of all atoms of residue `i`
- `side[i]` = sum of areas of the residue's side-chain atoms only

Residues with no atoms contribute nothing and are set to NaN in the result
(`absolute`); a residue with no side-chain atoms (glycine) gets NaN for
`sidechain_absolute`. Because unresolved residues have no atoms, they also do
not occlude their neighbours, so residues next to a gap read as more exposed
than they would be in the full chain.

### 5. Relative SASA

Relative accessibility divides the absolute area by the **maximum possible
area for that residue type at the same probe radius** (the "Gly-X-Gly"
reference, see below):

```
relative[i]            = absolute[i]            / max_asa[residue_name[i]]
sidechain_relative[i]  = sidechain_absolute[i]  / sidechain_max_asa[residue_name[i]]
```

For example, a leucine with 60 square Angstrom of 1.4 A SASA has relative
SASA `60 / 201.5 = 0.30` against the bundled table.

Details:

- The reference is looked up by probe radius. If none is registered for the
  requested radius, `relative_sasa` raises an error instead of silently reusing
  a 1.4 A table (`absolute_sasa` still works).
- A requested radius must match the table's radius (checked with `isclose`).
- Residue names absent from the table (for example `UNK` or non-standard
  residues) give NaN, and are listed in `metadata["unreferenced_residues"]`.
- Side-chain relative values are only produced when the table contains
  side-chain maxima; the bundled tables do, except for glycine.
- Relative values are **not** capped by default and can exceed 1. Reasons
  include chain termini (the reference has no `OXT` and has flanking glycines
  on both sides), the 0.80 clash scale in the reference scan, and real
  structures sampling geometry outside the scanned grid. Pass `clip=1.0` (or
  another value) to cap them.
- `reference.n_points` is deliberately not compared with the call's
  `n_points`: tables are built at higher resolution (10000 points) than routine
  calls because a maximum over many noisy estimates is biased upward. Only a
  differing radii table sets `metadata["reference_config_mismatch"] = True`
  (reported, not fatal).

### 6. Masking and the result object

`compute_accessibility` returns an `AccessibilityResult` whose arrays are all
length `N` (the full sequence):

| Field | Meaning |
| --- | --- |
| `absolute` | Residue SASA in Angstrom^2; NaN for unresolved residues |
| `relative` | `absolute / max_asa`; NaN if no reference value |
| `sidechain_absolute` | Side-chain-atom SASA; NaN for glycine and unresolved |
| `sidechain_relative` | Side-chain SASA over side-chain maximum |
| `mask` | Residue is resolved and not within `gap_flank` of an internal gap |
| `y` | Headline vector (`relative` for `relative_sasa`, `absolute` for `absolute_sasa`) with masked residues set to NaN |
| `valid_mask` | `isfinite(y)`: residues usable in an error metric |
| `metadata` | `n_points`, `radii_table`, `gap_flank`, `clip`, `reference` name, `unreferenced_residues`, optional `reference_config_mismatch` |

`values(kind)` returns any of the four arrays with the mask applied.

**Gap masking.** With `gap_flank = k > 0`, a resolved residue is masked if an
unresolved residue lies within `k` sequence positions of it *inside* the chain.
Unresolved residues at the N- or C-terminal ends are treated as truncation, not
gaps, and do not trigger masking. `gap_flank = 0` masks only unresolved
residues themselves.

New forward models can be added with `register_method(name)` without
changing callers; the two built-ins are `absolute_sasa` and `relative_sasa`.

### 7. Probe-radius profiles

`compute_accessibility_profile(structure, (1.4, 2.5, 4.0, 6.0))` runs the same
calculation at several radii and returns `{radius: AccessibilityResult}`. Each
radius uses its own reference table (override with `references={radius: table}`).
Larger probes ignore fine surface roughness and report coarser, more
shape-level exposure. This is a sensitivity analysis, not an assignment of
dye radii.

## How the reference (maximum ASA) tables are built

Relative SASA needs a per-residue-type denominator. The bundled tables are
generated by `accessibility/tripeptide.py` following the idea of Tien et al.
(2013): the maximum accessible area of residue X is its largest SASA over
every sterically allowed conformation in a **Gly-X-Gly** tripeptide.

For each of the 20 residue types and each probe radius:

1. **Build conformers.** PeptideBuilder builds Gly-X-Gly with default bond
   lengths and angles. The backbone (phi, psi) of X is swept over a 10-degree
   grid from -180 to 170 in both angles (36 x 36 = 1296 backbone states). The
   flanking glycines stay at PeptideBuilder's default backbone angles
   (phi = -120, psi = 140) and the C-terminal residue has no `OXT`.
2. **Enumerate side-chain rotamers.** For each backbone state, the side-chain
   chi angles are set to every combination from a library-free *staggered
   grid* (`StaggeredChiGrid`):
   - sp3-sp3 chi angles take +60, 180, -60 degrees;
   - chi angles into a planar, non-symmetric group (Asn chi2, Gln chi3,
     His chi2, Trp chi2) use a 30-degree grid over the full circle;
   - chi angles into a planar, 2-fold symmetric group (Asp, Phe, Tyr chi2;
     Glu chi3) use a 30-degree grid over 180 degrees.

   Atoms are rotated about each chi bond (Rodrigues rotation) from a fixed
   clash-free reference conformer's connectivity. Gly and Ala have no chi
   angles and proline keeps its rigid ring geometry. Every grid point is
   evaluated; there is no random subsampling. Because the statistic is a
   maximum, rotamer populations do not matter, only which chi values are
   allowed. An optional `DunbrackLibrary` can replace the grid.
3. **Reject impossible conformers.** A conformer is discarded if any pair of
   atoms at least four bonds apart is closer than `clash_scale * (r_i + r_j)`
   (default `clash_scale = 0.80`), or if its coordinates are non-finite
   (PeptideBuilder emits NaN for some degenerate angles; NaN would otherwise
   look clash-free and fully exposed).
4. **Coarse SASA.** For surviving conformers, the Shrake-Rupley area of every
   atom of the *middle residue X* is computed (same algorithm and lattice as
   `atom_sasa`, batched, float32, 1000 points). All atoms of the tripeptide,
   including the flanking glycines, occlude. The conformer's score is the sum
   over residue X (and, separately, over its side-chain atoms).
5. **Refine the maximum.** Taking a maximum over many noisy estimates is
   biased upward, so the top 24 conformers (by total, and by side-chain area)
   are re-evaluated with 10000 points in float64. The reported maximum
   (`max_asa`) is the largest refined total; `sidechain_max_asa` is the largest
   refined side-chain area among those candidates.
6. **Record.** One table per probe radius is written as JSON, with the
   configuration, per-residue counts of rejected conformers, coarse maximum,
   99th percentile, and the argmax (phi, psi, chi) in `extra`.

Each residue is scanned once for all requested radii. The default maxima sit
near (phi, psi) of about (70, -70), at the edge of what the hard-sphere filter
allows, so values depend on `clash_scale`.

`build_empirical_reference` offers an alternative: the 99th percentile (by
default) of observed residue areas, grouped by residue type, over a set of
structures. Types with fewer than `min_count = 30` examples are omitted and
yield NaN relative values. Use development-split structures only so test
structures do not leak into the normalization.

Reference registry (`reference.py`): the bundled JSON tables are loaded and
registered on import as the default for their probe radius (a duplicate radius
in the directory raises an error). The literature table from Tien et al.
(`tien2013_theoretical`, 1.4 A, computed with DSSP) is registered by name for
comparison only and is not the default, since mixing its denominators with
this package's numerators would mix two SASA implementations. Use
`register_reference(table, default=True)` to add your own and
`get_reference(radius, name=...)` to retrieve one.

## Bundled reference values

Whole-residue maximum ASA (Angstrom^2) in the bundled tables
(`gxg_staggered_chi_grid_v1`, Bondi radii, 10000 refinement points):

| Residue | 1.4 A | 2.5 A | 4.0 A | 6.0 A | | Residue | 1.4 A | 2.5 A | 4.0 A | 6.0 A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALA | 130.5 | 194.6 | 300.2 | 472.6 | | LEU | 201.5 | 287.2 | 427.6 | 661.3 |
| ARG | 265.4 | 374.1 | 546.0 | 825.0 | | LYS | 227.2 | 324.0 | 480.2 | 731.6 |
| ASN | 186.4 | 269.0 | 404.6 | 626.6 | | MET | 220.6 | 314.5 | 467.7 | 715.7 |
| ASP | 184.7 | 266.6 | 401.6 | 621.5 | | PHE | 238.5 | 335.8 | 493.9 | 750.9 |
| CYS | 165.2 | 241.1 | 365.8 | 569.2 | | PRO | 157.2 | 231.5 | 354.3 | 555.5 |
| GLN | 215.4 | 308.1 | 459.3 | 703.5 | | SER | 147.9 | 217.7 | 332.8 | 521.4 |
| GLU | 213.5 | 305.4 | 455.9 | 698.3 | | THR | 169.4 | 246.2 | 372.5 | 578.7 |
| GLY | 105.4 | 158.3 | 248.1 | 395.5 | | TRP | 279.5 | 389.3 | 567.3 | 854.1 |
| HIS | 216.6 | 308.0 | 457.6 | 700.1 | | TYR | 255.3 | 358.1 | 524.1 | 790.9 |
| ILE | 198.9 | 284.9 | 426.3 | 657.3 | | VAL | 175.5 | 254.2 | 383.5 | 596.7 |

Side-chain maxima are in the same JSON files (`sidechain_max_asa`), for every
residue except glycine.

## Settings that must match

A relative value is only meaningful when numerator and denominator come from
the same configuration:

- **Probe radius:** enforced. A table is only valid for the radius it was built at.
- **Radii table:** should match; a mismatch is flagged in the metadata.
- **`n_points`:** pick one value and use it for both truth and candidates.
- **Chain and atom preparation:** heavy atoms only, one intended chain, no
  mixed altlocs. Atoms present in the structure (including other residues in the
  same chain) are the only occluders; ligands, waters and other chains are not
  included unless you add them to the structure.

## Validation

See [validation/README.md](validation/README.md) for results and limitations.
Recorded checks cover three manually inspected structures (1UBQ, 1CRN, 1LYZ),
independent PyMOL measurements at all four radii, smaller-probe comparisons,
and recorded reference-conformer areas.

## Reference Conventions and Limitations

Bundled normalization tables use Gly-X-Gly conformer maxima following the
approach of Tien et al. (2013, PLoS ONE 8:e80635), with PeptideBuilder geometry,
a staggered side-chain grid and a 0.80 clash scale. These choices are not an
exact reproduction of the published method (the paper uses its own mined
geometry, the Dunbrack library with random subsampling, and DSSP); relative
values can exceed one.
The core needs only NumPy and SciPy. Reference-building functions remain part
of the package API; `.[reference-builder]` supplies their optional dependencies
(PeptideBuilder, Biopython).

`DunbrackLibrary` is an optional rotamer source, not the source of the bundled
tables. It uses mean chi angles, a 10-degree backbone grid and a default 0.01
probability floor. The external library is not redistributed; cite Shapovalov
and Dunbrack (2011, Structure 19:844) when using it.

The reference generator uses a rigid proline ring, fixed flanking Gly
backbones and no OXT. The refined maximum covers only the top 24 coarse
candidates, so it can be biased low by a fraction of a percent. Arg chi4
(planar) is sampled on the sp3 grid. Manual validation is a consistency check,
not a blinded accuracy study or independent validation of side-chain
normalization. Experimental-structure and candidate batch workflows belong to
Stage 3.
