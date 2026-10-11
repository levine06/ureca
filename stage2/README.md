# Stage 2: Accessibility Package

`accessfold` calculates conventional solvent-accessible surface area (SASA)
per residue using Shrake-Rupley sampling and Bondi atomic radii.

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

## Validation

See [validation/README.md](validation/README.md) for results and limitations.
Recorded checks cover three manually inspected structures (1UBQ, 1CRN, 1LYZ),
independent PyMOL measurements at all four radii, smaller-probe comparisons,
and recorded reference-conformer areas.

## Reference Conventions and Limitations

Bundled normalization tables use Gly-X-Gly conformer maxima following the
approach of Tien et al. (2013, PLoS ONE 8:e80635), with PeptideBuilder geometry,
a staggered side-chain grid and a 0.80 clash scale. These choices are not an
exact reproduction of the published method; relative values can exceed one.
The core needs only NumPy and SciPy. Reference-building functions remain part
of the package API; `.[reference-builder]` supplies their optional dependencies.

`DunbrackLibrary` is an optional rotamer source, not the source of the bundled
tables. It uses mean chi angles, a 10-degree backbone grid and a default 0.01
probability floor. The external library is not redistributed; cite Shapovalov
and Dunbrack (2011, Structure 19:844) when using it.

The reference generator uses a rigid proline ring, fixed flanking Gly
backbones and no OXT. Manual validation is a consistency check, not a blinded
accuracy study or independent validation of side-chain normalization.
Experimental-structure and candidate batch workflows belong to Stage 3.
