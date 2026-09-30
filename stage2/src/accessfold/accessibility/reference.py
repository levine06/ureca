"""Per-amino-acid maximum accessible surface area, needed to turn absolute SASA into relative SASA.

A maximum-ASA table is only valid for the (probe radius, radii set, algorithm settings) it was
computed with. In particular the literature tables are for a 1.4 A probe: at 2.5/4.0/6.0 A the
table must be rebuilt. Two builders exist:

* `accessfold.accessibility.tripeptide.build_tripeptide_references` - theoretical Gly-X-Gly scan after
  Tien et al. 2013 (the default; tables for probe radii 1.4/2.5/4.0/6.0 A ship with the package in
  `accessfold/data/references/` and are registered on import).
* `build_empirical_reference` - high percentile of observed residue ASA over a set of structures
  (use the DEVELOPMENT split only, to avoid leaking test structures into the normalisation).

The literature table itself (`TIEN2013_THEORETICAL_1P4`, computed by the paper with DSSP) stays registered
under its own name for comparison, but is NOT the default: mixing its denominators with this package's
numerators would mix two SASA implementations.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from accessfold.accessibility.radii import radii_from_elements
from accessfold.accessibility.sasa import DEFAULT_N_POINTS, atom_sasa
from accessfold.structures.atomic import AtomicStructure

STANDARD_RESIDUES = (
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)

# Tien et al. 2013, PLoS ONE 8(11):e80635, Table 1, "theoretical" column (A^2, probe 1.4 A, DSSP).
# Checked against the published Table 1 (all 20 values). Derived with the paper's own radii/algorithm,
# so expect approximate, not exact, agreement with this package's Bondi radii.
_TIEN_THEORETICAL = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


@dataclass(frozen=True)
class ReferenceTable:
    """Maximum ASA per residue type for one specific SASA configuration.

    max_asa:           residue name -> Angstrom^2 (whole residue)
    sidechain_max_asa: optional residue name -> Angstrom^2 (side-chain atoms only)
    """

    name: str
    probe_radius: float
    max_asa: Mapping[str, float]
    provenance: str
    radii_table: Optional[str] = None
    n_points: Optional[int] = None
    sidechain_max_asa: Optional[Mapping[str, float]] = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def lookup(self, residue_names: np.ndarray, sidechain: bool = False) -> np.ndarray:
        """[N] max-ASA values for [N] residue names; NaN for residues absent from the table."""
        table = self.sidechain_max_asa if sidechain else self.max_asa
        if table is None:
            return np.full(len(residue_names), np.nan)
        return np.array([table.get(str(r), np.nan) for r in residue_names], dtype=np.float64)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> "ReferenceTable":
        return cls(**json.loads(Path(path).read_text()))


TIEN2013_THEORETICAL_1P4 = ReferenceTable(
    name="tien2013_theoretical",
    probe_radius=1.4,
    max_asa=_TIEN_THEORETICAL,
    provenance="Tien et al. 2013 PLoS ONE 8:e80635, Table 1 theoretical column (DSSP, Gly-X-Gly).",
    radii_table=None,
)

_REGISTRY: dict[tuple[str, float], ReferenceTable] = {}
_DEFAULTS: dict[float, str] = {}


def register_reference(table: ReferenceTable, default: bool = False) -> None:
    """Register a table; `default=True` makes it the automatic choice for its probe radius."""
    _REGISTRY[(table.name, float(table.probe_radius))] = table
    if default or float(table.probe_radius) not in _DEFAULTS:
        _DEFAULTS[float(table.probe_radius)] = table.name


def get_reference(probe_radius: float, name: Optional[str] = None) -> ReferenceTable:
    """Look up a registered reference table; raises with a clear message if none exists."""
    r = float(probe_radius)
    if name is None:
        name = _DEFAULTS.get(r)
        if name is None:
            raise ValueError(
                f"no maximum-ASA reference registered for probe radius {r} A. Build one with "
                "accessfold.accessibility.tripeptide.build_tripeptide_reference(probe_radius) (or "
                "build_empirical_reference(...)) and register_reference(...), or use "
                "method='absolute_sasa'. Do NOT reuse a 1.4 A table at another radius."
            )
    try:
        return _REGISTRY[(name, r)]
    except KeyError as exc:
        raise ValueError(f"no reference named {name!r} for probe radius {r} A") from exc


BUNDLED_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "references"


def load_bundled_references(directory: Optional[Path] = None) -> list:
    """Register every `*.json` table in the bundled directory as the default for its probe radius.

    Returns the list of registered tables (empty if none have been built yet).
    """
    directory = Path(directory) if directory is not None else BUNDLED_REFERENCE_DIR
    loaded, seen = [], {}
    for path in sorted(directory.glob("*.json")):
        table = ReferenceTable.from_json(path)
        key = float(table.probe_radius)
        if key in seen:  # never let filename order silently decide which table is the default
            raise ValueError(f"two bundled reference tables for probe radius {key} A: {seen[key]} and {path.name}")
        seen[key] = path.name
        loaded.append(table)
    for table in loaded:  # register only after the whole directory validated
        register_reference(table, default=True)
    return loaded


# Bundled Gly-X-Gly tables first (they become the defaults); the literature table is registered by name
# only, and becomes the 1.4 A default solely if no bundled table exists.
load_bundled_references()
register_reference(TIEN2013_THEORETICAL_1P4)


def residue_sasa(structure: AtomicStructure, probe_radius: float, n_points: int = DEFAULT_N_POINTS,
                 radii_table: str = "bondi") -> tuple[np.ndarray, np.ndarray]:
    """Per-residue (total, side-chain) SASA sums, each [N]; unresolved residues are 0 / NaN-free here."""
    radii = radii_from_elements(structure.elements, radii_table)
    per_atom = atom_sasa(structure.coords, radii, probe_radius, n_points)
    n = structure.n_residues
    total = np.bincount(structure.residue_index, weights=per_atom, minlength=n)
    side = np.bincount(structure.residue_index, weights=per_atom * structure.is_sidechain, minlength=n)
    return total, side


def build_empirical_reference(
    structures: Iterable[AtomicStructure],
    probe_radius: float,
    percentile: float = 99.0,
    min_count: int = 30,
    n_points: int = DEFAULT_N_POINTS,
    radii_table: str = "bondi",
    name: Optional[str] = None,
) -> ReferenceTable:
    """Percentile of observed per-residue ASA, grouped by residue type.

    Residue types seen fewer than `min_count` times are left out (their RSA will be NaN and is
    reported by `compute_accessibility`) rather than estimated from too few examples.
    Use development-split structures only.
    """
    total_by: dict[str, list[float]] = {}
    side_by: dict[str, list[float]] = {}
    for s in structures:
        total, side = residue_sasa(s, probe_radius, n_points, radii_table)
        resolved = s.resolved_mask
        has_side = np.bincount(s.residue_index, weights=s.is_sidechain.astype(float), minlength=s.n_residues) > 0
        for i in np.flatnonzero(resolved):
            total_by.setdefault(str(s.residue_names[i]), []).append(total[i])
            if has_side[i]:
                side_by.setdefault(str(s.residue_names[i]), []).append(side[i])

    max_asa = {r: float(np.percentile(v, percentile)) for r, v in total_by.items() if len(v) >= min_count}
    side_max = {r: float(np.percentile(v, percentile)) for r, v in side_by.items() if len(v) >= min_count}
    skipped = sorted(set(total_by) - set(max_asa))
    if skipped:
        warnings.warn(f"residue types with < {min_count} examples omitted from reference: {skipped}")
    return ReferenceTable(
        name=name or f"empirical_p{percentile:g}",
        probe_radius=float(probe_radius),
        max_asa=max_asa,
        sidechain_max_asa=side_max,
        provenance=f"empirical {percentile:g}th percentile over {sum(len(v) for v in total_by.values())} residues",
        radii_table=radii_table,
        n_points=n_points,
    )
