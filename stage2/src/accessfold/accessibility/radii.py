"""Atomic van der Waals radii (Angstrom) used to build the solvent-accessible surface.

The radii set is a *modelling choice* that changes every SASA value slightly, so it is named and
recorded in every result and reference table. Default: element-based Bondi radii (Bondi 1964),
heavy atoms only. Note that united-atom "ProtOr" radii (used by FreeSASA/Rose-style analyses) or
the radii behind published maximum-ASA tables differ; do not mix a reference table with a
different radii set without re-validating (see tests/test_reference.py).
"""

from __future__ import annotations

import numpy as np

RADII_TABLES = {
    "bondi": {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90, "P": 1.80},
}


def radii_from_elements(elements: np.ndarray, table: str = "bondi") -> np.ndarray:
    """Return [A] radii for an [A] array of upper-case element symbols."""
    try:
        lookup = RADII_TABLES[table]
    except KeyError as exc:
        raise ValueError(f"unknown radii table {table!r}; available: {sorted(RADII_TABLES)}") from exc
    elements = np.asarray(elements, dtype=str)
    unknown = sorted(set(elements.tolist()) - set(lookup))
    if unknown:
        raise ValueError(f"no radius for elements {unknown} in table {table!r}")
    return np.array([lookup[e] for e in elements], dtype=np.float64)
