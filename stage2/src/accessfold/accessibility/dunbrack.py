"""Reader for the Dunbrack backbone-dependent rotamer library (Shapovalov & Dunbrack 2011, Structure 19:844).

Plugs into `tripeptide.RotamerSource`. Point it at `ALL.bbdep.rotamers.lib` from the "Simple Mode" download
(10-degree phi/psi grid; one row per rotamer with its probability and mean chi1..chi4). The library itself is
NOT bundled (about 80 MB; licensed Open Data Commons Attribution - cite the paper above if you use it).

Row format (comment lines start with "# "):
    T  Phi  Psi  Count  r1 r2 r3 r4  Probability  chi1Val..chi4Val  chi1Sig..chi4Sig

Choices made here, all recorded by `describe()`:
* only each rotamer's MEAN chi values are used (the standard deviations are ignored), so the maximum is taken
  over library-mean conformations rather than over the full spread of each rotamer;
* rotamers with probability below `min_probability` at that (phi, psi) are dropped (the library lists
  rotamers down to ~1e-6); if none survive, the single most probable one is kept;
* (phi, psi) is snapped to the nearest 10-degree grid point (exact for the package's default scan);
* residue `CYS` (the library's undifferentiated cysteine) is used for Cys; Pro has no chi variation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from accessfold.accessibility.tripeptide import CHI_ATOMS

_LIB_STEP = 10.0
# (path, mtime, resname) -> {(phi_idx, psi_idx): (probs [K], chi [K, n_chi])}
_CACHE: Dict[Tuple[str, float, str], Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]] = {}


def _parse_residue(path: Path, resname: str, n_chi: int) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    rows: Dict[Tuple[int, int], list] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(resname + " "):
                continue
            f = line.split()
            if f[0] != resname or len(f) < 9 + 4:
                continue
            key = (int(round(float(f[1]) / _LIB_STEP)), int(round(float(f[2]) / _LIB_STEP)))
            rows.setdefault(key, []).append((float(f[8]), [float(x) for x in f[9:9 + n_chi]]))
    return {k: (np.array([p for p, _ in v]), np.array([c for _, c in v]).reshape(len(v), n_chi))
            for k, v in rows.items()}


@dataclass(frozen=True)
class DunbrackLibrary:
    """`RotamerSource` backed by a Dunbrack `ALL.bbdep.rotamers.lib` file (Simple Mode)."""

    path: str
    min_probability: float = 0.01
    name: str = "dunbrack2010_simple"

    def _table(self, resname: str):
        p = Path(self.path)
        key = (str(p.resolve()), p.stat().st_mtime, resname)
        if key not in _CACHE:
            n_chi = len(CHI_ATOMS[resname])
            table = _parse_residue(p, resname, n_chi)
            if not table:
                raise ValueError(f"{resname}: no rows found in {p}")
            _CACHE[key] = table
        return _CACHE[key]

    def chi_sets(self, resname: str, phi: float, psi: float) -> np.ndarray:
        if resname not in CHI_ATOMS:
            return np.zeros((1, 0))
        table = self._table(resname)
        n = int(round(360.0 / _LIB_STEP))  # 36 grid steps per revolution; the file stores -180 AND +180

        def lookup(i: int, j: int):
            for di in (0, -n, n):
                for dj in (0, -n, n):
                    if (i + di, j + dj) in table:
                        return table[(i + di, j + dj)]
            raise KeyError(f"{resname}: no library row near phi={phi}, psi={psi}")

        probs, chi = lookup(int(round(phi / _LIB_STEP)), int(round(psi / _LIB_STEP)))
        keep = probs >= self.min_probability
        if not keep.any():
            keep = probs == probs.max()
        return chi[keep]

    def describe(self) -> dict:
        return {"source": self.name, "file": Path(self.path).name, "min_probability": self.min_probability,
                "uses": "rotamer mean chi only (sigmas ignored)", "phi_psi_grid_deg": _LIB_STEP,
                "backbone_dependent": True,
                "citation": "Shapovalov & Dunbrack 2011, Structure 19:844 (Open Data Commons Attribution)"}
