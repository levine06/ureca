"""Synthetic stand-ins for candidate structures, to test the Stage 3 scoring pipeline before OpenFold3 output exists.

`smooth_displacement` bends a structure with a random low-frequency displacement field (a few random plane waves,
wavelength 15-40 A by default) scaled so that the root-mean-square displacement of the C-alpha atoms is exactly `sigma`.
Because the field is smooth, local geometry stays roughly intact while domains and secondary-structure elements move,
which is closer to a real prediction error than independent noise on every atom. This is a pipeline test, NOT a model of
OpenFold3 errors: a correlation found with such candidates shows the code works, not that the experiment will.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from accessfold.structures.atomic import AtomicStructure
from accessfold.structures.mmcif import ONE_TO_THREE  # noqa: F401  (re-exported for convenience)

_HEADER = """data_SYNTHETIC
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.pdbx_PDB_model_num
"""


def smooth_displacement(coords: np.ndarray, reference_ca: np.ndarray, sigma: float, seed: int,
                        wavelength=(15.0, 40.0), n_modes: int = 6) -> np.ndarray:
    """`coords` [A, 3] displaced by a smooth random field normalised to RMS `sigma` over `reference_ca` [n, 3]."""
    rng = np.random.default_rng(seed)

    def field(x: np.ndarray, params) -> np.ndarray:
        u = np.zeros_like(x)
        for k, amp, phase in params:
            u += amp[None, :] * np.sin(x @ k + phase)[:, None]
        return u

    params = []
    for _ in range(n_modes):
        k = rng.normal(size=3)
        k *= 2.0 * np.pi / rng.uniform(*wavelength) / np.linalg.norm(k)
        params.append((k, rng.normal(size=3), rng.uniform(0.0, 2.0 * np.pi)))
    scale = np.sqrt((field(reference_ca, params) ** 2).sum(1).mean())
    return coords + (sigma / scale) * field(coords, params) if scale > 0 else coords.copy()


def write_chain_cif(path: str | Path, structure: AtomicStructure, coords: np.ndarray | None = None,
                    asym_id: str = "A") -> None:
    """Write one chain as a minimal mmCIF (label_seq_id = sequence position). Residue names come from the structure."""
    xyz = structure.coords if coords is None else coords
    lines = []
    for i in range(structure.n_atoms):
        p = int(structure.residue_index[i])
        lines.append(f"ATOM {structure.elements[i]} {structure.atom_names[i]} . {structure.residue_names[p]} {asym_id} "
                     f"{p + 1} {xyz[i, 0]:.3f} {xyz[i, 1]:.3f} {xyz[i, 2]:.3f} 1.0 1\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt") as fh:
        fh.write(_HEADER + "".join(lines))
