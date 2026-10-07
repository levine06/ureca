"""Accessibility of an experimental ("true") chain at several probe radii, with the masks Stage 3 needs.

Stage 3 compares y = f(X_true) with y_hat_k = f(X_k) for candidate structures X_k. To keep that comparison
meaningful the SAME mask must be applied to both: `mask` here is saved with the result so candidates can be
evaluated on exactly these residues. A residue is masked when it is
  * unresolved (no usable atoms), or only partly modelled (missing heavy atoms),
  * chemically modified / different from the sequence residue (e.g. a phospho- or S-methyl-cysteine): its value is not
    comparable with the ordinary-residue reference or with an unmodified prediction (selenomethionine excepted), or
  * in the "shadow" of a modification: its own area changes by more than `shadow_threshold` Angstrom^2 at any probe
    radius when the modification's extra atoms are removed. Candidates carry no such atoms, so these residues would
    disagree for chemical, not structural, reasons.
Masked residues' atoms still occlude their neighbours in the truth calculation (except the removal test above).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from accessfold.accessibility.compute import compute_accessibility
from accessfold.structures.atomic import AtomicStructure
from accessfold.structures.mmcif import gap_distance


def _strip_extra_atoms(structure: AtomicStructure, modified: Sequence[dict]) -> AtomicStructure:
    drop = np.zeros(structure.n_atoms, dtype=bool)
    for m in modified:
        if m.get("extra_atoms"):
            drop |= (structure.residue_index == m["index"]) & np.isin(structure.atom_names, m["extra_atoms"])
    keep = ~drop
    return AtomicStructure(structure.coords[keep], structure.elements[keep], structure.atom_names[keep],
                           structure.residue_index[keep], structure.residue_names, structure.chain_id)


def truth_arrays(structure: AtomicStructure, complete: np.ndarray, probe_radii: Iterable[float] = (1.4, 2.5, 4.0, 6.0),
                 n_points: int = 1000, mask_incomplete: bool = True, modified: Optional[Sequence[dict]] = None,
                 shadow_threshold: float = 5.0) -> Dict[str, np.ndarray]:
    """Dict of arrays, each [N] (sequence positions). Keys:

    residue_names, resolved, complete, modified, shadow, gap_distance, mask  and, per radius r (formatted `:g`):
    abs_r, rel_r, sc_abs_r, sc_rel_r  (NaN where the residue is not in `mask`).
    `modified` is the list of dicts from `load_chain_from_mmcif(...)[2]["modified_residues"]`.
    """
    radii = [float(r) for r in probe_radii]
    modified = list(modified or [])
    resolved = structure.resolved_mask
    n = structure.n_residues
    is_mod = np.zeros(n, dtype=bool)
    for m in modified:
        is_mod[m["index"]] = True

    results = {r: compute_accessibility(structure, "relative_sasa", probe_radius=r, n_points=n_points) for r in radii}
    shadow = np.zeros(n, dtype=bool)
    if any(m.get("extra_atoms") for m in modified):
        stripped = _strip_extra_atoms(structure, modified)
        for r in radii:
            plain = compute_accessibility(stripped, "absolute_sasa", probe_radius=r, n_points=n_points)
            with np.errstate(invalid="ignore"):
                shadow |= np.nan_to_num(np.abs(results[r].absolute - plain.absolute), nan=0.0) > shadow_threshold
        shadow &= ~is_mod

    mask = resolved & ~is_mod & ~shadow
    if mask_incomplete:
        mask &= np.asarray(complete, dtype=bool)
    out: Dict[str, np.ndarray] = {
        "residue_names": structure.residue_names.astype("<U3"), "resolved": resolved,
        "complete": np.asarray(complete, dtype=bool), "modified": is_mod, "shadow": shadow,
        "gap_distance": gap_distance(resolved), "mask": mask,
    }
    for r in radii:
        res = results[r]
        out[f"abs_{r:g}"] = np.where(mask, res.absolute, np.nan)
        out[f"rel_{r:g}"] = np.where(mask, res.relative, np.nan)
        out[f"sc_abs_{r:g}"] = np.where(mask, res.sidechain_absolute, np.nan)
        out[f"sc_rel_{r:g}"] = np.where(mask, res.sidechain_relative, np.nan)
    return out
