"""Stage 3 scoring of candidate structures against one experimental entry.

For a protein with experimental structure X_true (accessibility y, saved by `truth.truth_arrays`) and K candidates
X_k, this computes for every candidate
  * structural error: Ca RMSD (primary), lDDT-Ca, TM-score  (accessfold.structure_metrics), and
  * accessibility error: how far the candidate's per-residue accessibility y_hat_k is from y, over exactly the
    residues in the truth mask, at every probe radius.

Candidate environment (`environment`): the truth accessibility was computed on the atoms the experiment resolved
(unresolved residues absent, partly modelled residues truncated, hydrogens/ligands/other chains dropped). Comparing it
with a candidate that has every atom would mix a structural difference with a bookkeeping difference, so by default the
candidate is first restricted to the atoms the truth has:
  "truth_atoms"     keep only candidate atoms that also exist in the truth (same residue position and atom name;
                    truth selenomethionine SE is matched to the candidate's SD). Default.
  "truth_residues"  delete the candidate residues that are unresolved in the truth, keep the rest whole.
  "all_atoms"       use the complete candidate (sensitivity analysis).
Truth residues outside the mask never contribute to an error; predicted coordinates are never used to fill truth values.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from accessfold.accessibility.compute import compute_accessibility
from accessfold.structures.atomic import AtomicStructure

ENVIRONMENTS = ("truth_atoms", "truth_residues", "all_atoms")
KINDS = {"rel": "relative", "abs": "absolute", "sc_abs": "sidechain_absolute", "sc_rel": "sidechain_relative"}


def _atom_keys(s: AtomicStructure) -> np.ndarray:
    names = np.where(s.atom_names == "SE", "SD", s.atom_names)       # selenomethionine SE takes the place of SD
    return np.char.add(np.char.add(s.residue_index.astype(str), ":"), names)


def restrict_environment(candidate: AtomicStructure, truth: AtomicStructure, environment: str = "truth_atoms") -> AtomicStructure:
    """The candidate with the atoms removed that the truth calculation did not see (see module docstring)."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}")
    if candidate.n_residues != truth.n_residues:
        raise ValueError(f"candidate has {candidate.n_residues} residues, truth {truth.n_residues}")
    if environment == "all_atoms":
        return candidate
    if environment == "truth_residues":
        keep = truth.resolved_mask[candidate.residue_index]
    else:
        keep = np.isin(_atom_keys(candidate), _atom_keys(truth))
    return AtomicStructure(candidate.coords[keep], candidate.elements[keep], candidate.atom_names[keep],
                           candidate.residue_index[keep], candidate.residue_names, candidate.chain_id)


def candidate_profiles(candidate: AtomicStructure, truth_structure: AtomicStructure, truth: Dict[str, np.ndarray],
                       radii: Sequence[float], n_points: int, environment: str = "truth_atoms") -> Dict[str, np.ndarray]:
    """All four [N, R] profiles, using one surface calculation per radius.

    Values outside the reference mask are NaN; undefined side-chain values
    (including glycine) remain NaN rather than being replaced with zero.
    """
    env = restrict_environment(candidate, truth_structure, environment)
    cols = {kind: [] for kind in KINDS}
    for radius in radii:
        result = compute_accessibility(env, "relative_sasa", probe_radius=float(radius), n_points=n_points)
        for kind, attr in KINDS.items():
            cols[kind].append(np.where(truth["mask"], getattr(result, attr), np.nan))
    return {kind: np.stack(values, axis=1) for kind, values in cols.items()}


def candidate_profile(candidate: AtomicStructure, truth_structure: AtomicStructure, truth: Dict[str, np.ndarray],
                      radii: Sequence[float], n_points: int, environment: str = "truth_atoms",
                      kind: str = "rel") -> np.ndarray:
    """[N, R] accessibility of one candidate at each radius; NaN outside the truth mask (and where the value is undefined).

    `truth` is the dict of arrays saved by the truth pipeline (needs `mask`); `n_points` must be the value used there."""
    return candidate_profiles(candidate, truth_structure, truth, radii, n_points, environment)[kind]


def truth_profile(truth: Dict[str, np.ndarray], radii: Sequence[float], kind: str = "rel") -> np.ndarray:
    """[N, R] truth accessibility from the saved arrays (NaN outside the mask)."""
    if kind not in KINDS:
        raise ValueError(f"unknown accessibility kind {kind!r}")
    prefix = kind
    return np.stack([np.asarray(truth[f"{prefix}_{float(r):g}"], float) for r in radii], axis=1)


# ---------------------------------------------------------------------------------------- accessibility error
def error_matrix(y: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate, per-radius errors against the truth profile.

    y [N, R] truth, c [K, N, R] candidates; entries that are NaN in either are ignored.
    Returns (mae [K, R], rmse [K, R], n [K, R]) with NaN where fewer than one residue is available."""
    d = c - y[None]
    ok = np.isfinite(d)
    n = ok.sum(1)
    dz = np.where(ok, d, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(n > 0, np.abs(dz).sum(1) / np.maximum(n, 1), np.nan)
        rmse = np.where(n > 0, np.sqrt((dz ** 2).sum(1) / np.maximum(n, 1)), np.nan)
    return mae, rmse, n


def corr_error_matrix(y: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate, per-radius 1 - Pearson r between the candidate and truth profiles over the jointly finite residues.

    Unlike the mean absolute error this ignores the overall level and spread of the values and measures only whether the
    right residues are the buried/exposed ones, so a randomly permuted truth gives errors centred on 1 for every
    candidate (a clean null). Returns (err [K, R], n [K, R]); NaN where fewer than 3 residues or either side is constant."""
    ok = np.isfinite(c) & np.isfinite(y[None])
    n = ok.sum(1)
    cz, yz = np.where(ok, c, 0.0), np.where(ok, np.broadcast_to(y[None], c.shape), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.maximum(n, 1)
        cm, ym = cz.sum(1) / nn, yz.sum(1) / nn
        cc, yc = np.where(ok, c - cm[:, None, :], 0.0), np.where(ok, np.broadcast_to(y[None], c.shape) - ym[:, None, :], 0.0)
        den = np.sqrt((cc ** 2).sum(1) * (yc ** 2).sum(1))
        r = np.where((den > 0) & (n >= 3), (cc * yc).sum(1) / den, np.nan)
    return 1.0 - r, n


def aggregate_radii(err: np.ndarray, n: Optional[np.ndarray] = None, min_residues: int = 5) -> np.ndarray:
    """[K] mean over probe radii of a [K, R] error matrix, ignoring radii with fewer than `min_residues` residues.

    This unweighted mean over radii is the pre-declared single accessibility error used for the main plot."""
    e = np.array(err, float)
    if n is not None:
        e[np.asarray(n) < min_residues] = np.nan
    with np.errstate(all="ignore"):
        out = np.nanmean(e, axis=1)
    return out
