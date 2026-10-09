"""Structural-accuracy metrics for comparing a candidate structure with the experimental one (Stage 3).

All functions work on C-alpha coordinates given in SEQUENCE order, i.e. arrays [N, 3] where row i is residue i+1 of the
chain (the same indexing as `AtomicStructure`), plus a boolean `has` array [N] saying which rows hold a real
coordinate. Residues are paired by position, so no alignment step is needed (truth and candidate share one sequence).

Metrics
  * C-alpha RMSD after optimal rigid superposition (Kabsch 1976). Primary metric of the Stage 3 plan. Dominated by the
    worst-placed region and grows with chain length, hence the two metrics below are always reported next to it.
  * lDDT-C-alpha (local Distance Difference Test, Mariani et al. 2013, Bioinformatics 29:2722): inclusion radius 15 A,
    thresholds 0.5/1/2/4 A, same-residue pairs excluded, pooled over all reference pairs. Superposition-free. Reference
    residues absent from the candidate count as not conserved. This is OBSERVED accuracy against the experimental
    structure; it is not pLDDT (AlphaFold's predicted confidence, which is trained to estimate this same quantity).
  * TM-score (Zhang & Skolnick 2004, Proteins 57:702) for the FIXED residue correspondence, normalised by the number of
    reference residues, with d0 = 1.24 (L-15)^(1/3) - 1.8. The superposition is found with a fragment-seeded iterative
    search modelled on the TM-score program; it is a lower bound of the fixed-correspondence optimum, and a direct
    random-restart optimiser finds at most ~0.001 more above TM 0.5 and ~0.004 more near TM 0.2 (tests). TM-align can score a little higher because
    it may re-align residues, which a fixed correspondence cannot. Use it for ranking/screening; for numbers you will
    publish, run the reference TM-score program (Zhang & Skolnick 2004; https://zhanggroup.org/TM-score/) on the pairs.
  All three metrics use C-alpha atoms only: they assess backbone placement, not side-chain packing, and a global value
  does not say whether the error is a loop, a domain or a helix orientation (that breakdown is separate, not yet done).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------------------------------- superposition
def kabsch(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(R, t) minimising |mobile @ R.T + t - reference|^2 over proper rotations R. Inputs [n, 3], n >= 3."""
    mc, rc = mobile.mean(0), reference.mean(0)
    h = (mobile - mc).T @ (reference - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d if d != 0 else 1.0]) @ u.T
    return r, rc - mc @ r.T


def rmsd_after_superposition(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Minimum RMSD over rigid motions (closed form from the singular values; no coordinates are moved)."""
    n = len(mobile)
    if n < 3:
        raise ValueError("need at least 3 paired points for a superposition")
    x, y = mobile - mobile.mean(0), reference - reference.mean(0)
    s = np.linalg.svd(x.T @ y, compute_uv=False)
    d = np.sign(np.linalg.det(x.T @ y))
    e0 = (x ** 2).sum() + (y ** 2).sum()
    return float(np.sqrt(max(e0 - 2.0 * (s[0] + s[1] + (d if d != 0 else 1.0) * s[2]), 0.0) / n))


def pairwise_rmsd(coords: np.ndarray) -> np.ndarray:
    """[K, K] matrix of optimal-superposition RMSDs between K structures given as [K, L, 3] (same L points each)."""
    k, length, _ = coords.shape
    c = coords - coords.mean(1, keepdims=True)
    sq = (c ** 2).sum((1, 2))
    out = np.zeros((k, k))
    for i in range(k - 1):
        h = np.einsum("ld,jle->jde", c[i], c[i + 1:])                     # [K-i-1, 3, 3]
        u, s, vt = np.linalg.svd(h)
        det = np.sign(np.linalg.det(u) * np.linalg.det(vt))
        det = np.where(det == 0, 1.0, det)
        tr = s[:, 0] + s[:, 1] + det * s[:, 2]
        r = np.sqrt(np.maximum(sq[i] + sq[i + 1:] - 2.0 * tr, 0.0) / length)
        out[i, i + 1:] = r
        out[i + 1:, i] = r
    return out


# ----------------------------------------------------------------------------------------------------- lDDT
def lddt_ca(reference: np.ndarray, model: np.ndarray, ref_has: Optional[np.ndarray] = None,
            model_has: Optional[np.ndarray] = None, inclusion_radius: float = 15.0,
            thresholds=(0.5, 1.0, 2.0, 4.0), per_residue: bool = False):
    """lDDT on C-alpha atoms. Returns the pooled score in [0, 1], or (pooled, per-residue array [N]) if per_residue.

    Pairs are all i != j with reference distance < inclusion_radius, both residues present in the reference. A pair
    scores the fraction of thresholds at which |d_model - d_ref| < threshold; it scores 0 if either residue is absent
    from the model. Per-residue values are NaN for residues with no reference neighbours or absent from the reference."""
    n = len(reference)
    ref_has = np.ones(n, bool) if ref_has is None else np.asarray(ref_has, bool)
    model_has = np.ones(n, bool) if model_has is None else np.asarray(model_has, bool)
    idx = np.flatnonzero(ref_has)
    dref = np.linalg.norm(reference[idx, None] - reference[None, idx], axis=-1)
    both = model_has[idx][:, None] & model_has[idx][None, :]
    dmod = np.linalg.norm(np.where(model_has[:, None], model, 0.0)[idx, None] -
                          np.where(model_has[:, None], model, 0.0)[None, idx], axis=-1)
    pair = (dref < inclusion_radius) & ~np.eye(len(idx), dtype=bool)
    diff = np.abs(dmod - dref)
    score = np.zeros(diff.shape)
    for t in thresholds:
        score += (diff < t) & both
    score /= len(thresholds)
    total_pairs = pair.sum()
    pooled = float((score * pair).sum() / total_pairs) if total_pairs else float("nan")
    if not per_residue:
        return pooled
    cnt = pair.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(cnt > 0, (score * pair).sum(1) / np.maximum(cnt, 1), np.nan)
    full = np.full(n, np.nan)
    full[idx] = per
    return pooled, full


# ------------------------------------------------------------------------------------------------ TM-score
def tm_d0(length: int) -> float:
    return 0.5 if length <= 21 else 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8


def _tm_sum(d2: np.ndarray, d0: float) -> float:
    return float((1.0 / (1.0 + d2 / (d0 * d0))).sum())


def tm_score(reference: np.ndarray, model: np.ndarray, length_norm: Optional[int] = None) -> float:
    """TM-score of paired points [n, 3] (fixed correspondence), normalised by `length_norm` (default n)."""
    n = len(reference)
    if n < 3:
        return float("nan")
    length = int(length_norm or n)
    d0 = tm_d0(length)
    d0_search = min(max(d0, 4.5), 8.0)
    best = 0.0
    frag_lengths = []
    lf = n
    while lf >= 4:
        frag_lengths.append(lf)
        lf //= 2
    if not frag_lengths or frag_lengths[-1] > 4:
        frag_lengths.append(min(4, n))
    for lf in dict.fromkeys(frag_lengths):
        step = max(1, lf // 2)
        for start in range(0, n - lf + 1, step):
            sel = np.arange(start, start + lf)
            prev = None
            for _ in range(20):
                r, t = kabsch(model[sel], reference[sel])
                d2 = ((model @ r.T + t - reference) ** 2).sum(1)
                best = max(best, _tm_sum(d2, d0) / length)
                cut = d0_search ** 2
                new = np.flatnonzero(d2 < cut)
                while len(new) < 3 and cut < 1e6:
                    cut += 2.0 * d0_search + 1.0
                    new = np.flatnonzero(d2 < cut)
                if prev is not None and len(new) == len(prev) and np.array_equal(new, prev):
                    break
                prev, sel = new, new
    return float(best)


# ------------------------------------------------------------------------------------------------ combined
def ca_coordinates(structure) -> Tuple[np.ndarray, np.ndarray]:
    """([N, 3] C-alpha coordinates with NaN for residues without one, [N] bool present) from an AtomicStructure."""
    n = structure.n_residues
    xyz = np.full((n, 3), np.nan)
    sel = structure.atom_names == "CA"
    xyz[structure.residue_index[sel]] = structure.coords[sel]
    return xyz, np.isfinite(xyz[:, 0])


def structural_metrics(reference_ca: np.ndarray, reference_has: np.ndarray, model_ca: np.ndarray,
                       model_has: np.ndarray) -> Dict[str, float]:
    """RMSD / lDDT-Ca / TM-score of a model against the experimental reference.

    RMSD and TM-score use the residues present in both (TM-score is still normalised by the number of reference
    residues, so a model missing residues is penalised). lDDT penalises absent model residues as described above."""
    common = reference_has & model_has
    n_ref, n_common = int(reference_has.sum()), int(common.sum())
    out = {"n_reference": n_ref, "n_common": n_common, "rmsd_ca": float("nan"), "lddt_ca": float("nan"),
           "tm_score": float("nan")}
    if n_common >= 3:
        out["rmsd_ca"] = rmsd_after_superposition(model_ca[common], reference_ca[common])
        out["tm_score"] = tm_score(reference_ca[common], model_ca[common], length_norm=n_ref)
    out["lddt_ca"] = lddt_ca(reference_ca, np.nan_to_num(model_ca), reference_has, model_has)
    return out
