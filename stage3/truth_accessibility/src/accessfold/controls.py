"""Controls for the Stage 3 question "does agreement with the accessibility measurement predict structural accuracy?".

Per protein (the independent unit) we have K candidates with a structural error s_k (Ca RMSD, 1 - lDDT, 1 - TM-score)
and an accessibility error e_k (mean over probe radii of the mean absolute difference between the candidate's and the
truth per-residue relative accessibility, over the truth-mask residues). The statistic is the Spearman rank correlation
rho(e, s) within the protein; positive rho means "candidates that agree better with the accessibility data are
structurally more accurate". Every control changes ONLY the truth side (or the set of residues used); the candidates
and their structural errors never change.

  real           the actual comparison.
  shuffle        the truth values are permuted among the masked-in residues (all radii of a residue move together).
                 Keeps the set of values (overall exposure) and destroys which residue has which value. Repeat many
                 times to get a null distribution. NOTE: this null need not be centred on 0. If worse candidates are
                 systematically more/less exposed overall, a shuffled profile still correlates with structural error
                 through that overall exposure; the quantity of interest is the shift of `real` relative to the shuffle
                 null (position-specific information), not the distance of `real` from 0.
  reverse        the truth profile of the masked-in residues is reversed along the sequence.
  other_protein  the profile of a different protein, compacted and linearly resampled to this protein's number of
                 masked-in residues (a sample of up to `n_other` other proteins).
  dropout:f      a random fraction f of the masked-in residues is kept (measurement sparsity; repeated).
  binary:t       truth AND candidate relative accessibilities are turned into buried (< t) / exposed (>= t) labels
                 (t a number, or "median" for a per-radius median split of the truth); the error is then the
                 fraction of residues whose label differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from accessfold.scoring import aggregate_radii, corr_error_matrix, error_matrix

STRUCT_METRICS = ("rmsd_ca", "lddt_ca", "tm_score")


@dataclass
class Entry:
    """One protein's saved scoring result (see `stage3/scoring/score_candidates.py`)."""
    name: str
    y: np.ndarray                      # [N, R] truth profile (NaN outside the mask)
    c: np.ndarray                      # [K, N, R] candidate profiles
    rmsd_ca: np.ndarray                # [K]
    lddt_ca: np.ndarray                # [K]
    tm_score: np.ndarray               # [K]
    radii: Sequence[float] = field(default_factory=tuple)

    def structural_error(self, metric: str) -> np.ndarray:
        """[K] error-like version of a metric (larger = worse): RMSD, 1 - lDDT, 1 - TM-score."""
        return {"rmsd_ca": self.rmsd_ca, "lddt_ca": 1.0 - self.lddt_ca, "tm_score": 1.0 - self.tm_score}[metric]

    @property
    def rmsd_spread(self) -> float:
        r = self.rmsd_ca[np.isfinite(self.rmsd_ca)]
        return float(r.max() - r.min()) if r.size else float("nan")


def load_entry(path, min_coverage: float = 1.0) -> "Entry":
    """Entry from a `<entry>.npz` written by stage3/scoring/score_candidates.py.

    Candidates whose coverage (Ca of the truth-resolved residues, or complete atoms on the truth-mask residues) is below
    `min_coverage` are dropped: a candidate that skips part of the structure would otherwise be compared on an easier
    subset. (score_candidates.py already rejects them at its default --min-coverage 1.0; this guards results that were
    produced with a lower setting.)"""
    import json
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        keep = (z["coverage_ca"] >= min_coverage - 1e-12) & (z["coverage_mask"] >= min_coverage - 1e-12)
        return Entry(name=meta["entry"], y=z["y"].astype(float), c=z["c"].astype(float)[keep], rmsd_ca=z["rmsd_ca"][keep],
                     lddt_ca=z["lddt_ca"][keep], tm_score=z["tm_score"][keep], radii=tuple(meta["radii"]))


def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 4) -> float:
    """Spearman rho over the jointly finite pairs; NaN if fewer than `min_n` pairs or either side is constant."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


ERROR_METRICS = ("mae", "corr")


def accessibility_error(y: np.ndarray, c: np.ndarray, min_residues: int = 5, metric: str = "mae") -> np.ndarray:
    """[K] single accessibility error: mean over radii of a per-radius error.

    metric "mae":  mean absolute difference (default; sensitive to overall exposure level as well as to which
                   residues are exposed).
    metric "corr": 1 - Pearson correlation of the profiles (sensitive only to WHICH residues are exposed; its shuffle
                   null is centred on a known value, which makes the shuffle control easy to read)."""
    if metric == "mae":
        mae, _, n = error_matrix(y, c)
        return aggregate_radii(mae, n, min_residues)
    if metric == "corr":
        err, n = corr_error_matrix(y, c)
        return aggregate_radii(err, n, min_residues)
    raise ValueError(f"metric must be one of {ERROR_METRICS}")


def _valid_rows(y: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(y).any(axis=1))


# ------------------------------------------------------------------------------------- truth-side transforms
def shuffled(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows = _valid_rows(y)
    out = np.full_like(y, np.nan)
    out[rows] = y[rows[rng.permutation(rows.size)]]
    return out


def reversed_profile(y: np.ndarray) -> np.ndarray:
    rows = _valid_rows(y)
    out = np.full_like(y, np.nan)
    out[rows] = y[rows[::-1]]
    return out


def resampled_other(y: np.ndarray, y_other: np.ndarray) -> np.ndarray:
    """Profile of another protein, compacted and linearly resampled onto this protein's masked-in rows."""
    rows, rows_o = _valid_rows(y), _valid_rows(y_other)
    if rows_o.size < 2 or y_other.shape[1] != y.shape[1]:
        raise ValueError("incompatible other-protein profile")
    src = np.linspace(0.0, 1.0, rows_o.size)
    dst = np.linspace(0.0, 1.0, rows.size)
    out = np.full_like(y, np.nan)
    for j in range(y.shape[1]):
        out[rows, j] = np.interp(dst, src, y_other[rows_o, j])
    return out


def keep_fraction(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    rows = _valid_rows(y)
    n_keep = max(1, int(round(frac * rows.size)))
    out = np.full_like(y, np.nan)
    sel = rows[rng.choice(rows.size, n_keep, replace=False)]
    out[sel] = y[sel]
    return out


def binarise(y: np.ndarray, c: np.ndarray, threshold) -> tuple:
    """(y_bin, c_bin): 1.0 = exposed (>= threshold), 0.0 = buried; NaN preserved. threshold: float or 'median'."""
    if threshold == "median":
        thr = np.nanmedian(y, axis=0)[None, :]
    else:
        thr = float(threshold)
    yb = np.where(np.isfinite(y), (y >= thr).astype(float), np.nan)
    cb = np.where(np.isfinite(c), (c >= (thr[None] if np.ndim(thr) else thr)).astype(float), np.nan)
    return yb, cb


# ------------------------------------------------------------------------------------------------ drivers
def rho_for(entry: Entry, e: np.ndarray, metrics: Iterable[str] = STRUCT_METRICS) -> Dict[str, float]:
    return {m: spearman(e, entry.structural_error(m)) for m in metrics}


def run_control(entry: Entry, control: str, rng: np.random.Generator, n_rep: int = 1000,
                others: Optional[List[Entry]] = None, n_other: int = 20, min_residues: int = 5,
                metrics: Iterable[str] = STRUCT_METRICS, error_metric: str = "mae") -> Dict[str, np.ndarray]:
    """rho(accessibility error, structural error) for one protein under one control.

    Returns {metric: array of rho, one per repetition} (a single value for the deterministic controls
    `real` and `reverse`, and one per sampled protein for `other_protein`)."""
    metrics = tuple(metrics)
    name, _, param = control.partition(":")
    y, c = entry.y, entry.c

    def one(y_t: np.ndarray, c_t: np.ndarray = c) -> Dict[str, float]:
        return rho_for(entry, accessibility_error(y_t, c_t, min_residues, error_metric), metrics)

    results: List[Dict[str, float]] = []
    if name == "real":
        results.append(one(y))
    elif name == "reverse":
        results.append(one(reversed_profile(y)))
    elif name == "shuffle":
        results = [one(shuffled(y, rng)) for _ in range(n_rep)]
    elif name == "dropout":
        results = [one(keep_fraction(y, float(param), rng)) for _ in range(n_rep)]
    elif name == "binary":
        yb, cb = binarise(y, c, "median" if param == "median" else float(param or 0.2))
        results.append(one(yb, cb))
    elif name == "other_protein":
        pool = [o for o in (others or []) if o.name != entry.name and o.y.shape[1] == y.shape[1]]
        if not pool:
            raise ValueError("other_protein control needs other entries")
        pick = rng.choice(len(pool), size=min(n_other, len(pool)), replace=False)
        results = [one(resampled_other(y, pool[i].y)) for i in pick]
    else:
        raise ValueError(f"unknown control {control!r}")
    return {m: np.array([r[m] for r in results]) for m in metrics}


# ------------------------------------------------------------------------------------- across proteins
def bootstrap_median(values: np.ndarray, n_boot: int = 10000, rng: Optional[np.random.Generator] = None,
                     alpha: float = 0.05) -> tuple:
    """(median, lo, hi): percentile bootstrap over proteins (the independent unit) of the median."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    meds = np.median(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def protein_level_null(null_rho: Dict[str, np.ndarray], rng: np.random.Generator, n_draw: int = 10000) -> np.ndarray:
    """Null distribution of the across-protein MEDIAN rho: each draw picks one shuffle repetition independently
    for every protein. `null_rho` maps entry name -> array of shuffle rhos."""
    arrays = [a[np.isfinite(a)] for a in null_rho.values()]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return np.array([])
    draws = np.stack([a[rng.integers(0, a.size, size=n_draw)] for a in arrays], axis=1)
    return np.median(draws, axis=1)
