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
                 times to get a null distribution. NOTE: do not assume this null is centred on 0 (rho) or on 0.5 (top-1
                 percentile) for ANY accessibility error. If worse candidates are systematically more/less exposed overall,
                 or the error of some candidates simply varies more than others, a shuffled profile still correlates with
                 or selects through that; the quantity of interest is the shift of `real` relative to the generated shuffle
                 null (position-specific information), not the distance of `real` from 0 or 0.5.
  reverse        the truth profile of the masked-in residues is reversed along the sequence.
  other_protein  the profile of a different protein, compacted and linearly resampled to this protein's number of
                 masked-in residues (a sample of up to `n_other` other proteins).
  dropout:f      a random fraction f of the masked-in residues is kept (measurement sparsity; repeated).
  binary:t       truth AND candidate relative accessibilities are turned into buried (< t) / exposed (>= t) labels
                 (t a number, or "median" for a per-radius median split of the truth). The error is then computed
                 on the 0/1 profiles with the chosen accessibility error: with "mae" it equals the fraction of residues
                 whose label differs; with "corr" it is 1 - the Pearson correlation of the two 0/1 profiles (undefined
                 when a profile is constant, e.g. no residue below the threshold).
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
    """One protein's saved scoring result (see `scripts/score_candidates.py`)."""
    name: str
    y: np.ndarray                      # [N, R] truth profile (NaN outside the mask)
    c: np.ndarray                      # [K, N, R] candidate profiles
    rmsd_ca: np.ndarray                # [K]
    lddt_ca: np.ndarray                # [K]
    tm_score: np.ndarray               # [K]
    radii: Sequence[float] = field(default_factory=tuple)
    candidates: Sequence[str] = field(default_factory=tuple)     # candidate ids (paths relative to the candidate root)
    split: str = ""                                              # "development" / "validation" when the ids start with it
    join_report: dict = field(default_factory=dict)              # how the candidates of the .npz were matched to the structure table

    def structural_error(self, metric: str) -> np.ndarray:
        """[K] error-like version of a metric (larger = worse): RMSD, 1 - lDDT, 1 - TM-score."""
        return {"rmsd_ca": self.rmsd_ca, "lddt_ca": 1.0 - self.lddt_ca, "tm_score": 1.0 - self.tm_score}[metric]

    @property
    def rmsd_spread(self) -> float:
        r = self.rmsd_ca[np.isfinite(self.rmsd_ca)]
        return float(r.max() - r.min()) if r.size else float("nan")


STRUCT_COLUMNS = ("rmsd_ca", "lddt_ca", "tm_score")
COVERAGE_COLUMNS = ("coverage_ca", "coverage_mask")


def normalise_id(cid: str) -> str:
    """Candidate id with Windows separators and a leading './' removed (the only normalisation applied before matching)."""
    c = str(cid).replace("\\", "/")
    return c[2:] if c.startswith("./") else c


def read_table(path, columns: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{entry: {candidate id: {column: float}}} from a CSV with `entry`, `candidate` and (some of) `columns`
    (for example structure_candidate_scores.csv written by score_candidates.py). Columns absent from the file are simply not
    present in the rows; an unparseable value becomes NaN. Duplicate (entry, candidate) rows are an error."""
    import csv
    import sys
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if not {"entry", "candidate"} <= set(rd.fieldnames or []):
            raise ValueError(f"{path}: needs columns 'entry' and 'candidate'")
        have = [c for c in columns if c in rd.fieldnames]
        for r in rd:
            cid = r["candidate"]
            row = {}
            for c in have:
                try:
                    row[c] = float(r[c])
                except (TypeError, ValueError):
                    row[c] = float("nan")
            if cid in out.setdefault(r["entry"], {}):
                raise ValueError(f"{path}: duplicate row for entry {r['entry']!r}, candidate {cid!r}")
            out[r["entry"]][cid] = row
    return out


def match_ids(wanted: Sequence[str], table_entry: Dict[str, Dict[str, float]]):
    """Match candidate ids by NAME, never by position: exact id first, then the id with separators/'./' normalised, then the
    file's base name when that is unique on both sides. Returns (rows, counts, extra): rows[i] is the table row for wanted[i]
    or None; counts = {"exact", "normalised", "basename", "unmatched"}; extra = table ids that matched no wanted id."""
    counts = {"exact": 0, "normalised": 0, "basename": 0, "unmatched": 0}
    rows: List[Optional[Dict[str, float]]] = [None] * len(wanted)
    used = set()
    norm = {}
    for tid in table_entry:
        norm.setdefault(normalise_id(tid), []).append(tid)
    base_t: Dict[str, List[str]] = {}
    for tid in table_entry:
        base_t.setdefault(normalise_id(tid).rsplit("/", 1)[-1], []).append(tid)
    base_w: Dict[str, int] = {}
    for w in wanted:
        base_w[normalise_id(w).rsplit("/", 1)[-1]] = base_w.get(normalise_id(w).rsplit("/", 1)[-1], 0) + 1
    for i, w in enumerate(wanted):
        tid, how = None, ""
        if w in table_entry and w not in used:
            tid, how = w, "exact"
        elif len(norm.get(normalise_id(w), [])) == 1 and norm[normalise_id(w)][0] not in used:
            tid, how = norm[normalise_id(w)][0], "normalised"
        else:
            b = normalise_id(w).rsplit("/", 1)[-1]
            if base_w[b] == 1 and len(base_t.get(b, [])) == 1 and base_t[b][0] not in used:
                tid, how = base_t[b][0], "basename"
        if tid is None:
            counts["unmatched"] += 1
        else:
            counts[how] += 1
            used.add(tid)
            rows[i] = table_entry[tid]
    return rows, counts, [t for t in table_entry if t not in used]


def load_entry(path, min_coverage: float = 1.0, structure: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None) -> "Entry":
    """Entry from a `<entry>.npz` written by scripts/score_candidates.py.

    Structural errors (rmsd_ca, lddt_ca, tm_score) come from `structure` (a table from `read_table`, matched to the .npz's
    candidates by id) when given, else from the .npz itself; if neither has them a clear error is raised. When both exist they
    must agree (a disagreement is an error: two different runs were mixed). Candidates that have no row in the table, or a
    non-finite structural value, are dropped and counted in `Entry.join_report`.

    Candidates whose coverage (Ca of the truth-resolved residues, or complete atoms on the truth-mask residues) is below
    `min_coverage` are dropped: a candidate that skips part of the structure would otherwise be compared on an easier
    subset. Coverage comes from the .npz, else from the table's coverage columns, else it cannot be checked (reported)."""
    import json
    from pathlib import Path
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        files = set(z.files)
        c_all = z["c"].astype(float)
        k_all = c_all.shape[0]
        ids = list(meta.get("candidates", []))
        if ids and len(ids) != k_all:
            raise ValueError(f"{Path(path).name}: {len(ids)} candidate ids for {k_all} candidate profiles")
        npz_struct = {m: z[m].astype(float) for m in STRUCT_COLUMNS if m in files}
        npz_cov = {m: z[m].astype(float) for m in COVERAGE_COLUMNS if m in files}
        y = z["y"].astype(float)
    name = meta["entry"]
    report = {"n_npz_candidates": k_all, "exact": 0, "normalised": 0, "basename": 0, "unmatched": 0, "extra_in_table": 0,
              "structure_source": "npz", "coverage_checked": True, "dropped_nonfinite": 0, "dropped_coverage": 0}
    matched = np.ones(k_all, bool)
    arrays = dict(npz_struct)
    cov = dict(npz_cov)
    if structure is not None:
        if not ids:
            raise ValueError(f"{Path(path).name}: has no candidate ids, so it cannot be joined to the structure table by name")
        rows, counts, extra = match_ids(ids, structure.get(name, {}))
        report.update(counts)
        report["extra_in_table"] = len(extra)
        report["structure_source"] = "table"
        matched = np.array([r is not None for r in rows])
        tab = {}
        for m in STRUCT_COLUMNS + COVERAGE_COLUMNS:
            if all(m in r for r in rows if r is not None) and matched.any():
                tab[m] = np.array([r[m] if r is not None else np.nan for r in rows], float)
        missing = [m for m in STRUCT_COLUMNS if m not in tab]
        if missing and matched.any():
            raise ValueError(f"structure table has no column(s) {missing} for entry {name}")
        for m in STRUCT_COLUMNS:                                   # two sources must agree, never silently prefer one
            if m in npz_struct and m in tab:
                both = matched & np.isfinite(npz_struct[m]) & np.isfinite(tab[m])
                if both.any() and not np.allclose(npz_struct[m][both], tab[m][both], rtol=1e-6, atol=1e-6):
                    raise ValueError(f"{name}: {m} in the .npz and in the structure table disagree (different runs mixed?)")
        arrays = {m: tab.get(m, np.full(k_all, np.nan)) for m in STRUCT_COLUMNS}
        for m in COVERAGE_COLUMNS:
            if m not in cov and m in tab:
                cov[m] = tab[m]
    elif not all(m in npz_struct for m in STRUCT_COLUMNS):
        raise ValueError(f"{Path(path).name}: this score file has no structural arrays ({', '.join(m for m in STRUCT_COLUMNS if m not in npz_struct)}); "
                         "pass the structural-score table with --structure-csv (it is joined by entry + candidate id)")
    keep = matched.copy()
    finite = np.isfinite(arrays["rmsd_ca"]) & np.isfinite(arrays["lddt_ca"]) & np.isfinite(arrays["tm_score"])
    report["dropped_nonfinite"] = int((keep & ~finite).sum())
    keep &= finite
    if all(m in cov for m in COVERAGE_COLUMNS):
        ok_cov = (cov["coverage_ca"] >= min_coverage - 1e-12) & (cov["coverage_mask"] >= min_coverage - 1e-12)
        report["dropped_coverage"] = int((keep & ~ok_cov).sum())
        keep &= ok_cov
    else:
        report["coverage_checked"] = False
    kept_ids = [cid for cid, k in zip(ids, keep) if k] if ids else []
    first = (kept_ids[0] if kept_ids else "").replace("\\", "/")
    split = first.split("/")[0] if "/" in first else str(meta.get("split", ""))
    report["n_kept"] = int(keep.sum())
    return Entry(name=name, y=y, c=c_all[keep], rmsd_ca=arrays["rmsd_ca"][keep], lddt_ca=arrays["lddt_ca"][keep],
                 tm_score=arrays["tm_score"][keep], radii=tuple(meta["radii"]), candidates=tuple(kept_ids), split=split,
                 join_report=report)


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
    metric "corr": 1 - Pearson correlation of the profiles (insensitive to the overall exposure level of a candidate, so it
                   is closer to a test of WHICH residues are exposed). Under a random shuffle of the reference its expected
                   value per candidate is 1, but that does NOT make the across-candidate rho or top-1 null centred on 0 or 0.5
                   (candidates whose error varies more are over-represented among the minima); read the generated null."""
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


def control_errors(entry: Entry, control: str, rng: np.random.Generator, n_rep: int = 1000,
                   others: Optional[List[Entry]] = None, n_other: int = 20, min_residues: int = 5,
                   error_metric: str = "mae") -> List[np.ndarray]:
    """The [K] accessibility-error vector of the candidates for every repetition of one control (a single vector for the
    deterministic controls `real`, `reverse`, `binary`; `n_rep` for shuffle/dropout; one per sampled protein for other_protein).
    Everything downstream (rho, top-1 selection) is computed from these vectors, so every statistic sees the SAME control draws."""
    name, _, param = control.partition(":")
    y, c = entry.y, entry.c

    def err(y_t: np.ndarray, c_t: np.ndarray = c) -> np.ndarray:
        return accessibility_error(y_t, c_t, min_residues, error_metric)

    if name == "real":
        return [err(y)]
    if name == "reverse":
        return [err(reversed_profile(y))]
    if name == "shuffle":
        return [err(shuffled(y, rng)) for _ in range(n_rep)]
    if name == "dropout":
        return [err(keep_fraction(y, float(param), rng)) for _ in range(n_rep)]
    if name == "binary":
        yb, cb = binarise(y, c, "median" if param == "median" else float(param or 0.2))
        return [err(yb, cb)]
    if name == "other_protein":
        pool = [o for o in (others or []) if o.name != entry.name and o.y.shape[1] == y.shape[1]]
        if not pool:
            raise ValueError("other_protein control needs other entries")
        pick = rng.choice(len(pool), size=min(n_other, len(pool)), replace=False)
        return [err(resampled_other(y, pool[i].y)) for i in pick]
    raise ValueError(f"unknown control {control!r}")


def run_control(entry: Entry, control: str, rng: np.random.Generator, n_rep: int = 1000,
                others: Optional[List[Entry]] = None, n_other: int = 20, min_residues: int = 5,
                metrics: Iterable[str] = STRUCT_METRICS, error_metric: str = "mae") -> Dict[str, np.ndarray]:
    """rho(accessibility error, structural error) for one protein under one control.

    Returns {metric: array of rho, one per repetition} (a single value for the deterministic controls
    `real` and `reverse`, and one per sampled protein for `other_protein`)."""
    metrics = tuple(metrics)
    errs = control_errors(entry, control, rng, n_rep, others, n_other, min_residues, error_metric)
    results = [rho_for(entry, e, metrics) for e in errs]
    return {m: np.array([r[m] for r in results]) for m in metrics}


# ------------------------------------------------------------------------------------- top-1 selection
def candidate_outcomes(s: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-candidate outcome of picking it, from the structural errors s [K] of one protein:
    pct   rank percentile of s within the protein, 0 = the most accurate candidate, 1 = the least (average ranks for ties); a
          uniformly random pick has expected pct 0.5 whatever the protein, so it is comparable across proteins;
    delta s minus the protein's mean s (same unit as s: Angstrom for RMSD); a random pick has expected delta 0."""
    k = len(s)
    r = rankdata(s)
    return {"pct": (r - 1.0) / (k - 1.0) if k > 1 else np.full(k, np.nan), "delta": s - np.mean(s)}


def selection_outcome(e: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    """Outcome of picking the candidate with the lowest accessibility error e [K]. Ties for the minimum are averaged (the
    expected outcome of breaking them at random).

    The outcome is read against the FIXED structural ranking of ALL the protein's candidates (finite s), never against only the
    candidates that happen to have a finite control error: if the control leaves the genuinely best candidate undefined, the
    pick cannot be the best and is not reported as such. Needs >= 2 eligible candidates (finite e); NaN otherwise. How many
    were eligible is `n_eligible` in the caller's rows."""
    fs = np.isfinite(s)
    if fs.sum() < 2:
        return {"pct": float("nan"), "delta": float("nan")}
    out = candidate_outcomes(s[fs])
    ee = e[fs]
    ok = np.isfinite(ee)
    if ok.sum() < 2:
        return {"pct": float("nan"), "delta": float("nan")}
    tied = ee[ok] <= ee[ok].min() + 1e-12
    return {"pct": float(out["pct"][ok][tied].mean()), "delta": float(out["delta"][ok][tied].mean())}


def run_selection(entry: Entry, control: str, rng: np.random.Generator, n_rep: int = 1000,
                  others: Optional[List[Entry]] = None, n_other: int = 20, min_residues: int = 5,
                  metrics: Iterable[str] = STRUCT_METRICS, error_metric: str = "mae",
                  errors: Optional[List[np.ndarray]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Top-1 selection under one control: {metric: {"pct": [reps], "delta": [reps]}}. Pass `errors` (from `control_errors`) to
    reuse the draws that were used for rho."""
    metrics = tuple(metrics)
    errs = errors if errors is not None else control_errors(entry, control, rng, n_rep, others, n_other, min_residues, error_metric)
    out = {}
    for m in metrics:
        s = entry.structural_error(m)
        res = [selection_outcome(e, s) for e in errs]
        out[m] = {k: np.array([r[k] for r in res]) for k in ("pct", "delta")}
    return out


def baseline_selection(entry: Entry, rank_scores: Optional[Dict[str, float]] = None,
                       metrics: Iterable[str] = STRUCT_METRICS, higher_is_better: bool = False
                       ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Reference selection rules: {"random": ..., "oracle": ..., "ranking_top1": ...} -> {metric: {"pct", "delta"}}.
    `rank_scores` maps candidate id -> a ranking supplied by the user (default: a rank, 1 = best; with higher_is_better a score,
    larger = best). The ranking is matched to the candidates by id (never by position); ranking_top1 is omitted unless EVERY
    candidate is matched to a finite value. Where the ranking comes from (the model's own ranking score, mean pLDDT, ...) is the
    caller's to state; this function does not assume it."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {"random": {}, "oracle": {}}
    top = None
    if rank_scores and len(entry.candidates) == len(entry.rmsd_ca):
        rows, _, _ = match_ids(entry.candidates, {k: {"v": v} for k, v in rank_scores.items()})
        r = np.array([row["v"] if row is not None else np.nan for row in rows], float)
        if np.isfinite(r).all():
            top = r >= r.max() - 1e-12 if higher_is_better else r <= r.min() + 1e-12
            out["ranking_top1"] = {}
    for m in metrics:
        s = entry.structural_error(m)
        oc = candidate_outcomes(s)
        out["random"][m] = {"pct": 0.5, "delta": 0.0}
        best = s <= np.nanmin(s) + 1e-12
        out["oracle"][m] = {"pct": float(oc["pct"][best].mean()), "delta": float(oc["delta"][best].mean())}
        if top is not None:
            out["ranking_top1"][m] = {"pct": float(oc["pct"][top].mean()), "delta": float(oc["delta"][top].mean())}
    return out


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


def bootstrap_mean(values: np.ndarray, n_boot: int = 10000, rng: Optional[np.random.Generator] = None,
                   alpha: float = 0.05) -> tuple:
    """(mean, lo, hi): percentile bootstrap over proteins of the mean."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def protein_level_null(null_rho: Dict[str, np.ndarray], rng: np.random.Generator, n_draw: int = 10000,
                       statistic: str = "median") -> np.ndarray:
    """Null distribution of the across-protein MEDIAN (or `statistic="mean"`) of a per-protein quantity (rho, top-1 percentile,
    ...): each draw picks one shuffle repetition independently for every protein. `null_rho` maps entry name -> array of
    the quantity under the shuffle."""
    arrays = [a[np.isfinite(a)] for a in null_rho.values()]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return np.array([])
    draws = np.stack([a[rng.integers(0, a.size, size=n_draw)] for a in arrays], axis=1)
    return np.mean(draws, axis=1) if statistic == "mean" else np.median(draws, axis=1)
