#!/usr/bin/env python
"""Stage 4: does accessibility improve candidate selection beyond OpenFold's own confidence?

Per protein there are K candidates (25 = 5 seeds x 5 samples). Three quantities per candidate:
    C  OpenFold confidence  `rank_score` (higher = better)                    used to SELECT
    E  accessibility error  `ac_error`   (lower = better; MAE against the reference accessibility computed from the
                                          experimental structure, mean over probe radii)                          used to SELECT
    R  structural error     `rmsd_ca`    (Angstrom, to the experimental structure)                 used ONLY to EVALUATE, never to select
The candidate set of a protein is fixed once (finite C, E and R) and is identical for every rule and comparison.

Rules, all evaluated per protein with the SAME frozen scaling and lambda:
    openfold      highest C                                                        (= combined at lambda = 0)
    combined      highest  S = z(C) - lambda * z(E)
    access_alone  lowest E                                                         (= combined as lambda -> infinity)
    shuffled      the combined rule with E recomputed against a SHUFFLED reference profile (residue values permuted, same
                  lambda, each draw z-scored the same way); the outcome is the mean over --n-perm shuffle draws
    oracle        lowest R
Ties for the best score are averaged (expected outcome of breaking them at random).

Scaling (fixed in advance, NO fitted parameters): z(.) is the z-score of the K candidates of ONE protein (population sd; a protein
with a constant quantity gets 0 for that term). It puts C (whose spread inside a protein is tiny) and E (a different unit) on the
same scale, so lambda = 1 means equal weight, and it makes lambda comparable across proteins. `--scaling rank` (within-protein
rank percentile) exists for sensitivity checks only. Lambda grid (fixed in advance): 0, 0.1, 0.25, 0.5, 1, 2, 4.

Procedure (three sub-commands; the held-out splits are never read before the rule is frozen):
  gap       the opportunity: OpenFold choice vs oracle, per protein (default: development + validation only).
  select    reads ONLY the development and validation rows, computes the lambda curve on both, picks lambda on the selection
            split(s) (default validation), and writes frozen_rule.json (scaling, grid, lambda, a fingerprint of the rows used).
  evaluate  loads the frozen rule, refuses to run if the development/validation rows changed, then scores every protein, writes
            rerank_per_protein.csv, rerank_summary.csv, rerank_figure.png|svg, README_stage4.md and appends to
            evaluation_log.json (how many times the held-out splits have been evaluated).
Lambda criterion (default): the mean over selection proteins of (structural rank percentile of the combined choice minus that of
the OpenFold choice); lower is better, and lambda = 0 (change 0 by definition) wins unless some lambda is better. It is scale-free
on purpose: the available RMSD improvement is concentrated in a few proteins (see `gap`), so a criterion in Angstrom would be set by
them. `--select-on rmsd` uses the mean RMSD change instead; the one-standard-error rule (default; `--no-one-se` for the plain minimum) takes the smallest lambda within one standard error of the
best (guards against noise-chasing with few selection proteins); `--select-splits development validation` pools both.

History to disclose: Test A/B were inspected in Stage 3 (per-protein correlations, controls and selection plots of accessibility
error against structural error, not reranking results). They are therefore not blind to the accessibility signal; no lambda was
tuned on them, and the frozen file plus evaluation_log.json record this. Do not change the rule after `evaluate`.

    TABLES="--scores-csv development_validation/candidate_scores.csv test/candidate_scores.csv"
    python scripts/rerank_candidates.py gap      $TABLES --out stage4          # development + validation rows only
    python scripts/rerank_candidates.py select   $TABLES --out stage4          # never parses Test A/B rows; freezes lambda
    python scripts/rerank_candidates.py evaluate $TABLES --scores-dir scores_out --out stage4
    python scripts/rerank_candidates.py outcomes --out stage4                    # exploratory, from the SAVED results only (see below)
`outcomes` classifies proteins by the gain available (OpenFold - oracle) and achieved, and splits the little-room ones by an absolute oracle-RMSD
benchmark (0.5 A, expert guidance); it never touches lambda, selection, supporting statistics, the frozen rule or the evaluation log.
Separate tables (--structure-csv, --accessibility-csv, optional --rank-csv) still work. Inputs are checked strictly (25 candidates per
protein; 30/15/20/20 proteins in development/validation/test_a/test_b; every candidate matched and finite; valid split labels):
a failure stops the run unless --allow-incomplete, which records what was dropped. `select` refuses to overwrite a frozen rule
(--new-version archives it as frozen_rule.v<k>.json); `evaluate` stops if the code changed since freezing, if the held-out splits were
already evaluated with this frozen rule (--re-evaluate), or if the shuffled control is missing (--no-shuffle explicitly omits that comparison).
There is no binary success verdict. `evaluate` also generates the exploratory A/B/C-like reports from its saved results.
Evaluations run from another folder or a copy of the files cannot be tracked by the log. Needs numpy, scipy, matplotlib and accessfold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
import csv as _csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import accessfold.controls as _controls
from accessfold.controls import (STRUCT_COLUMNS, COVERAGE_COLUMNS, bootstrap_mean, bootstrap_median, control_errors, load_entry,
                                 match_ids, protein_level_null, read_table)

# Post-evaluation EXPLORATORY reading of the saved per-protein results (`outcomes` sub-command). Not part of the frozen rule or lambda
# or selection. Two thresholds that answer different questions and must not be merged:
#   available-gain threshold  does reranking have meaningful ROOM (OpenFold RMSD - oracle RMSD > threshold)?   a convention, not a discovery: it was
#                             set while the development/validation gap distribution was visible; sensitivity values show how the counts move;
#   absolute-RMSD benchmark   is the best candidate itself materially inaccurate (oracle RMSD >= 0.5 A)? Project-specific expert guidance: for the
#                             SAME protein an RMSD of 0.5 A to the PDB structure is already a lot. It says nothing about how big an improvement must be.
OUTCOME_DEFAULTS = {"substantial_gap_A": 0.5, "sensitivity_gap_A": [0.1, 0.25, 0.5, 1.0], "much_recovered_fraction": 0.5, "oracle_benchmark_A": 0.5}
MATCHED = "held-out, proteins with a shuffled control"
GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
TUNED = ("development", "validation")
SPLIT_LABEL = {"development": "development", "validation": "validation", "test_a": "Test A", "test_b": "Test B"}
TOL = 0.01                          # Angstrom: |change| <= TOL counts as "unchanged"
TIE = 1e-9                         # tie tolerance, on the RAW confidence for lambda = 0 (the OpenFold baseline) and on S for lambda > 0
EXPECT_COUNTS = {"development": 30, "validation": 15, "test_a": 20, "test_b": 20}
EXPECT_CANDIDATES = 25
RULES = ("combined", "access_alone", "shuffled", "oracle")
RULE_LABEL = {"combined": "combined\n(real\naccessibility)", "access_alone": "accessibility\nalone",
              "shuffled": "combined\n(shuffled\naccessibility)", "oracle": "oracle\n(lowest\nRMSD)"}
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e1", band="#c9c8c2",
                  development="#2a78d6", validation="#eb6834", test_a="#16865c", test_b="#a345a0"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8c8b84", grid="#2e2e2c", band="#4a4a46",
                 development="#3987e5", validation="#d95926", test_a="#51c993", test_b="#d78bd0"),
}


# ---------------------------------------------------------------------------------------------------- core rules
def scale(v, how="z"):
    """Within-protein scaling of one quantity over the K candidates (no fitted parameters)."""
    v = np.asarray(v, float)
    if how == "rank":
        r = rankdata(v)
        return (r - 1.0) / (len(v) - 1.0) if len(v) > 1 else np.zeros(len(v))
    sd = v.std()
    return np.zeros(len(v)) if not np.isfinite(sd) or sd < 1e-12 else (v - v.mean()) / sd


def best_mask(score):
    return score >= np.nanmax(score) - TIE


def combined_score(C, E, lam, how="z"):
    """S = scaled C - lam * scaled E (higher = better). lam = 0 depends on C only."""
    return scale(C, how) - lam * scale(E, how)


def combined_mask(C, E, lam, how="z"):
    """Candidates chosen by the combined rule. lambda = 0 IS the OpenFold baseline, so it uses the same raw-confidence tie test
    (scaling would amplify a raw difference below the tolerance, since the spread of C inside a protein is ~1e-3). For lambda > 0 a tie
    is a tie in S; a raw near-tie in C can then be split by the accessibility term, which is what the rule intends."""
    return best_mask(np.asarray(C, float)) if lam == 0 else best_mask(combined_score(C, E, lam, how))


def outcome(R, mask):
    """(mean RMSD, mean rank percentile) of the candidates in `mask` (ties averaged); percentile 0 = best, 1 = worst."""
    k = len(R)
    pct = (rankdata(R) - 1.0) / (k - 1.0) if k > 1 else np.full(k, np.nan)
    return float(R[mask].mean()), float(pct[mask].mean())


def rule_outcomes(p, lam, how, E_shuffled=None):
    """Outcomes of every rule for one protein dict `p` (C, E, R arrays over the fixed candidate set)."""
    C, E, R = p["C"], p["E"], p["R"]
    out = {"openfold": outcome(R, best_mask(C)), "combined": outcome(R, combined_mask(C, E, lam, how)),
           "access_alone": outcome(R, best_mask(-E)), "oracle": outcome(R, R <= R.min() + TIE),
           "mean_candidates": (float(R.mean()), 0.5)}
    if E_shuffled is not None and len(E_shuffled):
        draws = [outcome(R, combined_mask(C, e, lam, how)) for e in E_shuffled]
        out["shuffled_draws"] = np.array(draws)                           # [n_draws, 2] (rmsd, pct)
        out["shuffled"] = tuple(np.mean(out["shuffled_draws"], axis=0))
    return out


# ---------------------------------------------------------------------------------------------------- inputs
def split_of(cid):
    for part in str(cid).replace("\\", "/").split("/"):
        if part in SPLIT_LABEL:
            return part
    return ""


def read_scores(paths, columns, keep_splits, report):
    """{entry: {candidate id: {column: float}}} from one or more CSVs (entry, candidate, ...). Rows of splits not in `keep_splits` are
    skipped BEFORE any value is parsed (only the candidate id / `split` column is looked at to decide), so `select` never parses held-out
    values. The split label is the `split` column if present, else the path component of the candidate id. Returns (table, split_of_entry)."""
    limit = sys.maxsize
    while True:
        try:
            _csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    table, splits, bad_label, ignored = {}, {}, [], set()
    for path in paths:
        with open(path, newline="") as fh:
            rd = _csv.DictReader(fh)
            if not {"entry", "candidate"} <= set(rd.fieldnames or []):
                sys.exit(f"{path}: needs columns 'entry' and 'candidate'")
            have = [c for c in columns if c in rd.fieldnames]
            for r in rd:
                entry, cid = r["entry"], r["candidate"]
                sp = split_of(cid)
                if "split" in r and r["split"]:
                    if sp and r["split"] != sp:
                        sys.exit(f"{path}: split column ({r['split']}) disagrees with the candidate path ({sp}) for {entry} {cid}")
                    sp = r["split"]
                if sp not in SPLIT_LABEL:
                    bad_label.append((entry, cid))
                    continue
                if keep_splits is not None and sp not in keep_splits:
                    ignored.add(entry)
                    continue
                if splits.setdefault(entry, sp) != sp:
                    sys.exit(f"entry {entry} carries candidates of two splits ({splits[entry]} and {sp})")
                row = {}
                for c in have:
                    try:
                        row[c] = float(r[c])
                    except (TypeError, ValueError):
                        row[c] = float("nan")
                if cid in table.setdefault(entry, {}):
                    sys.exit(f"duplicate row for entry {entry!r}, candidate {cid!r} (in {path} or across the supplied tables)")
                table[entry][cid] = row
    report["ignored_entries"] += len(ignored)
    report["bad_split_label"] += bad_label
    return table, splits


def input_paths(a):
    if a.scores_csv:
        return list(a.scores_csv), list(a.scores_csv), False
    return [a.accessibility_csv], [a.structure_csv], True


def load_inputs(a, keep_splits=None):
    """Join the tables by (entry, candidate id) and keep the candidates with finite C, E and R. Returns (proteins, report).
    `keep_splits`: only entries of these splits are read; the others are counted and never parsed. Completeness is checked strictly
    (expected candidates per protein, expected proteins per split, every candidate matched and finite, valid split labels): any failure
    stops the run unless --allow-incomplete, in which case the failures are recorded in report["incomplete"]. A protein dict has
    ids, C, E, R, split, entry."""
    report = {"ignored_entries": 0, "dropped_candidates": {}, "rank_source": set(), "dropped_proteins": {}, "unmatched_structure": 0,
              "rank_conflicts": [], "bad_split_label": [], "incomplete": []}
    acc_paths, st_paths, separate = input_paths(a)
    acc, splits = read_scores(acc_paths, ("ac_error", a.rank_column), keep_splits, report)
    st, _ = read_scores(st_paths, ("rmsd_ca", a.rank_column), keep_splits, {"ignored_entries": 0, "bad_split_label": []})
    rk = read_scores([a.rank_csv], (a.rank_column,), keep_splits, {"ignored_entries": 0, "bad_split_label": []})[0] if a.rank_csv else None
    where = "scores table" if not separate else None
    proteins = []
    problems = [f"candidate without a valid split label: {e} {c}" for e, c in report["bad_split_label"][:5]]
    if report["bad_split_label"]:
        problems.append(f"{len(report['bad_split_label'])} candidates in total have no valid split label (development/validation/test_a/test_b)")
    for entry, rows in acc.items():
        ids = list(rows)
        split = splits[entry]
        srows, counts, _ = match_ids(ids, st.get(entry, {}))
        report["unmatched_structure"] += counts["unmatched"]
        rrows = None
        if rk is not None:
            rrows, _, _ = match_ids(ids, rk.get(entry, {}))
        C, E, R = [], [], []
        for j, (cid, s) in enumerate(zip(ids, srows)):
            r = rows[cid]
            if rrows is not None:
                cands = [("rank-csv", rrows[j])]
            elif where:
                cands = [(where, r)]
            else:
                cands = [("accessibility table", r), ("structure table", s)]
            vals = [(src, row[a.rank_column]) for src, row in cands if row is not None and np.isfinite(row.get(a.rank_column, np.nan))]
            if len({round(v, 12) for _, v in vals}) > 1:
                report["rank_conflicts"].append((entry, cid))
            C.append(vals[0][1] if vals else np.nan)
            if vals:
                report["rank_source"].add(vals[0][0])
            E.append(r.get("ac_error", np.nan))
            R.append(s.get("rmsd_ca", np.nan) if s is not None else np.nan)
        C, E, R = np.array(C, float), np.array(E, float), np.array(R, float)
        ok = np.isfinite(C) & np.isfinite(E) & np.isfinite(R)
        report["dropped_candidates"][entry] = int((~ok).sum())
        if len(ids) != a.expect_candidates:
            problems.append(f"{entry}: {len(ids)} candidates, expected {a.expect_candidates}")
        if counts["unmatched"]:
            problems.append(f"{entry}: {counts['unmatched']} candidates have no structure row")
        if (~ok).sum() and not counts["unmatched"]:
            problems.append(f"{entry}: {int((~ok).sum())} candidates with a missing/non-finite C, E or R")
        if ok.sum() < a.min_candidates:
            report["dropped_proteins"][entry] = f"{int(ok.sum())} usable candidates"
            problems.append(f"{entry}: only {int(ok.sum())} usable candidates (minimum {a.min_candidates})")
            continue
        proteins.append({"entry": entry, "split": split, "ids": [i for i, k in zip(ids, ok) if k], "C": C[ok], "E": E[ok], "R": R[ok]})
    if report["rank_conflicts"]:
        sys.exit(f"{a.rank_column} differs between tables for {len(report['rank_conflicts'])} candidates (first: {report['rank_conflicts'][0]}); "
                 "pass --rank-csv to say which one to use")
    n_by = {}
    for e, sp in splits.items():
        n_by[sp] = n_by.get(sp, 0) + 1
    for sp, want in a.expect_counts.items():
        if (keep_splits is None or sp in keep_splits) and n_by.get(sp, 0) != want:
            problems.append(f"split {sp}: {n_by.get(sp, 0)} proteins, expected {want}")
    report["incomplete"] = problems
    if problems and not a.allow_incomplete:
        sys.exit("STOP: the inputs are not complete (pass --expect-counts/--expect-candidates if the design differs, or --allow-incomplete to drop "
                 "and record the failures):\n  " + "\n  ".join(problems[:12]) + (f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""))
    return proteins, report


def describe_inputs(proteins, report, a):
    ks = sorted({len(p["R"]) for p in proteins})
    ties_top = sum(int(best_mask(p["C"]).sum() > 1) for p in proteins)
    drop = {k: v for k, v in report["dropped_candidates"].items() if v}
    by = {}
    for p in proteins:
        by[p["split"]] = by.get(p["split"], 0) + 1
    print(f"inputs: {len(proteins)} proteins ({', '.join(str(k or 'unlabelled') + ': ' + str(v) for k, v in sorted(by.items()))}); candidates per protein {ks}")
    print(f"  C = {a.rank_column} (higher = better; from {', '.join(sorted(report['rank_source'])) or 'NOWHERE'}), E = ac_error (lower = better), "
          f"R = rmsd_ca (evaluation only)")
    if not report["rank_source"]:
        sys.exit(f"no finite {a.rank_column} in any table (checked {', '.join(map(str, input_paths(a)[0] + input_paths(a)[1]))}); pass --rank-csv")
    print(f"  proteins with tied top confidence: {ties_top} (ties averaged); candidates dropped for a missing value: {sum(drop.values())} "
          f"in {len(drop)} proteins; candidates without a structure row: {report['unmatched_structure']}")
    if report["ignored_entries"]:
        print(f"  {report['ignored_entries']} entries of splits other than {sorted(by)} were NOT read (held out)")
    if report["dropped_proteins"]:
        print(f"  proteins dropped: {report['dropped_proteins']}")
    for name, key in ((a.rank_column, "C"), ("ac_error", "E")):
        sp = np.array([np.ptp(p[key]) for p in proteins])
        flat = [p["entry"] for p in proteins if np.ptp(p[key]) < 1e-6]
        print(f"  within-protein range of {name}: median {np.median(sp):.4g}, min {sp.min():.4g}, max {sp.max():.4g}"
              + (f"; NEAR-CONSTANT in {len(flat)} proteins ({', '.join(flat[:5])}): that term carries no information there" if flat else ""))
    if report["incomplete"]:
        print(f"  INCOMPLETE inputs accepted by --allow-incomplete ({len(report['incomplete'])} problems; recorded): {report['incomplete'][:3]}")


def fingerprint(proteins):
    h = hashlib.sha1()
    for p in sorted(proteins, key=lambda q: q["entry"]):
        for cid, c, e, r in sorted(zip(p["ids"], p["C"], p["E"], p["R"])):
            h.update(f"{p['entry']}|{cid}|{c!r}|{e!r}|{r!r}\n".encode())
    return h.hexdigest()


def code_fingerprint():
    h = hashlib.sha1()
    for f in (Path(__file__).resolve(), Path(_controls.__file__).resolve()):
        h.update(f.read_bytes())
    return h.hexdigest()


def frozen_hash(fz):
    return hashlib.sha1(json.dumps({k: v for k, v in fz.items() if k != "frozen_hash"}, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------- summaries
def ratio_ci(num, den, n_boot, seed=0):
    """Ratio of sums (pooled share of the available gap that is closed) with a percentile bootstrap over proteins."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.size == 0 or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, num.size, size=(n_boot, num.size))
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(d > 0, num[idx].sum(axis=1) / d, np.nan)
    return float(num.sum() / den.sum()), float(np.nanquantile(r, 0.025)), float(np.nanquantile(r, 0.975))


def counts(delta):
    d = np.asarray(delta, float)
    return int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()), int((d > TOL).sum())


# ---------------------------------------------------------------------------------------------------- gap
def cmd_gap(a):
    keep = None if a.include_test else TUNED
    proteins, report = load_inputs(a, keep)
    describe_inputs(proteins, report, a)
    rows = []
    for p in proteins:
        o = rule_outcomes(p, 0.0, a.scaling)
        rows.append({"entry": p["entry"], "split": p["split"], "n_candidates": len(p["R"]), "rmsd_range_A": float(np.ptp(p["R"])),
                     "rmsd_openfold": o["openfold"][0], "rmsd_oracle": o["oracle"][0], "gap_A": o["openfold"][0] - o["oracle"][0],
                     "rmsd_mean_candidates": o["mean_candidates"][0], "pct_openfold": o["openfold"][1],
                     "range_confidence": float(np.ptp(p["C"])), "range_ac_error": float(np.ptp(p["E"]))})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "gap_per_protein.csv", rows)
    gap = np.array([r["gap_A"] for r in rows])
    order = np.sort(gap)[::-1]
    print(f"\nAvailable improvement = OpenFold-choice RMSD - oracle RMSD ({len(gap)} proteins; Angstrom)")
    for sp in sorted({r["split"] for r in rows}) + ["all"]:
        g = np.array([r["gap_A"] for r in rows if sp == "all" or r["split"] == sp])
        m, lo, hi = bootstrap_mean(g, 5000)
        print(f"  {sp:<12} n={len(g):>3}  mean {m:.3f} [{lo:.3f}, {hi:.3f}]  median {np.median(g):.3f}  zero {int((g <= TOL).sum()):>2}  "
              f">0.1 A {int((g > 0.1).sum()):>2}  >0.5 A {int((g > 0.5).sum()):>2}  >1 A {int((g > 1).sum()):>2}  max {g.max():.2f}")
    print(f"  the 3 largest gaps ({', '.join(f'{x:.2f}' for x in order[:3])} A) are {100 * order[:3].sum() / gap.sum():.0f} % of the total {gap.sum():.1f} A")
    bm = OUTCOME_DEFAULTS["oracle_benchmark_A"]
    print(f"  oracle RMSD >= {bm:g} A (best candidate still materially inaccurate, expert benchmark): {sum(r['rmsd_oracle'] >= bm for r in rows)} of {len(rows)}; "
          f"OpenFold choice >= {bm:g} A: {sum(r['rmsd_openfold'] >= bm for r in rows)} of {len(rows)}")
    for thr in OUTCOME_DEFAULTS["sensitivity_gap_A"]:
        small = [r for r in rows if r["gap_A"] <= thr]
        print(f"  available gain > {thr:g} A: {int((gap > thr).sum())} of {len(gap)}; with little room ({len(small)}): oracle >= {bm:g} A in "
              f"{sum(r['rmsd_oracle'] >= bm for r in small)}, below in {sum(r['rmsd_oracle'] < bm for r in small)}")
    pe = np.array([r["pct_openfold"] for r in rows])
    print(f"  OpenFold's choice has mean structural rank percentile {pe.mean():.3f} (0 best; 0.5 is the expectation of a random pick)")
    print(f"wrote {out}/gap_per_protein.csv")


# ---------------------------------------------------------------------------------------------------- select
def lambda_curve(proteins, a):
    """{split: {lam: {"pct": [per protein change], "rmsd": [per protein change]}}} for the real combined rule vs OpenFold."""
    curve = {}
    for p in proteins:
        base = outcome(p["R"], best_mask(p["C"]))
        for lam in GRID:
            o = outcome(p["R"], combined_mask(p["C"], p["E"], lam, a.scaling))
            d = curve.setdefault(p["split"], {}).setdefault(lam, {"pct": [], "rmsd": []})
            d["rmsd"].append(o[0] - base[0])
            d["pct"].append(o[1] - base[1])
    return curve


def choose_lambda(curve, a):
    key = a.select_on
    pooled = {lam: np.concatenate([curve[s][lam][key] for s in a.select_splits if s in curve]) for lam in GRID}
    means = np.array([pooled[lam].mean() for lam in GRID])
    best = int(np.argmin(means))                    # np.argmin returns the first minimum: ties go to the smaller lambda
    if a.one_se:
        se = pooled[GRID[best]].std(ddof=1) / np.sqrt(len(pooled[GRID[best]])) if len(pooled[GRID[best]]) > 1 else 0.0
        best = int(np.flatnonzero(means <= means[best] + se)[0])
    return GRID[best], means


def cmd_select(a):
    out = Path(a.out)
    old_fp, version, supersedes = out / "frozen_rule.json", 1, None
    if old_fp.exists():
        old = json.loads(old_fp.read_text())
        if not a.new_version:
            sys.exit(f"STOP: {old_fp} already exists (version {old.get('version', 1)}, lambda {old['lambda']:g}). A frozen rule is never overwritten; "
                     "pass --new-version to archive it and freeze a new, disclosed iteration")
        version, supersedes = old.get("version", 1) + 1, old.get("frozen_hash")
    proteins, report = load_inputs(a, TUNED)
    describe_inputs(proteins, report, a)
    missing = [s for s in a.select_splits if not any(p["split"] == s for p in proteins)]
    if missing:
        sys.exit(f"no proteins in the selection split(s) {missing}")
    curve = lambda_curve(proteins, a)
    lam, means = choose_lambda(curve, a)
    n_sel = sum(1 for p in proteins if p["split"] in a.select_splits)
    unit = "mean rank-percentile change" if a.select_on == "pct" else "mean RMSD change (A)"
    print(f"\nlambda curve ({unit} of the combined choice vs the OpenFold choice; negative = better; scaling {a.scaling})")
    print(f"  {'lambda':>7}" + "".join(f"{SPLIT_LABEL.get(s, s):>14}" for s in sorted(curve)) + f"{'   selection':>12}")
    for i, g in enumerate(GRID):
        print(f"  {g:>7g}" + "".join(f"{np.mean(curve[s][g][a.select_on]):>14.4f}" for s in sorted(curve)) + f"{means[i]:>12.4f}" + ("   <- chosen" if g == lam else ""))
    print(f"\nchosen lambda = {lam:g} on {n_sel} proteins of {', '.join(a.select_splits)} (select-on {a.select_on}{', one-SE rule' if a.one_se else ''})"
          + ("; the selection procedure retained lambda = 0 (OpenFold alone); with one-SE this does not imply every positive lambda had a worse mean" if lam == 0 else ""))
    frozen = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scaling": a.scaling, "grid": list(GRID), "lambda": lam,
              "select_on": a.select_on, "select_splits": list(a.select_splits), "one_se": a.one_se, "n_selection_proteins": n_sel,
              "n_tuned_proteins": len(proteins), "rank_column": a.rank_column, "rank_source": sorted(report["rank_source"]),
              "tie_rule": "ties for the best score are averaged; tolerance 1e-9 on the raw confidence for lambda = 0 (= the OpenFold baseline) "
                          "and on S for lambda > 0", "tie_tolerance": TIE, "tolerance_A": TOL, "version": version, "supersedes": supersedes,
              "reporting_scope": "Descriptive A/B/C-like outcome analysis; RMSD intervals and shuffle comparisons are supporting evidence, not a binary verdict.",
              "code_fingerprint": code_fingerprint(),
              "input_policy": {"expect_candidates": a.expect_candidates, "expect_counts": a.expect_counts, "allow_incomplete": a.allow_incomplete,
                               "min_candidates": a.min_candidates, "incomplete_problems_recorded": report["incomplete"],
                               "inputs": [Path(x).name for x in (input_paths(a)[0] + input_paths(a)[1] + ([a.rank_csv] if a.rank_csv else []))]},
              "tuned_rows_fingerprint": fingerprint(proteins),
              "curve": {s: {str(g): {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["pct"])} for g, d in c.items()} for s, c in curve.items()},
              "heldout_entries_not_read": report["ignored_entries"],
              "disclosure": "Test A and Test B were inspected in Stage 3 (per-protein correlations, controls and selection plots of accessibility error "
                            "against structural error). No reranking result on them was seen and no lambda was tuned on them. The scaling, grid, lambda "
                            "and decision rule are frozen here; the held-out splits are evaluated with `evaluate`, and every evaluation is logged."}
    out.mkdir(parents=True, exist_ok=True)
    if supersedes:
        old_fp.rename(out / f"frozen_rule.v{version - 1}.json")
    frozen["frozen_hash"] = frozen_hash(frozen)
    (out / "frozen_rule.json").write_text(json.dumps(frozen, indent=1))
    rows = [{"split": s, "lambda": g, "n": len(d["pct"]), "mean_pct_change": float(np.mean(d["pct"])), "mean_rmsd_change_A": float(np.mean(d["rmsd"])),
             "median_rmsd_change_A": float(np.median(d["rmsd"]))} for s, c in sorted(curve.items()) for g, d in c.items()]
    write_csv(out / "lambda_curve.csv", rows)
    print(f"wrote {out}/frozen_rule.json (version {version}, hash {frozen['frozen_hash'][:10]}) and lambda_curve.csv. Next: evaluate (reads the held-out splits).")


# ---------------------------------------------------------------------------------------------------- evaluate
def shuffle_draws(p, a, struct_table, scores_dir):
    """[n_perm] vectors of shuffled-reference E aligned to the protein's fixed candidate set, or (None, reason)."""
    f = Path(scores_dir) / f"{p['entry']}.npz"
    if not f.exists():
        return None, f"no {f.name} in {scores_dir}"
    try:
        e = load_entry(f, a.min_coverage, struct_table)
    except ValueError as exc:
        return None, str(exc)
    pos, _, _ = match_ids(p["ids"], {cid: {"i": float(i)} for i, cid in enumerate(e.candidates)})
    if any(r is None for r in pos):
        return None, "candidates of the fixed set are missing from the score file"
    idx = np.array([int(r["i"]) for r in pos])
    real = control_errors(e, "real", np.random.default_rng(0), error_metric="mae")[0][idx]
    if not np.allclose(real, p["E"], rtol=1e-4, atol=1e-6, equal_nan=False):
        sys.exit(f"STOP: {p['entry']}: ac_error in {', '.join(map(str, input_paths(a)[0]))} differs from the value recomputed from the profiles in {f.name} "
                 f"(max abs difference {np.abs(real - p['E']).max():.3g}); real and shuffled errors would come from different calculations")
    rng = np.random.default_rng([a.seed, zlib.crc32(p["entry"].encode())])
    draws = control_errors(e, "shuffle", rng, n_rep=a.n_perm, error_metric="mae")
    return [d[idx] for d in draws if np.isfinite(d[idx]).all()], None


def group_summary(rows, name, rules, a, null_arrays):
    """Summary rows for one group of per-protein result rows."""
    out = []
    if not rows:
        return out
    gap = np.array([r["gap_A"] for r in rows])
    for rule in rules:
        key = f"delta_{rule}"
        d = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))], float)
        if d.size == 0:
            continue
        sel = [r for r in rows if np.isfinite(r.get(key, np.nan))]
        m, lo, hi = bootstrap_mean(d, a.n_boot, np.random.default_rng(1))
        md, mlo, mhi = bootstrap_median(d, a.n_boot, np.random.default_rng(2))
        imp, same, wor = counts(d)
        closed = np.array([-r[key] for r in sel])
        cr, clo, chi = ratio_ci(closed, np.array([r["gap_A"] for r in sel]), a.n_boot)
        pc = np.array([r[f"pct_change_{rule}"] for r in sel])
        big = [r for r in sel if r["gap_A"] > TOL]
        fr = np.array([-r[key] / r["gap_A"] for r in big])
        row = {"group": name, "rule": rule, "n_proteins": d.size, "mean_delta_A": m, "mean_lo": lo, "mean_hi": hi, "median_delta_A": md,
               "median_lo": mlo, "median_hi": mhi, "n_improved": imp, "n_unchanged": same, "n_worsened": wor, "mean_pct_change": float(pc.mean()),
               "gap_closed_share": cr, "gap_closed_lo": clo, "gap_closed_hi": chi, "available_gap_mean_A": float(np.mean([r["gap_A"] for r in sel])),
               "n_with_gap": len(big), "gap_closed_median_per_protein": float(np.median(fr)) if len(fr) else float("nan"),
               "gap_closed_mean_per_protein": float(np.mean(fr)) if len(fr) else float("nan")}
        if rule == "combined" and null_arrays:
            arrs = {r["entry"]: null_arrays[r["entry"]] for r in rows if r["entry"] in null_arrays}
            if arrs:
                obs = float(np.mean([r["delta_combined"] for r in rows if r["entry"] in arrs]))
                nul = protein_level_null(arrs, np.random.default_rng(3), 20000, statistic="mean")
                row["n_proteins_vs_shuffle"] = len(arrs)
                row["shuffled_null_mean_A"] = float(nul.mean())
                row["shuffled_null_lo"], row["shuffled_null_hi"] = float(np.quantile(nul, 0.025)), float(np.quantile(nul, 0.975))
                row["p_combined_better_than_shuffled"] = float((1 + (nul <= obs).sum()) / (1 + nul.size))
                row["mean_combined_minus_shuffled_A"] = obs - float(np.mean([np.mean(arrs[e]) for e in arrs]))
        out.append(row)
    return out


def cmd_evaluate(a):
    out = Path(a.out)
    fp = out / "frozen_rule.json"
    if not fp.exists():
        sys.exit(f"{fp} not found: run `select` first (the rule must be frozen before the held-out splits are scored)")
    fz = json.loads(fp.read_text())
    if fz.get("frozen_hash") != frozen_hash(fz):
        sys.exit("STOP: frozen_rule.json was edited after it was written (hash mismatch)")
    a.scaling, lam = fz["scaling"], fz["lambda"]
    if not a.scores_dir and not a.no_shuffle:
        sys.exit("STOP: pass --scores-dir for the shuffled-accessibility comparison, or explicitly use --no-shuffle to omit it")
    code_now = code_fingerprint()
    code_changed = code_now != fz.get("code_fingerprint")
    if code_changed and not a.allow_code_change:
        sys.exit("STOP: rerank_candidates.py / accessfold/controls.py differ from the code that froze this rule; evaluate with the frozen code, "
                 "or pass --allow-code-change (logged) after checking the change cannot affect the rule")
    log = out / "evaluation_log.json"
    hist = json.loads(log.read_text()) if log.exists() else []
    if any(h.get("frozen_hash") == fz["frozen_hash"] for h in hist) and not a.re_evaluate:
        sys.exit(f"STOP: the held-out splits were already evaluated with this frozen rule ({fz['frozen_hash'][:10]}, see evaluation_log.json); "
                 "pass --re-evaluate to repeat (logged). Evaluations run in another folder or from a copy cannot be tracked here")
    tuned, report_t = load_inputs(a, TUNED)
    if fingerprint(tuned) != fz["tuned_rows_fingerprint"]:
        sys.exit("STOP: the development/validation rows differ from the ones the rule was frozen on (inputs changed); rerun `select`, "
                 "and treat that as a new, disclosed iteration")
    proteins, report = load_inputs(a, None)
    describe_inputs(proteins, report, a)
    struct_table = {}
    if a.scores_dir:
        for pth in input_paths(a)[1]:
            for ent, rws in read_table(pth, STRUCT_COLUMNS + COVERAGE_COLUMNS).items():
                struct_table.setdefault(ent, {}).update(rws)
    null_arrays, results, skipped = {}, [], {}
    for p in proteins:
        draws = None
        if a.scores_dir:
            draws, why = shuffle_draws(p, a, struct_table, a.scores_dir)
            if draws is None or not len(draws):
                skipped[p["entry"]] = why or "no finite shuffle draw"
                draws = None
        o = rule_outcomes(p, lam, a.scaling, draws)
        base_r, base_p = o["openfold"]
        row = {"entry": p["entry"], "split": p["split"], "tuned": p["split"] in TUNED, "n_candidates": len(p["R"]), "rmsd_range_A": float(np.ptp(p["R"])),
               "lambda": lam, "scaling": a.scaling, "rmsd_openfold": base_r, "rmsd_oracle": o["oracle"][0], "gap_A": base_r - o["oracle"][0],
               "rmsd_mean_candidates": o["mean_candidates"][0]}
        for rule in ("combined", "access_alone", "shuffled", "oracle"):
            if rule in o:
                row[f"rmsd_{rule}"] = o[rule][0]
                row[f"delta_{rule}"] = o[rule][0] - base_r
                row[f"pct_{rule}"] = o[rule][1]
                row[f"pct_change_{rule}"] = o[rule][1] - base_p
            else:
                row[f"rmsd_{rule}"] = row[f"delta_{rule}"] = row[f"pct_{rule}"] = row[f"pct_change_{rule}"] = np.nan
        row["pct_openfold"] = base_p
        for rule in ("combined", "access_alone", "shuffled"):
            if np.isfinite(row.get(f"delta_{rule}", np.nan)):
                row[f"achieved_gain_{rule}"] = -row[f"delta_{rule}"]                          # OpenFold RMSD - reranked RMSD
                row[f"remaining_gap_{rule}"] = row[f"rmsd_{rule}"] - o["oracle"][0]           # reranked RMSD - oracle RMSD
        row["changed_choice"] = bool(not np.array_equal(combined_mask(p["C"], p["E"], lam, a.scaling), best_mask(p["C"])))
        if "shuffled_draws" in o:
            dr = o["shuffled_draws"][:, 0]
            row["delta_shuffled_sd"] = float(np.std(dr - base_r))
            row["p_perm_combined_vs_shuffled"] = float((1 + (dr <= o["combined"][0] + 1e-12).sum()) / (1 + len(dr)))
            null_arrays[p["entry"]] = dr - base_r
        results.append(row)
    if skipped and not a.allow_incomplete:
        sys.exit(f"STOP: shuffled control undefined for {len(skipped)} proteins: {dict(list(skipped.items())[:5])}. "
                 "Fix the score files or use --allow-incomplete to report the shuffle comparison on matched proteins only")
    if skipped:
        print(f"shuffled control undefined for {len(skipped)} proteins (recorded; shuffle comparisons use matched proteins): {dict(list(skipped.items())[:5])}")
    if a.scores_dir is None:
        print("NOTE: --no-shuffle: the shuffled-accessibility rule and its comparison are NOT computed")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "rerank_per_protein.csv", results)

    groups = {"tuned (development + validation)": [r for r in results if r["tuned"]]}
    for s in ("development", "validation", "test_a", "test_b"):
        groups[SPLIT_LABEL[s]] = [r for r in results if r["split"] == s]
    groups["held-out (Test A + Test B)"] = [r for r in results if r["split"] in ("test_a", "test_b")]
    groups[MATCHED] = [r for r in results if r["split"] in ("test_a", "test_b") and r["entry"] in null_arrays]
    summary = []
    for name, rs in groups.items():
        summary += group_summary(rs, name, RULES, a, null_arrays)
    write_csv(out / "rerank_summary.csv", summary)

    hist.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "lambda": lam, "frozen_hash": fz["frozen_hash"],
                 "frozen_fingerprint": fz["tuned_rows_fingerprint"], "n_proteins": len(results), "heldout_proteins": len(groups["held-out (Test A + Test B)"]),
                 "matched_heldout_proteins": len(groups[MATCHED]), "code_changed": code_changed, "re_evaluate": bool(a.re_evaluate), "no_shuffle": bool(a.no_shuffle),
                 "allow_incomplete": bool(a.allow_incomplete)})
    log.write_text(json.dumps(hist, indent=1))
    n_eval = len(hist)

    print_summary(summary, lam, n_eval)
    plt, c = setup(a.theme)
    fig = figure(results, summary, fz, a, plt, c)
    fig.savefig(out / "rerank_figure.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / "rerank_figure.svg", bbox_inches="tight", pad_inches=0.15)
    (out / "README_stage4.md").write_text(readme(fz, summary, n_eval, results, report, skipped, a))
    print(f"\nwrote {out}/rerank_per_protein.csv, rerank_summary.csv, rerank_figure.png|svg, README_stage4.md, evaluation_log.json "
          f"(held-out splits evaluated {n_eval} time(s))")
    cmd_outcomes(a)


def outcome_analysis(groups, rules, orule, n_boot):
    """EXPLORATORY. Rows per group, rule and available-gain threshold: how much gain was available, how much the rule recovered, and the
    A-/B-/C-like counts; C-like proteins are split by the absolute-RMSD benchmark of the oracle (oracle >= benchmark: even the best candidate is
    materially inaccurate; below: under the project benchmark, not automatically correct)."""
    out, bm = [], orule["oracle_benchmark_A"]
    for gname, rows in groups.items():
        if not rows:
            continue
        gap = np.array([r["gap_A"] for r in rows])
        for thr in orule["sensitivity_gap_A"]:
            big = gap > thr
            m, lo, hi = bootstrap_mean(big.astype(float), n_boot, np.random.default_rng(4))
            for rule in rules:
                key = f"achieved_gain_{rule}"
                sel = [r for r in rows if np.isfinite(r.get(key, np.nan))]
                if not sel:
                    continue
                g = np.array([r["gap_A"] for r in sel])
                ach = np.array([r[key] for r in sel])
                orc_all = np.array([r["rmsd_oracle"] for r in sel])
                bg = g > thr
                frac = ach[bg] / g[bg]
                a_like = int((frac >= orule["much_recovered_fraction"]).sum())
                orc_c = orc_all[~bg]
                out.append({"group": gname, "rule": rule, "gap_threshold_A": thr, "primary": thr == orule["substantial_gap_A"], "n_proteins": len(sel),
                            "n_gap_above": int(bg.sum()), "share_gap_above": float(bg.mean()), "share_lo": lo, "share_hi": hi,
                            "A_like": a_like, "B_like": int(bg.sum()) - a_like, "C_like": int((~bg).sum()),
                            "C_like_oracle_ge_benchmark": int((orc_c >= bm).sum()), "C_like_oracle_below_benchmark": int((orc_c < bm).sum()),
                            "C_like_median_oracle_rmsd_A": float(np.median(orc_c)) if len(orc_c) else float("nan"),
                            "C_like_max_oracle_rmsd_A": float(orc_c.max()) if len(orc_c) else float("nan"),
                            "oracle_benchmark_A": bm, "n_oracle_ge_benchmark_all": int((orc_all >= bm).sum()),
                            "n_openfold_ge_benchmark_all": int((np.array([r["rmsd_openfold"] for r in sel]) >= bm).sum()),
                            "median_available_gain_A": float(np.median(g)), "median_achieved_gain_A": float(np.median(ach)),
                            "median_recovered_fraction_when_gap_above": float(np.median(frac)) if len(frac) else float("nan"),
                            "pooled_recovered_fraction_when_gap_above": float(ach[bg].sum() / g[bg].sum()) if bg.any() else float("nan"),
                            "median_remaining_gap_A": float(np.median([r[f"remaining_gap_{rule}"] for r in sel]))})
    return out


def bottleneck_text(oa, orule, group):
    """Plain description of where the gain is lost for one group at the primary available-gain threshold (no label is forced on the dataset)."""
    c = next((r for r in oa if r["group"] == group and r["rule"] == "combined" and r["primary"]), None)
    if not c:
        return None
    thr, bm = orule["substantial_gap_A"], orule["oracle_benchmark_A"]
    return (f"{c['n_gap_above']} of {c['n_proteins']} proteins had an available gain above {thr:g} A (share {c['share_gap_above']:.2f} "
            f"[{c['share_lo']:.2f}, {c['share_hi']:.2f}]). Among them the combined rule recovered at least {orule['much_recovered_fraction']:.0%} of the gain in {c['A_like']} "
            f"(A-like) and less in {c['B_like']} (B-like; median fraction recovered {c['median_recovered_fraction_when_gap_above']:.2f}, pooled {c['pooled_recovered_fraction_when_gap_above']:.2f}). "
            f"The other {c['C_like']} had little room (C-like): in {c['C_like_oracle_ge_benchmark']} even the best candidate has RMSD >= {bm:g} A (limited room for reranking, "
            f"and the best candidate is still materially inaccurate: better generation is worth investigating); in {c['C_like_oracle_below_benchmark']} the best candidate is below "
            f"{bm:g} A (limited room, best candidate under the project benchmark, which is not automatically correct). "
            f"Across all {c['n_proteins']} proteins the oracle is >= {bm:g} A in {c['n_oracle_ge_benchmark_all']} and OpenFold's own choice in {c['n_openfold_ge_benchmark_all']}.")


def cmd_outcomes(a):
    """Exploratory classification from the SAVED rerank_per_protein.csv: no data re-read, no lambda, selection or shuffle touched, the frozen
    rule and evaluation log are not modified."""
    import csv
    out = Path(a.out)
    src = out / "rerank_per_protein.csv"
    if not src.exists():
        sys.exit(f"{src} not found: run `evaluate` first")
    fz = json.loads((out / "frozen_rule.json").read_text()) if (out / "frozen_rule.json").exists() else {}
    rows = []
    for r in csv.DictReader(open(src)):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v) if v != "" else float("nan")
            except ValueError:
                d[k] = {"True": True, "False": False}.get(v, v)
        rows.append(d)
    derived = set()
    for d in rows:                                    # tables from an earlier evaluation lack these columns: derive them from the saved RMSDs
        if "gap_A" not in d and {"rmsd_openfold", "rmsd_oracle"} <= set(d):
            d["gap_A"] = d["rmsd_openfold"] - d["rmsd_oracle"]
            derived.add("gap_A")
        for rule in ("combined", "access_alone", "shuffled"):
            if f"rmsd_{rule}" in d and np.isfinite(d[f"rmsd_{rule}"]):
                for col, val in ((f"achieved_gain_{rule}", d["rmsd_openfold"] - d[f"rmsd_{rule}"]), (f"remaining_gap_{rule}", d[f"rmsd_{rule}"] - d["rmsd_oracle"])):
                    if not np.isfinite(d.get(col, float("nan"))):
                        d[col] = val
                        derived.add(col)
    need = {"entry", "split", "tuned", "gap_A", "rmsd_openfold", "rmsd_oracle", "rmsd_combined", "achieved_gain_combined", "remaining_gap_combined"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{src} lacks the columns {sorted(need - set(rows[0] if rows else []))} and they cannot be derived")
    if derived:
        print(f"derived from the saved RMSDs (not rerun): {', '.join(sorted(derived))}")
    orule = dict(OUTCOME_DEFAULTS, substantial_gap_A=a.gap_threshold, much_recovered_fraction=a.recovered_fraction, oracle_benchmark_A=a.oracle_benchmark)
    if a.gap_threshold not in orule["sensitivity_gap_A"]:
        orule["sensitivity_gap_A"] = sorted(orule["sensitivity_gap_A"] + [a.gap_threshold])
    groups = {"tuned (development + validation)": [r for r in rows if r["tuned"] is True]}
    for sp in ("development", "validation", "test_a", "test_b"):
        groups[SPLIT_LABEL[sp]] = [r for r in rows if r["split"] == sp]
    groups["held-out (Test A + Test B)"] = [r for r in rows if r["split"] in ("test_a", "test_b")]
    oa = outcome_analysis(groups, ("combined", "access_alone"), orule, a.n_boot)
    write_csv(out / "outcome_analysis.csv", oa)
    bm, thr, fr = orule["oracle_benchmark_A"], orule["substantial_gap_A"], orule["much_recovered_fraction"]
    per = []
    for r in rows:
        big = r["gap_A"] > thr
        cls = ("A-like" if r["achieved_gain_combined"] / r["gap_A"] >= fr else "B-like") if big else "C-like"
        per.append({"entry": r["entry"], "split": r["split"], "rmsd_openfold": r["rmsd_openfold"], "rmsd_combined": r["rmsd_combined"], "rmsd_oracle": r["rmsd_oracle"],
                    "available_gain_A": r["gap_A"], "achieved_gain_A": r["achieved_gain_combined"], "remaining_gap_A": r["remaining_gap_combined"],
                    "recovered_fraction": r["achieved_gain_combined"] / r["gap_A"] if big else float("nan"), "class": cls,
                    "oracle_ge_benchmark": bool(r["rmsd_oracle"] >= bm), "openfold_ge_benchmark": bool(r["rmsd_openfold"] >= bm)})
    write_csv(out / "outcome_per_protein.csv", per)
    lines = ["# Stage 4 outcome reporting (EXPLORATORY, post-evaluation)", "",
             "Computed from the saved `rerank_per_protein.csv` of the frozen evaluation"
             + (f" (lambda {fz['lambda']:g}, frozen rule version {fz.get('version', 1)}, hash {fz.get('frozen_hash', 'n/a (older frozen file)')[:10]})" if fz else "")
             + (f". Columns derived from the saved RMSDs: {', '.join(sorted(derived))}" if derived else "")
             + ". Nothing here changes lambda, selection or the supporting statistics in `README_stage4.md`. This report classifies available and recovered gain; "
               "the thresholds below were not declared before the evaluation.", "",
             "## Two thresholds that answer different questions",
             f"- **Available-gain threshold ({thr:g} A, sensitivity {orule['sensitivity_gap_A']}):** does reranking have meaningful room, i.e. OpenFold RMSD minus oracle RMSD above it? "
             "A convention fixed while the development/validation gap distribution was visible. An improvement smaller than this threshold can still matter.",
             f"- **Absolute-RMSD benchmark ({bm:g} A):** project-specific expert guidance: for the SAME protein an RMSD of 0.5 A to the PDB structure is already a lot"
             + ("" if bm == 0.5 else f" (this run overrides the value with {bm:g} A)") + ". "
             "It flags a materially inaccurate structure; it says nothing about how large an improvement must be, so it is not used as the gain threshold.", "",
             "## Where is the gain lost? (combined rule vs OpenFold; A-like = substantial gain mostly recovered, B-like = substantial gain largely missed or worsened, C-like = little room)"]
    for g in ("held-out (Test A + Test B)", "tuned (development + validation)"):
        bt = bottleneck_text(oa, orule, g)
        if bt:
            lines.append(f"- **{g}**{' (lambda was fitted here)' if g.startswith('tuned') else ''}: {bt}")
    lines += ["", "Interpretation of C-like proteins: with little room, reranking cannot help much. If the oracle is still >= the benchmark, the candidate pool may be inadequate "
              "for that protein (guided generation is worth investigating); if it is below the benchmark, the pool is already under the project benchmark, which is not the same as correct. "
              "Different proteins can fall in different classes; no single label is given to the dataset.", "",
              "Files: `outcome_analysis.csv` (counts by group, rule and threshold), `outcome_per_protein.csv` (class and oracle RMSD per protein)."]
    (out / "README_outcomes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:-3]))
    print(f"wrote {out}/outcome_analysis.csv, outcome_per_protein.csv, README_outcomes.md (frozen_rule.json and evaluation_log.json untouched)")


def print_summary(summary, lam, n_eval):
    print(f"\nfrozen lambda = {lam:g}.  Change in RMSD vs the OpenFold choice (A; negative = better); counts use |change| <= {TOL} A as unchanged")
    for r in summary:
        if r["rule"] == "oracle":
            continue
        print(f"  {r['group']:<34} {r['rule']:<13} n={r['n_proteins']:>3}  mean {r['mean_delta_A']:+.3f} [{r['mean_lo']:+.3f}, {r['mean_hi']:+.3f}]  "
              f"median {r['median_delta_A']:+.3f}  better/same/worse {r['n_improved']}/{r['n_unchanged']}/{r['n_worsened']}  "
              f"gap closed: pooled {r['gap_closed_share']:.2f}, median per protein {r['gap_closed_median_per_protein']:.2f} (n={r['n_with_gap']})" + (f"  p(vs shuffled) {fmt_p(r['p_combined_better_than_shuffled'])}" if "p_combined_better_than_shuffled" in r else ""))
    print("\nDescriptive evaluation complete. A/B/C-like counts are reported separately with threshold sensitivity; no binary success verdict is assigned.")


# ---------------------------------------------------------------------------------------------------- outputs
def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"= {p:.4f}"


def write_csv(path, rows):
    import csv
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and np.isnan(r[k]) else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r.get(k, ""))) for k in cols})


def setup(theme):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    c = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"], "text.color": c["text"],
        "axes.labelcolor": c["text2"], "xtick.color": c["text2"], "ytick.color": c["text2"], "axes.edgecolor": c["muted"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "svg.fonttype": "none", "figure.dpi": 100})
    return plt, c


def figure(results, summary, fz, a, plt, c):
    fig = plt.figure(figsize=(19.5, 6.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1.15, 1.15, 1.0], wspace=0.28, left=0.055, right=0.985, top=0.80, bottom=0.27)
    # --- A: lambda curve on the tuned splits (selection history; held-out splits are NOT drawn)
    ax = fig.add_subplot(gs[0])
    grid = fz["grid"]
    key = "mean_pct_change" if fz["select_on"] == "pct" else "mean_rmsd_change_A"
    ckey = "pct" if fz["select_on"] == "pct" else "rmsd"
    for s in TUNED:
        if s in fz["curve"]:
            ax.plot(range(len(grid)), [fz["curve"][s][str(g)][ckey] for g in grid], marker="o", color=c[s], lw=2,
                    label=f"{SPLIT_LABEL[s]} (n = {fz['curve'][s][str(grid[0])]['n']})" + ("  [selection]" if s in fz["select_splits"] else ""))
    ax.axhline(0, color=c["muted"], lw=1)
    ax.axvline(grid.index(fz["lambda"]), color=c["text"], lw=1.2, ls=(0, (4, 3)))
    ax.text(grid.index(fz["lambda"]), ax.get_ylim()[1], f" frozen λ = {fz['lambda']:g}", va="top", ha="left", fontsize=9, color=c["text"])
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([f"{g:g}" for g in grid])
    ax.set_xlabel("λ: weight of accessibility error relative to confidence\n(0 = OpenFold alone)")
    ax.set_ylabel("mean structural rank-percentile change vs OpenFold choice\n(negative = better)" if fz["select_on"] == "pct" else "mean RMSD change vs OpenFold choice (Å)")
    ax.grid(color=c["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.9), labelcolor=c["text2"])
    ax.set_title("A  Choosing λ")

    # --- B, C: per-protein RMSD change by rule, tuned vs held-out
    rules = [r for r in ("combined", "access_alone", "oracle")
             if any(np.isfinite(x.get(f"delta_{r}", np.nan)) for x in results)]
    rng = np.random.default_rng(0)
    panels = ((gs[1], [x for x in results if x["tuned"]], "tuned (development + validation)", "B  Tuned proteins"),
              (gs[2], [x for x in results if x["split"] in ("test_a", "test_b")], "held-out (Test A + Test B)", "C  Held-out proteins (frozen λ)"))
    axes = []
    for k, (spec, rs, gname, title) in enumerate(panels):
        bx = fig.add_subplot(spec, sharey=axes[0] if axes else None)
        axes.append(bx)
        bx.axhline(0, color=c["muted"], lw=1, zorder=1)
        for j, rule in enumerate(rules):
            vals = [(x[f"delta_{rule}"], x["split"]) for x in rs if np.isfinite(x.get(f"delta_{rule}", np.nan))]
            if not vals:
                continue
            v = np.array([t for t, _ in vals])
            bx.scatter(j + rng.uniform(-0.22, 0.22, len(v)), v, s=14, alpha=0.65, linewidths=0, color=[c[s] if s in c else c["muted"] for _, s in vals], zorder=2)
            row = next((r for r in summary if r["group"] == gname and r["rule"] == rule), None)
            if row:
                bx.plot([j - 0.3, j + 0.3], [row["median_delta_A"]] * 2, color=c["text"], lw=2.4, zorder=4)
        bx.set_yscale("symlog", linthresh=0.1)
        bx.set_yticks([-10, -1, -0.1, 0, 0.1, 1, 10])
        bx.set_yticklabels(["\u221210", "\u22121", "\u22120.1", "0", "0.1", "1", "10"])
        bx.set_ylim(-20, 20)
        bx.set_xticks(range(len(rules)))
        bx.set_xticklabels([RULE_LABEL[r] for r in rules], fontsize=8.5)
        bx.set_xlim(-0.6, len(rules) - 0.4)
        bx.grid(axis="y", color=c["grid"], lw=0.6)
        bx.set_axisbelow(True)
        bx.set_title(f"{title} (n = {len(rs)})")
        if k == 0:
            bx.set_ylabel("RMSD change vs OpenFold choice (Å)\nnegative = better; symmetric-log axis")
        else:
            plt.setp(bx.get_yticklabels(), visible=False)
    # Use the same thresholds as the exploratory outcome tables.
    dx = fig.add_subplot(gs[3])
    hs = [x for x in results if x["split"] in ("test_a", "test_b") and np.isfinite(x.get("achieved_gain_combined", np.nan))]
    thr = getattr(a, "gap_threshold", OUTCOME_DEFAULTS["substantial_gap_A"])
    fr = getattr(a, "recovered_fraction", OUTCOME_DEFAULTS["much_recovered_fraction"])
    bm = getattr(a, "oracle_benchmark", OUTCOME_DEFAULTS["oracle_benchmark_A"])
    numbers = [0, 0, 0, 0]
    for r in hs:
        if r["gap_A"] > thr:
            index = 0 if r["achieved_gain_combined"] / r["gap_A"] >= fr else 1
        else:
            index = 2 if r["rmsd_oracle"] >= bm else 3
        numbers[index] += 1
    labels = ["A-like\nGain recovered", "B-like\nGain missed",
              f"C-like\nOracle \u2265{bm:g} \u00c5", f"C-like\nOracle <{bm:g} \u00c5"]
    bars = dx.bar(range(4), numbers, color=[c["text2"]] * 4, width=0.65, zorder=3)
    for bar, count in zip(bars, numbers):
        percent = 100 * count / len(hs) if hs else 0
        dx.text(bar.get_x() + bar.get_width() / 2, count + max(0.4, len(hs) * 0.015),
                f"{count}\n({percent:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    dx.set_xticks(range(4))
    dx.set_xticklabels(labels, fontsize=8)
    dx.set_ylim(0, max(1, max(numbers)) * 1.25 + 1)
    from matplotlib.ticker import MaxNLocator
    dx.yaxis.set_major_locator(MaxNLocator(integer=True))
    dx.set_ylabel("Number of proteins")
    dx.set_xlabel(f"Exploratory thresholds: available gain >{thr:g} \u00c5;\nA-like recovers \u2265{fr:.0%}; C-like gain \u2264{thr:g} \u00c5", fontsize=8.5)
    dx.grid(axis="y", color=c["grid"], lw=0.6)
    dx.set_axisbelow(True)
    dx.set_title(f"D  Outcomes (held-out, n = {len(hs)})")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c[s], label=SPLIT_LABEL[s]) for s in SPLIT_LABEL if any(x["split"] == s for x in results)] +
                   [Line2D([], [], color=c["text"], lw=2.4, label="median")],
                   loc="lower center", bbox_to_anchor=(-0.1, -0.36), ncol=6, fontsize=8.5, labelcolor=c["text2"])
    prim = next((r for r in summary if r["group"] == "held-out (Test A + Test B)" and r["rule"] == "combined"), None)
    cap = ["Descriptive results: assess available and recovered gain; A/B/C-like counts are in README_outcomes.md. No binary success verdict."]
    if prim:
        line = (f"Held-out, combined vs OpenFold: mean {prim['mean_delta_A']:+.3f} Å [{prim['mean_lo']:+.3f}, {prim['mean_hi']:+.3f}], median {prim['median_delta_A']:+.3f} Å; "
                f"improved / unchanged / worsened {prim['n_improved']} / {prim['n_unchanged']} / {prim['n_worsened']}; share of the OpenFold-to-oracle gap closed {prim['gap_closed_share']:.2f} pooled, median per protein {prim['gap_closed_median_per_protein']:.2f}")
        mt = next((r for r in summary if r["group"] == MATCHED and r["rule"] == "combined"), None)
        if mt and "p_combined_better_than_shuffled" in mt:
            line += f"; shuffled accessibility null mean {mt['shuffled_null_mean_A']:+.3f} Å, p {fmt_p(mt['p_combined_better_than_shuffled'])} (n = {mt['n_proteins']})"
        cap.append(line + ".")
    cap.append("Test A/B were inspected in Stage 3 (accessibility vs structural error); no λ was tuned on them. Accessibility is computed from the experimental structure (best case, not a measured signal).")
    fig.text(0.055, 0.01, "\n".join(cap), fontsize=8.5, color=c["text2"], ha="left", va="bottom", linespacing=1.5, wrap=True)
    fig.suptitle("Does accessibility improve candidate selection beyond OpenFold confidence?", x=0.055, y=0.985, ha="left", fontsize=14, fontweight="bold", color=c["text"])
    fig.text(0.055, 0.915, f"S = z(confidence) − λ · z(accessibility error), z-scored within each protein's {int(np.median([x['n_candidates'] for x in results]))} candidates; λ frozen at {fz['lambda']:g}. "
             "Structural error (RMSD) is used only for evaluation.\nEach dot is a protein; black bars in B/C are medians. Means, confidence intervals and shuffled controls remain in rerank_summary.csv.",
             fontsize=9.5, color=c["text2"], ha="left", va="center", linespacing=1.5)
    return fig


def readme(fz, summary, n_eval, results, report, skipped, a):
    def row(group, rule):
        return next((r for r in summary if r["group"] == group and r["rule"] == rule), None)
    h = "held-out (Test A + Test B)"
    p, ac, sh, orc = row(h, "combined"), row(h, "access_alone"), row(h, "shuffled"), row(h, "oracle")
    mt = row(MATCHED, "combined")
    t = row("tuned (development + validation)", "combined")
    lines = ["# Stage 4: where is the reranking opportunity?", "",
             "**Main report:** see `README_outcomes.md` for A/B/C-like counts and sensitivity to thresholds. The statistics below are supporting evidence, not a binary pass/fail verdict.", "",
             "## What was done",
             f"Per protein, the candidates (identical for every rule) are scored by `{fz['rank_column']}` (OpenFold confidence, higher = better) and `ac_error` "
             f"(accessibility error against the reference computed from the experimental structure, lower = better). RMSD (`rmsd_ca`) is used only to evaluate. "
             f"Combined score S = z(confidence) - lambda * z(accessibility error), z-scored within each protein ({fz['scaling']} scaling, no fitted parameters). "
             f"Lambda grid {fz['grid']}; lambda was chosen on {', '.join(fz['select_splits'])} ({fz['n_selection_proteins']} proteins) by the mean change in structural rank percentile "
             f"vs the OpenFold choice (selection criterion `{fz['select_on']}`{', one-SE rule' if fz['one_se'] else ''}) and frozen at **{fz['lambda']:g}** "
             f"(frozen rule version {fz.get('version', 1)}, hash {fz['frozen_hash'][:10]}) on {fz['created_utc']} before the held-out splits were scored. "
             "The selection criterion (rank-percentile change, scale-free) differs on purpose from the declared PRIMARY OUTCOME used to evaluate "
             "(mean over proteins of RMSD of the combined choice minus RMSD of the OpenFold choice, Angstrom; negative = improvement). "
             "Tie handling: lambda = 0 uses the raw-confidence tie test, so it reproduces the OpenFold choice exactly.", ""]
    lines[4:4] = ["**Scope.** The main question is whether proteins have better candidates that reranking finds (A-like), misses (B-like), or little available gain (C-like). "
                  "Classification is exploratory and threshold-dependent; no category is forced on the whole dataset. "
                  "The per-protein table holds the continuous available gain (`gap_A`), "
                  "achieved gain (`achieved_gain_*`) and remaining gap (`remaining_gap_*`).", ""]
    if p:
        lines += ["## Held-out result (Test A + Test B, frozen lambda)",
                  f"- {p['n_proteins']} proteins. Combined choice minus OpenFold choice: mean {p['mean_delta_A']:+.3f} A [95 % CI {p['mean_lo']:+.3f}, {p['mean_hi']:+.3f}], median {p['median_delta_A']:+.3f} A (negative = improvement).",
                  f"- Improved / unchanged / worsened (|change| <= {TOL} A is unchanged): {p['n_improved']} / {p['n_unchanged']} / {p['n_worsened']}.",
                  f"- Closeness to the oracle: the available gap (OpenFold minus oracle) averages {p['available_gap_mean_A']:.3f} A; the combined rule closes {p['gap_closed_share']:.2f} of it [{p['gap_closed_lo']:.2f}, {p['gap_closed_hi']:.2f}] (pooled ratio of sums, dominated by the few proteins with a large gap); per protein, among the {p['n_with_gap']} proteins with a gap > {TOL} A, the median fraction closed is {p['gap_closed_median_per_protein']:.2f}.",
                  ]
        if ac:
            lines.append(f"- Accessibility alone: mean {ac['mean_delta_A']:+.3f} A [{ac['mean_lo']:+.3f}, {ac['mean_hi']:+.3f}], {ac['n_improved']} / {ac['n_unchanged']} / {ac['n_worsened']}.")
        if sh and mt and "p_combined_better_than_shuffled" in mt:
            lines.append(f"- Combined with shuffled accessibility (same lambda): mean {sh['mean_delta_A']:+.3f} A; null of the mean over proteins {mt['shuffled_null_mean_A']:+.3f} A "
                         f"[{mt['shuffled_null_lo']:+.3f}, {mt['shuffled_null_hi']:+.3f}]; real vs shuffled p {fmt_p(mt['p_combined_better_than_shuffled'])}. Supporting matched-protein results use the "
                         f"{mt['n_proteins']} held-out proteins that have a shuffled control, for both the interval ([{mt['mean_lo']:+.3f}, {mt['mean_hi']:+.3f}] A) and this comparison.")
        if orc:
            lines.append(f"- Oracle: mean {orc['mean_delta_A']:+.3f} A.")
        lines.append("")
    if t:
        lines += [f"Tuned splits (not independent evidence; lambda was fitted here): combined mean {t['mean_delta_A']:+.3f} A [{t['mean_lo']:+.3f}, {t['mean_hi']:+.3f}], "
                  f"{t['n_improved']} / {t['n_unchanged']} / {t['n_worsened']}.", ""]
    lines += ["## Caveats",
              "- **Test history.** Test A and Test B were inspected in Stage 3 (per-protein correlations, controls and selection plots of accessibility error vs structural error). "
              "No reranking result on them was seen and no lambda was tuned on them, but they are not blind to the accessibility signal. "
              f"The held-out splits have been evaluated {n_eval} time(s) (evaluation_log.json); do not change the rule after looking.",
              "- **Best-case accessibility.** The reference accessibility is computed from the experimental structure, not measured; a positive result here is an upper bound for real data.",
              "- **Heavy-tailed gain.** The available improvement is concentrated in a few proteins, so the mean in Angstrom can be driven by them; read the median, the counts and the share of the gap closed alongside it.",
              f"- **Confidence column.** `{fz['rank_column']}` was taken as OpenFold's confidence as stated by the user (source: {', '.join(fz['rank_source'])}); its within-protein spread is tiny, "
              "hence the within-protein z-scoring. Ties for the best score are averaged.",
              f"- **Selection set size.** Lambda was chosen on {fz['n_selection_proteins']} proteins; with few proteins the choice is noisy (the one-SE rule is on by default; pooling development + validation is the pre-declared alternative).",
              "- **Reporting change:** the original evaluation used a binary success criterion. It is no longer used to classify success; intervals and shuffle statistics remain reported. This change followed inspection of the earlier test results.",
              "- **Raw spreads.** Within a protein the spread of the confidence is ~1e-3; z-scoring equalises the two terms but also amplifies tiny confidence differences, "
              "so lambda > 0 can split a raw near-tie. Near-constant terms are flagged in the console output of each run."]
    if fz["input_policy"]["incomplete_problems_recorded"] or report["incomplete"]:
        lines.append(f"- **Incomplete inputs accepted** (--allow-incomplete): {(fz['input_policy']['incomplete_problems_recorded'] + report['incomplete'])[:6]}")
    if skipped:
        lines.append(f"- Shuffled control undefined for {len(skipped)} proteins: {', '.join(list(skipped)[:8])}.")
    lines += ["", "Files: `frozen_rule.json`, `lambda_curve.csv`, `rerank_per_protein.csv`, `rerank_summary.csv`, `rerank_figure.png|svg`, `evaluation_log.json` (older rule versions: `frozen_rule.v<k>.json`)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gap", "select", "evaluate", "outcomes"])
    ap.add_argument("--scores-csv", nargs="+", help="one or more COMBINED candidate tables holding entry, candidate (split label in its path), rank_score, "
                    "ac_error and rmsd_ca, e.g. development_validation/candidate_scores.csv test/candidate_scores.csv")
    ap.add_argument("--structure-csv", help="structure_candidate_scores.csv (rmsd_ca; may also hold rank_score) -- separate-table alternative to --scores-csv")
    ap.add_argument("--accessibility-csv", help="candidate_scores.csv (ac_error) -- separate-table alternative to --scores-csv")
    ap.add_argument("--rank-csv", help="table with entry, candidate and the confidence column, if neither table above holds it")
    ap.add_argument("--rank-column", default="rank_score", help="OpenFold confidence column, higher = better (default rank_score)")
    ap.add_argument("--scores-dir", help="score_candidates.py .npz folder (evaluate: needed for the shuffled-accessibility rule)")
    ap.add_argument("--out", default="stage4")
    ap.add_argument("--scaling", default="z", choices=["z", "rank"], help="within-protein scaling (z is the pre-declared primary)")
    ap.add_argument("--select-splits", nargs="+", default=["validation"], choices=list(TUNED))
    ap.add_argument("--select-on", default="pct", choices=["pct", "rmsd"])
    ap.add_argument("--one-se", action=argparse.BooleanOptionalAction, default=True,
                    help="smallest lambda within one standard error of the best (default on; --no-one-se for the plain minimum)")
    ap.add_argument("--expect-counts", nargs="+", default=[f"{k}={v}" for k, v in EXPECT_COUNTS.items()], metavar="SPLIT=N",
                    help="expected proteins per split (only splits being read are checked); default " + " ".join(f"{k}={v}" for k, v in EXPECT_COUNTS.items()))
    ap.add_argument("--expect-candidates", type=int, default=EXPECT_CANDIDATES, help="expected candidates per protein (default 25)")
    ap.add_argument("--allow-incomplete", action="store_true", help="drop incomplete proteins/candidates instead of stopping (recorded in the frozen rule and README)")
    ap.add_argument("--new-version", action="store_true", help="select: archive the existing frozen rule and freeze a new, disclosed version")
    ap.add_argument("--re-evaluate", action="store_true", help="evaluate: repeat an evaluation of the same frozen rule (logged)")
    ap.add_argument("--allow-code-change", action="store_true", help="evaluate: run although the code differs from the code that froze the rule (logged)")
    ap.add_argument("--gap-threshold", type=float, default=OUTCOME_DEFAULTS["substantial_gap_A"], help="outcomes: available-gain threshold in A (a convention; counts at other values are also written)")
    ap.add_argument("--recovered-fraction", type=float, default=OUTCOME_DEFAULTS["much_recovered_fraction"], help="outcomes: fraction of the available gain that counts as 'much recovered'")
    ap.add_argument("--oracle-benchmark", type=float, default=OUTCOME_DEFAULTS["oracle_benchmark_A"], help="outcomes: absolute RMSD (A) from expert guidance above which a structure is materially inaccurate for the same protein")
    ap.add_argument("--no-shuffle", action="store_true", help="evaluate: explicitly omit the supporting shuffled-accessibility comparison")
    ap.add_argument("--include-test", action="store_true", help="gap only: include the held-out splits")
    ap.add_argument("--min-candidates", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--n-perm", type=int, default=500, help="shuffle draws per protein")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theme", default="light", choices=list(THEMES))
    a = ap.parse_args()
    if a.command == "outcomes":
        cmd_outcomes(a)
        return
    if a.scores_csv and (a.structure_csv or a.accessibility_csv):
        ap.error("use either --scores-csv or --structure-csv/--accessibility-csv, not both")
    if not a.scores_csv and not (a.structure_csv and a.accessibility_csv):
        ap.error("give --scores-csv, or both --structure-csv and --accessibility-csv")
    try:
        a.expect_counts = {k: int(v) for k, v in (x.split("=") for x in a.expect_counts)}
    except ValueError:
        ap.error("--expect-counts takes SPLIT=N pairs, e.g. development=30 validation=15 test_a=20 test_b=20")
    if set(a.expect_counts) - set(SPLIT_LABEL):
        ap.error(f"--expect-counts splits must be among {sorted(SPLIT_LABEL)}")
    {"gap": cmd_gap, "select": cmd_select, "evaluate": cmd_evaluate}[a.command](a)


if __name__ == "__main__":
    main()
