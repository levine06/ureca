#!/usr/bin/env python
"""Stage 3, step 3: is accessibility a data type that informs structure prediction? Controls and the top-1 selection test.

Reads the per-entry .npz files written by score_candidates.py. If those files hold only accessibility errors (the current
output), pass the structural scores with --structure-csv structure_candidate_scores.csv; it is joined to each protein by
entry + candidate id (never by row position), and join_report.csv records matched / unmatched / dropped counts per protein.
The question is NOT whether accessibility error and structural
error correlate (the plot script shows that); it is whether the correlation, and the usefulness of choosing by accessibility,
come from the position-specific information in the reference accessibility profile. Every control therefore changes only
the REFERENCE side (or the set of residues used); the candidates and their structural errors never change.

Two outcomes are computed for every protein (the independent unit) and every control:
  rho     Spearman correlation between the accessibility error e_k and the structural error s_k over the K candidates
          (s = C-alpha RMSD primary; 1 - lDDT and 1 - TM-score as agreement checks).  rho > 0: lower accessibility error goes
          with lower structural error.
  pick    the outcome of CHOOSING the candidate with the lowest accessibility error: pick_pct = its rank percentile among the
          protein's candidates by structural error (0 = best, 1 = worst; a random pick has 0.5 whatever the protein) and
          pick_delta = its structural error minus the protein's mean (Angstrom for RMSD; a random pick has 0).
          Both are read against the FIXED structural ranking of ALL candidates; candidates whose control error is undefined are simply
          not eligible to be picked (min_eligible_candidates in controls_per_entry.csv), they are not removed from the ranking.
Controls (accessfold.controls documents each): real, shuffle (residue values permuted), reverse, other_protein, dropout:<f>,
binary:<threshold|median>.  Reference selection rules: random, oracle (the best candidate) and, with --rank-csv, ranking_top1
(the candidate ranked first by the supplied file and column; check what that column really is, e.g. OpenFold3's
sample_ranking_score, which is not the same as mean pLDDT).  Decision logic: `real` must beat the shuffle null; if accessibility only reflects "candidates that deviate more
differ more", a shuffled profile will do about as well as the real one. Compare `real` with the SHUFFLE null, not with the
random rule: the shuffle null of the selection test need not sit at 0.5, because choosing the lowest of K noisy errors tends
to pick the candidate whose error varies most (often a worse one), so the null can lie away from 0.5 (as in the synthetic tests).

    python scripts/run_controls.py --scores-dir scores_out --structure-csv structure_candidate_scores.csv \\
        --accessibility-csv candidate_scores.csv --rank-csv candidate_scores.csv --out controls_out
(--accessibility-csv makes `real` use your existing ac_error / ac_corr_error; it is checked against the recomputation the controls use
and the run STOPS if they disagree, so real and controls always come from one calculation. Plot: scripts/plot_controls.py, which
reads this output folder and draws real against the full shuffle distribution; plot_accessibility_vs_accuracy.py is for the real relationship.)

Summaries are over ALL proteins (any protein with >= --min-candidates candidates; --min-rmsd-spread 0 by default). The
subset whose RMSD range (highest minus lowest RMSD to the experimental structure; NOT the diversity of the candidates among
themselves) is >= --subset-rmsd-range (1 A) is reported separately as `set = subset`. Matched proteins: within a (set, error
metric, structural metric) the real value, every other control, the bootstrap interval and the shuffle null are computed over
the SAME proteins (defined real value AND a defined shuffle null); n_dropped_total counts those that fell out (split into n_dropped_real_or_shuffle_undefined and n_dropped_control_undefined).
Use a dedicated scores folder, or restrict the read with --csv <stage 1 CSVs>.

Outputs in --out: controls_per_entry.csv, controls_summary.csv (rho), selection_summary.csv (top-1 selection), shuffle_null.npz
(keys "<error_metric>|<entry>|<metric>|rho|pct|delta"), settings.json.
Two accessibility errors are analysed side by side: "mae" (mean absolute difference of relative accessibility) and "corr"
(1 - Pearson correlation of the profiles). The shuffle null of "mae" is NOT centred on 0 (overall exposure differences between
good and bad candidates survive a shuffle). For "corr" the expected error of each candidate under a shuffle is exactly 1, which
removes that offset, but the across-candidate Spearman and the top-1 pick of the generated null are still not guaranteed to be
centred on 0 / 0.5, so "corr" is the cleaner test of position-specific information, not a guaranteed-zero one. Always judge
`real` against the generated shuffle null (shuffle_null.npz), never against 0 or 0.5. binary:<t> with "mae" is the fraction of
mismatched exposed/buried labels; with "corr" it is 1 - Pearson of the 0/1 profiles (undefined, so the protein is dropped, if a
profile is constant).
"""
import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np

from accessfold.controls import (STRUCT_METRICS, baseline_selection, bootstrap_mean, bootstrap_median, control_errors, load_entry,
                                 match_ids, protein_level_null, read_table, rho_for, run_selection)

DEFAULT_CONTROLS = ["real", "shuffle", "reverse", "other_protein", "dropout:0.5", "dropout:0.2", "binary:0.2",
                    "binary:median"]
BASELINES = ["random", "oracle", "ranking_top1"]


def rng_for(seed, *names):
    return np.random.default_rng([seed] + [zlib.crc32(n.encode()) for n in names])


def raise_csv_field_limit():
    limit = 2 ** 31 - 1
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def peek_name(path):
    """Entry name stored in a score .npz (reads only the small meta record)."""
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))["entry"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", default="controls_out")
    ap.add_argument("--controls", nargs="+", default=DEFAULT_CONTROLS)
    ap.add_argument("--n-perm", type=int, default=1000, help="repetitions for shuffle / dropout")
    ap.add_argument("--n-other", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--min-candidates", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="drop candidates with lower coverage (see score_candidates.py --min-coverage) before any analysis")
    ap.add_argument("--min-rmsd-spread", type=float, default=0.0,
                    help="hard exclusion of proteins whose RMSD range is below this (A); default 0 = nothing excluded")
    ap.add_argument("--subset-rmsd-range", type=float, default=1.0,
                    help="supplementary subset: proteins whose RMSD range is at least this (A); reported as set=subset")
    ap.add_argument("--metrics", nargs="+", default=list(STRUCT_METRICS), help="structural-error metrics")
    ap.add_argument("--error-metrics", nargs="+", default=["mae", "corr"], choices=["mae", "corr"],
                    help="accessibility error: mae (mean absolute difference) and/or corr (1 - profile correlation)")
    ap.add_argument("--structure-csv", help="structural-score table (entry, candidate, rmsd_ca, lddt_ca, tm_score[, coverage_ca, coverage_mask]), e.g. "
                                            "structure_candidate_scores.csv. REQUIRED when the .npz files hold accessibility only. Joined to the .npz "
                                            "candidates by entry + candidate id (never by order); if both have structural values they must agree")
    ap.add_argument("--strict-join", action="store_true", help="exit 1 if any .npz candidate has no row in --structure-csv")
    ap.add_argument("--accessibility-csv", help="candidate_scores.csv from score_candidates.py. If given, the `real` control uses its ac_error / "
                    "ac_corr_error values (joined by entry + candidate id) instead of recomputing them; the recomputation is still run once as a consistency check")
    ap.add_argument("--rank-csv", help="table with entry, candidate and a ranking column for the ranking_top1 reference rule. The script does not "
                                       "assume where the ranking comes from (the model's ranking score, mean pLDDT, ...): the rule is labelled with "
                                       "the file and column you give")
    ap.add_argument("--rank-column", default="rank", help="column of --rank-csv holding the ranking (default: rank, 1 = best)")
    ap.add_argument("--rank-higher-better", action="store_true", help="the --rank-column is a score where larger = better (not a rank)")
    ap.add_argument("--csv", nargs="+", help="optional Stage 1 CSV(s); only entries listed there are analysed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(Path(a.scores_dir).glob("*.npz")) if not f.name.startswith(".")]
    if a.csv:
        raise_csv_field_limit()
        wanted = set()
        for pth in a.csv:
            with open(pth, newline="") as fh:
                wanted |= {f"{r['pdb_id'].strip().upper()}_{r['label_chain_id'].strip()}" for r in csv.DictReader(fh)}
        ignored = sorted(peek_name(f) for f in files if peek_name(f).upper() not in wanted)
        files = [f for f in files if peek_name(f).upper() in wanted]
        if ignored:
            print(f"ignoring {len(ignored)} entries in {a.scores_dir} that are not in --csv: {ignored[:10]}")
    structure = read_table(a.structure_csv, ("rmsd_ca", "lddt_ca", "tm_score", "coverage_ca", "coverage_mask")) if a.structure_csv else None
    entries, load_errors = [], {}
    for f in files:
        try:
            entries.append(load_entry(f, a.min_coverage, structure))
        except ValueError as exc:
            load_errors[f.name] = str(exc)
    if load_errors:
        for k, v in list(load_errors.items())[:5]:
            print(f"cannot load {k}: {v}")
    if a.strict_join and load_errors:
        for k, v in load_errors.items():
            print(f"--strict-join: cannot load {k}: {v}")
        raise SystemExit(f"--strict-join: {len(load_errors)} of {len(files)} score files could not be loaded; stopping")
    if load_errors:
        print(f"NOTE: {len(load_errors)} of {len(files)} score files could not be loaded and are NOT in the analysis (use --strict-join to stop instead)")
    if not entries:
        raise SystemExit(f"no usable .npz files in {a.scores_dir}" + (" (see the messages above)" if load_errors else ""))
    # how the structural table was joined, per protein (a table row never silently pairs with the wrong candidate)
    join_rows = [{"entry": e.name, **{k: e.join_report.get(k) for k in ("structure_source", "n_npz_candidates", "exact", "normalised", "basename",
                                                                        "unmatched", "extra_in_table", "dropped_nonfinite", "dropped_coverage",
                                                                        "coverage_checked", "n_kept")}} for e in entries]
    n_un = sum(r["unmatched"] for r in join_rows)
    n_fb = sum((r["normalised"] or 0) + (r["basename"] or 0) for r in join_rows)
    if structure is not None:
        print(f"structure table join: {sum(r['n_kept'] for r in join_rows)} candidates kept of {sum(r['n_npz_candidates'] for r in join_rows)}; "
              f"{n_un} without a table row; {sum(r['extra_in_table'] for r in join_rows)} table rows without a score-file candidate; "
              f"{n_fb} matched by normalised id or base name (the rest by exact id)")
    if a.strict_join and n_un:
        print(f"--strict-join: {n_un} .npz candidates have no row in {a.structure_csv} (see join_report.csv); stopping")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "join_report.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(join_rows[0]))
            w.writeheader()
            w.writerows(join_rows)
        raise SystemExit(1)
    nocov = [r["entry"] for r in join_rows if not r["coverage_checked"]]
    if nocov:
        print(f"WARNING: coverage could not be checked for {len(nocov)} entries (no coverage arrays or columns): {nocov[:5]}")
    ac_table = read_table(a.accessibility_csv, ("ac_error", "ac_corr_error")) if a.accessibility_csv else None
    ac_col = {"mae": "ac_error", "corr": "ac_corr_error"}
    ranks = {}
    if a.rank_csv:
        tab = read_table(a.rank_csv, (a.rank_column,))
        ranks = {e: {cid: row[a.rank_column] for cid, row in d.items() if a.rank_column in row} for e, d in tab.items()}
        if not any(ranks.values()):
            raise SystemExit(f"{a.rank_csv}: no column {a.rank_column!r} with values")

    included, reason = [], {}
    for e in entries:
        k = len(e.rmsd_ca)
        if k < a.min_candidates:
            reason[e.name] = f"only {k} candidates"
        elif not (e.rmsd_spread >= a.min_rmsd_spread):
            reason[e.name] = f"RMSD range {e.rmsd_spread:.2f} A < {a.min_rmsd_spread}"
        else:
            included.append(e)
    inc = {e.name for e in included}
    subset = {e.name for e in included if e.rmsd_spread >= a.subset_rmsd_range}
    sets = {"all": inc, "subset": subset}
    print(f"{len(included)} of {len(entries)} proteins analysed" + (f"; excluded: {reason}" if reason else "")
          + f"; supplementary subset (RMSD range >= {a.subset_rmsd_range:g} A): {len(subset)}")
    n_rank = sum(1 for e in included if "ranking_top1" in baseline_selection(e, ranks.get(e.name), a.metrics, a.rank_higher_better))
    if a.rank_csv:
        print(f"ranking_top1 reference ({Path(a.rank_csv).name}, column {a.rank_column}): matched for every candidate of {n_rank} of {len(included)} proteins")

    per_entry, summary, selection, null_store = [], [], [], {}
    rng = np.random.default_rng(a.seed)
    for em in a.error_metrics:
        ac_mismatch = []
        rho, pct, delta = {}, {}, {}                            # (control, metric) -> {entry: mean value over repetitions}
        null = {k: {m: {} for m in a.metrics} for k in ("rho", "pct", "delta")}
        rows_em = []
        for e in entries:
            if e.name in reason:
                continue
            for control in a.controls:
                try:
                    errs = control_errors(e, control, rng_for(a.seed, e.name, control, em), n_rep=a.n_perm, others=included,
                                          n_other=a.n_other, error_metric=em)
                    if control == "real" and ac_table is not None:
                        rows_t, _, _ = match_ids(list(e.candidates), ac_table.get(e.name, {}))
                        col = ac_col[em]
                        if any(r is None or col not in r for r in rows_t):
                            raise ValueError(f"{sum(r is None or col not in r for r in rows_t)} candidates have no {col} in {Path(a.accessibility_csv).name}")
                        given = np.array([r[col] for r in rows_t], float)
                        both = np.isfinite(given) & np.isfinite(errs[0])
                        if (np.isfinite(given) != np.isfinite(errs[0])).any() or not np.allclose(given[both], errs[0][both], rtol=1e-4, atol=1e-6):
                            ac_mismatch.append((e.name, float(np.abs(given[both] - errs[0][both]).max()) if both.any() else float("nan"),
                                                int((np.isfinite(given) != np.isfinite(errs[0])).sum())))
                        errs = [given]
                except ValueError as exc:
                    print(f"skip {e.name} {control}: {exc}")
                    continue
                res = {m: np.array([rho_for(e, x, (m,))[m] for x in errs]) for m in a.metrics}
                sel = run_selection(e, control, None, metrics=a.metrics, errors=errs)
                row = {"error_metric": em, "entry": e.name, "split": e.split, "control": control, "in_subset": e.name in subset,
                       "n_candidates": len(e.rmsd_ca), "rmsd_range_A": round(e.rmsd_spread, 3), "n_reps": len(errs),
                       "min_eligible_candidates": int(min(np.isfinite(x).sum() for x in errs)) if errs else 0}
                for m in a.metrics:
                    v, pc, dl = res[m], sel[m]["pct"], sel[m]["delta"]
                    row[f"rho_{m}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                    row[f"rho_sd_{m}"] = float(np.nanstd(v)) if v.size > 1 else np.nan
                    row[f"pick_pct_{m}"] = float(np.nanmean(pc)) if np.isfinite(pc).any() else np.nan
                    row[f"pick_delta_{m}"] = float(np.nanmean(dl)) if np.isfinite(dl).any() else np.nan
                    rho.setdefault((control, m), {})[e.name] = row[f"rho_{m}"]
                    pct.setdefault((control, m), {})[e.name] = row[f"pick_pct_{m}"]
                    delta.setdefault((control, m), {})[e.name] = row[f"pick_delta_{m}"]
                    if control == "shuffle":
                        for kind, arr in (("rho", v), ("pct", pc), ("delta", dl)):
                            null[kind][m][e.name] = arr
                            null_store[f"{em}|{e.name}|{m}|{kind}"] = arr
                rows_em.append(row)
            base = baseline_selection(e, ranks.get(e.name), a.metrics, a.rank_higher_better)
            for bname, per_m in base.items():
                row = {"error_metric": em, "entry": e.name, "split": e.split, "control": bname, "in_subset": e.name in subset,
                       "n_candidates": len(e.rmsd_ca), "rmsd_range_A": round(e.rmsd_spread, 3), "n_reps": 1}
                for m in a.metrics:
                    row[f"pick_pct_{m}"], row[f"pick_delta_{m}"] = per_m[m]["pct"], per_m[m]["delta"]
                    pct.setdefault((bname, m), {})[e.name] = per_m[m]["pct"]
                    delta.setdefault((bname, m), {})[e.name] = per_m[m]["delta"]
                rows_em.append(row)

        if ac_mismatch:                       # `real` and the controls must come from ONE calculation: never mix them
            for name_, d_, n_ in ac_mismatch[:10]:
                print(f"{name_}: {ac_col[em]} in {Path(a.accessibility_csv).name} differs from the value recomputed from the profiles "
                      f"(max abs difference {d_:.3g}; {n_} candidates not comparable)")
            raise SystemExit(f"STOP: {len(ac_mismatch)} proteins have accessibility errors in {a.accessibility_csv} that do not match the recomputation "
                             "the controls use (different radii / min residues / masks / candidate set?). Rerun without --accessibility-csv "
                             "(real is then recomputed consistently) or regenerate the CSV with the same settings.")
        # one-sided permutation p per protein: the real value vs the protein's own shuffle null (descriptive: the K candidates
        # are not independent, so these are not corrected or relied on; the protein-level null below is the test)
        real_rho = {(r["entry"], m): r[f"rho_{m}"] for r in rows_em if r["control"] == "real" for m in a.metrics}
        real_pct = {(r["entry"], m): r[f"pick_pct_{m}"] for r in rows_em if r["control"] == "real" for m in a.metrics}
        for row in rows_em:
            if row["control"] == "shuffle":
                for m in a.metrics:
                    nul = null["rho"][m].get(row["entry"], np.array([]))
                    nul, rr = nul[np.isfinite(nul)], real_rho.get((row["entry"], m), np.nan)
                    row[f"p_perm_{m}"] = (1 + (nul >= rr).sum()) / (1 + nul.size) if nul.size and np.isfinite(rr) else np.nan
                    nul = null["pct"][m].get(row["entry"], np.array([]))
                    nul, rr = nul[np.isfinite(nul)], real_pct.get((row["entry"], m), np.nan)
                    row[f"p_perm_pick_{m}"] = (1 + (nul <= rr).sum()) / (1 + nul.size) if nul.size and np.isfinite(rr) else np.nan
        per_entry += rows_em

        for set_name, members in sets.items():
            for m in a.metrics:
                # Matching. `base` = proteins with a defined real value AND (if a shuffle was run) a defined shuffle null; the
                # real/shuffle comparison and its p-value use exactly these. Every other control is then summarised over
                # `base` restricted to the proteins where THAT control is defined, and its row restates the real value over the
                # same proteins, so no row compares different protein sets without saying so. The three exclusion counts add up.
                def exclusions(base, matched):
                    return {"n_dropped_total": len(members) - len(matched),
                            "n_dropped_real_or_shuffle_undefined": len(members) - len(base),
                            "n_dropped_control_undefined": len(base) - len(matched)}

                # ---- rho
                real_d = rho.get(("real", m), {})
                base = {k for k in members if np.isfinite(real_d.get(k, np.nan))}
                if "shuffle" in a.controls:
                    base = {k for k in base if np.isfinite(null["rho"][m].get(k, np.array([]))).any()}
                for control in a.controls:
                    d = rho.get((control, m), {})
                    matched = {k for k in base if np.isfinite(d.get(k, np.nan))}
                    vals = np.array([d[k] for k in sorted(matched)])
                    med, lo, hi = bootstrap_median(vals, a.n_boot, rng)
                    real_same = np.array([real_d[k] for k in sorted(matched)])
                    row = {"set": set_name, "error_metric": em, "control": control, "metric": m, "n_proteins": vals.size,
                           **exclusions(base, matched), "median_rho": med, "ci95_lo": lo, "ci95_hi": hi,
                           "mean_rho": float(vals.mean()) if vals.size else np.nan,
                           "frac_rho_positive": float((vals > 0).mean()) if vals.size else np.nan,
                           "real_median_rho_same_proteins": float(np.median(real_same)) if real_same.size else np.nan}
                    if control == "real" and vals.size and "shuffle" in a.controls:
                        nm = protein_level_null({k: v for k, v in null["rho"][m].items() if k in matched}, rng, 10000)
                        if nm.size:
                            row["shuffle_null_median_mean"] = float(nm.mean())
                            row["shuffle_null_median_q025"] = float(np.quantile(nm, 0.025))
                            row["shuffle_null_median_q95"] = float(np.quantile(nm, 0.95))
                            row["shuffle_null_median_q975"] = float(np.quantile(nm, 0.975))
                            row["p_median_vs_shuffle"] = float((1 + (nm >= med).sum()) / (1 + nm.size))
                            sh = rho.get(("shuffle", m), {})
                            diffs = np.array([d[k] - sh[k] for k in matched if k in sh and np.isfinite(sh[k])])
                            row["median_real_minus_shuffle_mean"] = float(np.median(diffs)) if diffs.size else np.nan
                    summary.append(row)

                # ---- top-1 selection (same matching, on the pick percentile)
                real_p = pct.get(("real", m), {})
                base_s = {k for k in members if np.isfinite(real_p.get(k, np.nan))}
                if "shuffle" in a.controls:
                    base_s = {k for k in base_s if np.isfinite(null["pct"][m].get(k, np.array([]))).any()}
                rk = pct.get(("ranking_top1", m), {})
                for control in a.controls + BASELINES:
                    d, dd = pct.get((control, m), {}), delta.get((control, m), {})
                    matched = {k for k in base_s if np.isfinite(d.get(k, np.nan))}
                    keys = sorted(matched)
                    vals = np.array([d[k] for k in keys])
                    dv = np.array([dd[k] for k in keys])
                    mean, lo, hi = bootstrap_mean(vals, a.n_boot, rng)
                    dmed, dlo, dhi = bootstrap_median(dv, a.n_boot, rng)
                    real_same = np.array([real_p[k] for k in keys])
                    row = {"set": set_name, "error_metric": em, "control": control, "metric": m, "n_proteins": vals.size,
                           **exclusions(base_s, matched), "mean_pick_pct": mean, "ci95_lo": lo, "ci95_hi": hi,
                           "median_pick_pct": float(np.median(vals)) if vals.size else np.nan,
                           "frac_better_than_random": float((vals < 0.5).mean()) if vals.size else np.nan,
                           "median_pick_delta": dmed, "delta_ci95_lo": dlo, "delta_ci95_hi": dhi,
                           "real_mean_pick_pct_same_proteins": float(real_same.mean()) if real_same.size else np.nan}
                    if control == "real" and vals.size:
                        if "shuffle" in a.controls:
                            nm = protein_level_null({k: v for k, v in null["pct"][m].items() if k in matched}, rng, 10000, statistic="mean")
                            if nm.size:
                                row["shuffle_null_mean_pct_mean"] = float(nm.mean())
                                row["shuffle_null_mean_pct_q025"] = float(np.quantile(nm, 0.025))
                                row["shuffle_null_mean_pct_q05"] = float(np.quantile(nm, 0.05))
                                row["shuffle_null_mean_pct_q975"] = float(np.quantile(nm, 0.975))
                                row["p_mean_pct_vs_shuffle"] = float((1 + (nm <= mean).sum()) / (1 + nm.size))
                        both = [k for k in keys if k in rk and np.isfinite(rk[k])]
                        if both:
                            diff = np.array([d[k] - rk[k] for k in both])
                            row["n_proteins_vs_ranking"] = len(both)
                            row["real_minus_ranking_mean_pct"] = float(diff.mean())
                            row["frac_real_better_than_ranking"] = float(((diff < 0).sum() + 0.5 * (diff == 0).sum()) / len(diff))
                    selection.append(row)

    def write(path, rows):
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if (isinstance(r.get(c), float) and np.isnan(r[c])) else r.get(c, "")) for c in cols})

    write(out / "controls_per_entry.csv", per_entry)
    write(out / "controls_summary.csv", summary)
    write(out / "selection_summary.csv", selection)
    write(out / "join_report.csv", join_rows)
    np.savez_compressed(out / "shuffle_null.npz", **null_store)
    (out / "settings.json").write_text(json.dumps({**vars(a), "n_analysed": len(included), "n_subset": len(subset), "excluded": reason,
                                                      "load_errors": load_errors, "candidates_without_structure_row": int(n_un)}, indent=1))

    print(f"\n[rho] {'set':<7}{'err':<5}{'control':<16}{'metric':<9}{'n':>3} {'median rho':>10}   95% CI (bootstrap over proteins)  frac>0")
    for r in summary:
        if r["control"] in ("real", "shuffle", "reverse", "other_protein"):
            print(f"      {r['set']:<7}{r['error_metric']:<5}{r['control']:<16}{r['metric']:<9}{r['n_proteins']:>3} {r['median_rho']:>10.3f}   "
                  f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]{'':<10}{r['frac_rho_positive']:.2f}"
                  + (f"   p(vs shuffle null)={r['p_median_vs_shuffle']:.4f}" if "p_median_vs_shuffle" in r else ""))
    print(f"\n[top-1 selection] pick_pct: 0 = picked the most accurate candidate, 0.5 = random, lower is better")
    print(f"      {'set':<7}{'err':<5}{'rule':<16}{'metric':<9}{'n':>3} {'mean pct':>9}   95% CI            median d(A)")
    for r in selection:
        if r["metric"] == "rmsd_ca" or r["control"] == "real":
            print(f"      {r['set']:<7}{r['error_metric']:<5}{r['control']:<16}{r['metric']:<9}{r['n_proteins']:>3} {r['mean_pick_pct']:>9.3f}   "
                  f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]   {r['median_pick_delta']:+.3f}"
                  + (f"   p(vs shuffle)={r['p_mean_pct_vs_shuffle']:.4f}" if "p_mean_pct_vs_shuffle" in r else ""))
    print(f"\nwrote {out}/controls_per_entry.csv, controls_summary.csv, selection_summary.csv")


if __name__ == "__main__":
    main()
