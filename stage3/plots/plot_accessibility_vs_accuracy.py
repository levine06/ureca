#!/usr/bin/env python
"""Stage 3 figure: does low accessibility error (against the experimental-structure reference) go with structural accuracy?

For every candidate structure
    x = accessibility error against the experimental-structure reference   (candidate_scores.csv: ac_error = mean-over-radii absolute error;
                                                            ac_corr_error = 1 - profile correlation)
    y = actual structural error against the experimental structure  (structure_candidate_scores.csv: rmsd_ca primary;
                                                            1 - lddt_ca or 1 - tm_score as agreement checks)
A useful accessibility measurement gives a rising relationship: candidates that agree better with the reference accessibility
(small x) are structurally more accurate (small y). NOTE: the reference accessibility here is COMPUTED from the experimental
structure (compute_truth_accessibility.py); it is not a measured experimental signal, so this is the best-case input. Both axes are ERRORS (lower = better), so "works" = positive slope.

Why everything is within-protein. Proteins differ enormously in size and in how hard they are (here Ca RMSD
0.3 .. 19 A, accessibility error 0.013 .. 0.10), so a raw pooled scatter would show a strong "relationship" that is only
"big/hard proteins are bad on both axes" and says nothing about whether accessibility ranks candidates of the SAME protein.
The protein is therefore the unit throughout, and candidates are only ever compared with their own protein's candidates.

Figures written to --out (PNG + SVG):
  fig1_accessibility_vs_accuracy   A: each protein's candidates are split into 5 equal groups by their within-protein rank of
                                      accessibility error (group 1 = the 20 % of candidates closest to the reference, group 5 =
                                      furthest; tie rule: the AVERAGE rank of tied candidates decides the group). For every protein
                                      and group the mean within-protein rank percentile of the structural error is computed
                                      (0 = most accurate candidate of that protein, 1 = least; 0.5 = expected average rank under unrelated rankings); the boxes
                                      are over proteins (one value per protein and group; line = median, box = quartiles,
                                      whiskers = 1.5 x quartile range, dots = proteins). A useful signal rises from group 1 to 5.
                                   B: per-protein Spearman rho, one row per dataset split (dots = proteins, bar = median with
                                      95 % bootstrap interval over proteins).
  fig2_per_protein_raw_p<k>        OPTIONAL (--raw-pages): small panels with RAW values and their own axes.
  plotted_points.csv, per_protein_stats.csv, bin_summary.csv   the numbers behind the figures
  dropped_rows.csv, plot_settings.json   rows that were NOT plotted (missing values) and the inputs/settings/matched counts
Ranks do not carry magnitude, so the size of the effect is also reported in raw units: per protein, the mean structural error of
group 5 minus group 1 (A for RMSD), summarised as the median over proteins with a bootstrap interval.
Main summary = ALL proteins with a defined rho (>= 4 candidates, neither axis constant): median of the per-protein rho with
a percentile bootstrap over proteins (the independent unit), overall and per split, plus the count with rho > 0.
Optional supplementary summary (disabled by default) = the subset whose "RMSD range" is >= --subset-rmsd-range, shown as ringed dots in B, a
dashed line in A and a second block in the summary box. RMSD range = highest minus lowest RMSD-to-the-experimental-structure
among the protein's candidates. It limits how much selection could gain, but it is NOT the diversity of the candidates
among themselves (that needs pairwise candidate RMSD, the `pair_rmsd` array of the score_candidates.py .npz files), and a
small range does not make a correlation invalid. The 1 A cut-off is a convention of mine, not an established threshold.
No p-value is drawn here: the shuffle/other-protein controls (run_controls.py) provide the null.

    python scripts/plot_accessibility_vs_accuracy.py --scores-dir scores_out --out figures
    python scripts/plot_accessibility_vs_accuracy.py --accessibility-csv candidate_scores.csv \\
        --structure-csv structure_candidate_scores.csv --y rmsd_ca --x ac_corr_error --out figures
Needs numpy, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

# Reference palette of the data-viz skill (categorical slots 1-2 validated for all pairs, both modes; contrast >= 3:1).
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e1",
                  dev="#2a78d6", val="#eb6834", test_a="#16865c", test_b="#a345a0", excl="#b5b4ae"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8c8b84", grid="#2e2e2c",
                 dev="#3987e5", val="#d95926", test_a="#51c993", test_b="#d78bd0", excl="#5a5955"),
}
SPLIT_COLOURS = {"development": "dev", "validation": "val", "test_a": "test_a", "test_b": "test_b"}
SPLIT_LABELS = {"development": "Development", "validation": "Validation", "test_a": "Test A", "test_b": "Test B"}
Y_CHOICES = {"rmsd_ca": ("Cα RMSD to the experimental structure (Å)", lambda v: v),
             "lddt_ca": ("1 − lDDT-Cα (observed)", lambda v: 1.0 - v),
             "tm_score": ("1 − TM-score", lambda v: 1.0 - v)}
X_LABELS = {"ac_error": "Accessibility error against the experimental-structure reference\n(mean absolute error of relative accessibility, mean over probe radii)",
            "ac_corr_error": "Accessibility error against the experimental-structure reference\n(1 − correlation of the accessibility profiles)",
            "ac_rmse": "Accessibility error against the experimental-structure reference\n(RMS error of relative accessibility)"}
YSHORT = {"rmsd_ca": "C\u03b1 RMSD", "lddt_ca": "1 \u2212 lDDT-C\u03b1", "tm_score": "1 \u2212 TM-score"}
SHORT_X = {"ac_error": "accessibility error (MAE)", "ac_corr_error": "accessibility error (1 − r)", "ac_rmse": "accessibility error (RMSE)"}


# ------------------------------------------------------------------------------------------------------- data
def input_paths(paths):
    return [paths] if isinstance(paths, (str, Path)) else list(paths)


def read_rows(paths):
    rows = []
    for path in input_paths(paths):
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def candidate_key(row):
    candidate = row["candidate"].strip().replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    return row["entry"].strip(), candidate


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(acc_csv, struct_csv, xcol, ycol, strict=False):
    """Join the two scoring tables on (entry, candidate). Returns (rows, report). Rows with a missing / non-finite x or y are NOT
    kept, and they are reported (report["dropped_nonfinite"], printed by main and written to dropped_rows.csv), never silent."""
    acc, st = read_rows(acc_csv), read_rows(struct_csv)
    for name, rows, col in (("accessibility", acc, xcol), ("structure", st, ycol)):
        if not rows or any(col not in row or "entry" not in row or "candidate" not in row for row in rows):
            sys.exit(f"{name} table has no column {col!r}; columns: {list(rows[0])[:12] if rows else 'none (empty table)'} ...")
    key = candidate_key
    sidx = {key(r): r for r in st}
    if len(sidx) != len(st) or len({key(r) for r in acc}) != len(acc):
        sys.exit("(entry, candidate) is not unique in one of the tables; the two tables must come from the same score_candidates.py run")
    out, only_acc, dropped = [], [], []
    transform = Y_CHOICES[ycol][1]
    for r in acc:
        s = sidx.get(key(r))
        if s is None:
            only_acc.append(key(r))
            continue
        x, y = num(r[xcol]), transform(num(s[ycol]))
        if np.isfinite(x) and np.isfinite(y):
            path_splits = {p for p in key(r)[1].split("/") if p in SPLIT_COLOURS}
            if len(path_splits) > 1:
                sys.exit(f"ambiguous dataset labels in candidate {r['candidate']!r}")
            split = next(iter(path_splits), r.get("split", "").strip())
            out.append({"entry": r["entry"], "candidate": r["candidate"], "split": split, "x": x, "y": y,
                        "rmsd_ca": num(s["rmsd_ca"]) if "rmsd_ca" in s else float("nan")})
        else:
            dropped.append({"entry": r["entry"], "candidate": r["candidate"], "reason": f"x ({xcol}) = {r[xcol]!r}" if not np.isfinite(x) else f"y ({ycol}) = {s[ycol]!r}"})
    acc_keys = {key(r) for r in acc}
    only_struct = [k for k in sidx if k not in acc_keys]
    report = {"rows_accessibility_table": len(acc), "rows_structure_table": len(st), "matched": len(acc) - len(only_acc),
              "kept": len(out), "only_in_accessibility_table": len(only_acc), "only_in_structure_table": len(only_struct),
              "dropped_nonfinite": dropped}
    problems = []
    if only_acc or only_struct:
        problems.append(f"{len(only_acc)} candidates only in the accessibility table and {len(only_struct)} only in the structure table "
                        "(were the two tables produced by the same run?)")
    if dropped:
        problems.append(f"{len(dropped)} matched candidates have a missing/non-finite {xcol} or {ycol} and are NOT plotted "
                        f"(first: {dropped[0]['entry']} {dropped[0]['candidate']}, {dropped[0]['reason']}); see dropped_rows.csv")
    for msg in problems:
        print("WARNING: " + msg, file=sys.stderr)
    if strict and problems:
        sys.exit("--strict: stopping because of the problems above")
    if not out:
        sys.exit("no candidates with finite x and y")
    return out, report


def rank_pct(v):
    """Within-protein rank percentile in [0, 1]: 0 = lowest error, 1 = highest (average ranks for ties)."""
    r = rankdata(v)
    return (r - 1.0) / (len(v) - 1.0) if len(v) > 1 else np.full(len(v), np.nan)


def spearman(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x) - rankdata(x).mean(), rankdata(y) - rankdata(y).mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def bootstrap_median(v, n_boot=10000, seed=0, alpha=0.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    meds = np.median(v[np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))], axis=1)
    return float(np.median(v)), float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def group(rows, subset_min):
    """Per-protein statistics and the within-protein rank coordinates of every candidate."""
    by = OrderedDict()
    for r in rows:
        by.setdefault(r["entry"], []).append(r)
    stats = []
    for name, rs in by.items():
        x, y = np.array([r["x"] for r in rs]), np.array([r["y"] for r in rs])
        rm = np.array([r["rmsd_ca"] for r in rs])
        spread = float(np.nanmax(rm) - np.nanmin(rm)) if np.isfinite(rm).any() else float("nan")
        px, py = rank_pct(x), rank_pct(y)
        for r, a, b in zip(rs, px, py):
            r["x_rank"], r["y_rank"] = float(a), float(b)
        split = rs[0]["split"]
        if any(r["split"] != split for r in rs):
            sys.exit(f"protein {name} occurs in multiple dataset splits")
        stats.append({"entry": name, "split": split, "n_candidates": len(rs), "rmsd_range_A": spread,
                      "x_min": float(x.min()), "x_max": float(x.max()), "y_min": float(y.min()), "y_max": float(y.max()),
                      "spearman_rho": spearman(x, y),
                      "in_summary": bool(len(rs) >= 4 and np.isfinite(spearman(x, y))),
                      "in_subset": bool(subset_min is not None and len(rs) >= 4 and spread >= subset_min and np.isfinite(spearman(x, y)))})
    return by, stats


# ------------------------------------------------------------------------------------------------------ drawing
def setup(theme):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    c = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"],
        "text.color": c["text"], "axes.labelcolor": c["text2"], "xtick.color": c["text2"], "ytick.color": c["text2"],
        "axes.edgecolor": c["muted"], "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "svg.fonttype": "none", "figure.dpi": 100,
    })
    return plt, c


def colour_of(split, c):
    return c[SPLIT_COLOURS[split]] if split in SPLIT_COLOURS else c["muted"]


def summarise(stats, a, splits):
    """{label: (n, median, lo, hi, n_positive)} for the main summary (all proteins) and the >= subset."""
    out = {}
    for block, flag in (("main", "in_summary"), ("subset", "in_subset")):
        for label, sel in (("all", lambda s: True),) + tuple((sp, (lambda s, sp=sp: s["split"] == sp)) for sp in splits if sp):
            vals = [s["spearman_rho"] for s in stats if s[flag] and sel(s)]
            out[(block, label)] = (len(vals),) + bootstrap_median(vals, a.n_boot) + (int(sum(v > 0 for v in vals)),)
    return out


N_BINS = 5


def bin_of(x):
    """Group index 0..4 of every candidate of ONE protein from the within-protein rank of its accessibility error. Uses the average
    rank r (1..K) of tied candidates: bin = floor((r - 1) / K * 5), so K = 25 gives exactly 5 candidates per group."""
    k = len(x)
    return np.minimum(((rankdata(x) - 1.0) / k * N_BINS).astype(int), N_BINS - 1)


def bin_table(by, in_sum):
    """One row per protein and group: mean within-protein structural rank percentile and mean raw structural error."""
    rows = []
    for name, rs in by.items():
        if name not in in_sum or len(rs) < N_BINS:
            continue
        x = np.array([r["x"] for r in rs])
        b = bin_of(x)
        for g in range(N_BINS):
            sel = b == g
            if sel.any():
                rows.append({"entry": name, "split": rs[0]["split"], "bin": g + 1, "n": int(sel.sum()),
                             "mean_y_rank": float(np.mean([r["y_rank"] for r, m in zip(rs, sel) if m])),
                             "mean_y": float(np.mean([r["y"] for r, m in zip(rs, sel) if m]))})
    return rows


def figure1(rows, stats, by, a, plt, c):
    in_sum = {s["entry"] for s in stats if s["in_summary"]}
    ylab = Y_CHOICES[a.y][0]
    found = {r["split"] for r in rows}
    splits = [s for s in SPLIT_COLOURS if s in found] + sorted(found - set(SPLIT_COLOURS))
    summary = summarise(stats, a, splits)
    bt = bin_table(by, in_sum)
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30, left=0.07, right=0.98, top=0.80, bottom=0.25)

    # --- A: structural rank by accessibility-rank group, one value per protein and group
    ax = fig.add_subplot(gs[0])
    ents = sorted({r["entry"] for r in bt})
    data = [[r["mean_y_rank"] for r in bt if r["bin"] == g] for g in range(1, N_BINS + 1)]
    ax.axhline(0.5, color=c["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.boxplot(data, positions=range(1, N_BINS + 1), widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=c["surface"], edgecolor=c["text2"], lw=1.2), medianprops=dict(color=c["text"], lw=2.2),
               whiskerprops=dict(color=c["text2"], lw=1.1), capprops=dict(color=c["text2"], lw=1.1), zorder=2)
    rng = np.random.default_rng(0)
    for g in range(1, N_BINS + 1):
        pts = [r for r in bt if r["bin"] == g]
        ax.scatter(g + rng.uniform(-0.2, 0.2, len(pts)), [r["mean_y_rank"] for r in pts], s=9,
                   color=[colour_of(r["split"], c) for r in pts], alpha=0.55, linewidths=0, zorder=3)
    ax.set_xticks(range(1, N_BINS + 1))
    ax.set_xticklabels(["1\nclosest", "2", "3", "4", "5\nfurthest"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Candidate group by SASA error\n(within each protein)")
    ax.set_ylabel("Structural rank percentile within protein\n(0 = most accurate)")
    ax.grid(axis="y", color=c["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("A  Structural rank by SASA-error group")
    med = [np.median(v) if v else np.nan for v in data]

    # --- B: per-protein rho, one row per split
    bx = fig.add_subplot(gs[1])
    rows_b = [("all", "All proteins")] + [(sp, SPLIT_LABELS.get(sp, sp or "Unspecified")) for sp in splits if sp]
    bx.axvline(0, color=c["muted"], lw=1, zorder=1)
    for yi, (key, lab) in enumerate(rows_b):
        vals = [(s["spearman_rho"], s["split"]) for s in stats if s["in_summary"] and (key == "all" or s["split"] == key)]
        if not vals:
            continue
        v = np.array([x for x, _ in vals])
        bx.scatter(v, yi + rng.uniform(-0.28, 0.28, len(v)), s=16, alpha=0.6, linewidths=0,
                   color=[colour_of(sp, c) for _, sp in vals], zorder=2)
        n_, m_, lo_, hi_, npos_ = summary[("main", key)]
        bx.plot([lo_, hi_], [yi, yi], color=c["text"], lw=1.6, solid_capstyle="butt", zorder=1.5)
        bx.plot([m_, m_], [yi - 0.4, yi + 0.4], color=c["text"], lw=2.4, zorder=1.6)
    bx.set_yticks(range(len(rows_b)))
    bx.set_yticklabels([f"{lab}  (n = {summary[('main', key)][0]})" for key, lab in rows_b])
    bx.set_ylim(len(rows_b) - 0.45, -0.7)
    bx.set_xlim(-1.02, 1.02)
    bx.set_xlabel("Per-protein Spearman \u03c1")
    bx.grid(axis="x", color=c["grid"], lw=0.6, zorder=0)
    bx.set_axisbelow(True)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_title("B  Per-protein Spearman \u03c1")

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=colour_of(sp, c), label=SPLIT_LABELS.get(sp, sp)) for sp in splits if sp],
               loc="lower right", bbox_to_anchor=(0.98, 0.075), ncol=len(splits), fontsize=9, labelcolor=c["text2"], frameon=False)
    ax.text(1.5, 0.51, "chance", color=c["muted"], fontsize=8, va="bottom", ha="center")
    n, medr, lo, hi, npos = summary[("main", "all")]
    diffs = []
    for e in ents:
        g1 = [r for r in bt if r["entry"] == e and r["bin"] == 1]
        g5 = [r for r in bt if r["entry"] == e and r["bin"] == N_BINS]
        if g1 and g5:
            diffs.append(g5[0]["mean_y"] - g1[0]["mean_y"])
    dmed, dlo, dhi = bootstrap_median(diffs, a.n_boot)
    unit = " \u00c5" if a.y == "rmsd_ca" else ""
    fig.text(0.07, 0.03,
             f"Median \u03c1 {medr:+.2f} [95 % CI {lo:+.2f}, {hi:+.2f}]; \u03c1 > 0 in {npos} of {n} proteins.   "
             f"Group 5 minus group 1 mean {YSHORT[a.y]}: {dmed:+.3g}{unit} [{dlo:+.3g}, {dhi:+.3g}].",
             fontsize=9, color=c["text2"], ha="left", va="bottom", linespacing=1.5)
    fig.suptitle("Do candidates with lower accessibility error tend to be structurally more accurate?",
                 x=0.07, y=0.985, ha="left", fontsize=14, fontweight="bold", color=c["text"])
    fig.text(0.07, 0.885, f"{len(rows)} candidates, {len(stats)} proteins. x = accessibility error ({SHORT_X[a.x].split('(')[-1].rstrip(')')}) vs PDB-derived SASA; y = {YSHORT[a.y]} vs PDB structure.",
             fontsize=9.5, color=c["text2"], ha="left", va="center", linespacing=1.5)
    return fig, bt


def figure2(items, stats, a, plt, c, page, n_pages):
    """One page of per-protein panels; `items` = [(entry, rows)] for this page."""
    from matplotlib.ticker import MaxNLocator
    n = len(items)
    ncol = min(a.cols, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol + 0.9, 2.7 * nrow + 1.5), squeeze=False)
    st = {s["entry"]: s for s in stats}
    ylab = Y_CHOICES[a.y][0]
    for ax_ in axes.ravel():
        ax_.set_visible(False)
    for i, (name, rs) in enumerate(items):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        s = st[name]
        ax.scatter([r["x"] for r in rs], [r["y"] for r in rs], s=18, color=colour_of(s["split"], c), alpha=0.85,
                   linewidths=0.3, edgecolors=c["surface"], zorder=3)
        rho = s["spearman_rho"]
        ax.set_title(f"{name}   " + (f"\u03c1 = {rho:+.2f}" if np.isfinite(rho) else "\u03c1 undefined"), fontsize=9.5, loc="left", pad=14,
                     color=c["text"])
        ax.text(0.0, 1.02, f"RMSD range {s['rmsd_ca_range_label']} \u00b7 {s['split'] or 'all'}", transform=ax.transAxes, fontsize=7.5,
                color=c["text2"], va="bottom")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8, length=2.5)
        ax.margins(0.14)
        ax.grid(color=c["grid"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.ticklabel_format(style="plain", useOffset=False)
        if i % ncol == 0:
            ax.set_ylabel(ylab.split(" to ")[0] if a.y == "rmsd_ca" else ylab, fontsize=8.5)
    fig.supxlabel(X_LABELS[a.x].replace("\n", " "), fontsize=10, color=c["text2"])
    fig.suptitle(f"Per-protein view, raw values (each panel has its own axes) \u2014 page {page} of {n_pages}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=c["text"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.965), h_pad=1.6, w_pad=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", help="score_candidates.py output folder (candidate_scores.csv + structure_candidate_scores.csv)")
    ap.add_argument("--accessibility-csv", nargs="+", help="one or more accessibility score tables, combined before matching")
    ap.add_argument("--structure-csv", nargs="+", help="one or more structural score tables; a combined score table may be used on both sides")
    ap.add_argument("--x", default="ac_error", choices=["ac_error", "ac_corr_error", "ac_rmse"], help="accessibility error column")
    ap.add_argument("--y", default="rmsd_ca", choices=list(Y_CHOICES), help="structural metric (RMSD is primary)")
    ap.add_argument("--subset-rmsd-range", type=float, default=None,
                    help="supplementary subset: proteins whose RMSD range (highest - lowest RMSD to the experimental structure) is at least "
                         "this many A. Disabled by default; nothing is excluded from the main summary")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--raw-pages", action="store_true", help="also write the per-protein raw-value pages (fig2_per_protein_raw_p<k>); off by default")
    ap.add_argument("--cols", type=int, default=5, help="columns of the per-protein figure")
    ap.add_argument("--per-page", type=int, default=20, help="proteins per page of the per-protein figure")
    ap.add_argument("--theme", default="light", choices=list(THEMES))
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only-splits", nargs="+", help="e.g. development validation (default: all found in the candidate paths)")
    ap.add_argument("--strict", action="store_true", help="stop (exit 1) if any row is unmatched between the tables or has a missing/non-finite value")
    a = ap.parse_args()

    acc_csv = a.accessibility_csv or (Path(a.scores_dir) / "candidate_scores.csv" if a.scores_dir else None)
    st_csv = a.structure_csv or (Path(a.scores_dir) / "structure_candidate_scores.csv" if a.scores_dir else None)
    if not acc_csv or not st_csv:
        ap.error("give --scores-dir, or both --accessibility-csv and --structure-csv")
    rows, report = load(acc_csv, st_csv, a.x, a.y, a.strict)
    n_before_split = len(rows)
    if a.only_splits:
        rows = [r for r in rows if r["split"] in set(a.only_splits)]
        print(f"--only-splits {a.only_splits}: kept {len(rows)} of {n_before_split} candidates")
    if not rows:
        ap.error("no candidates remain after dataset filtering")
    by, stats = group(rows, a.subset_rmsd_range)
    for s in stats:
        s["rmsd_ca_range_label"] = f"{s['rmsd_range_A']:.2g} Å" if np.isfinite(s["rmsd_range_A"]) else "n/a"

    plt, c = setup(a.theme)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f1, bt = figure1(rows, stats, by, a, plt, c)
    f1.savefig(out / "fig1_accessibility_vs_accuracy.png", dpi=200, bbox_inches="tight")
    f1.savefig(out / "fig1_accessibility_vs_accuracy.svg", bbox_inches="tight")
    items = list(by.items())
    n_pages = max(1, int(np.ceil(len(items) / a.per_page)))
    for pg in range(n_pages if a.raw_pages else 0):
        f2 = figure2(items[pg * a.per_page:(pg + 1) * a.per_page], stats, a, plt, c, pg + 1, n_pages)
        f2.savefig(out / f"fig2_per_protein_raw_p{pg + 1}.png", dpi=200)
        f2.savefig(out / f"fig2_per_protein_raw_p{pg + 1}.svg")
        plt.close(f2)

    # traceability: what went in, what was dropped, what was plotted
    import hashlib
    import json
    with open(out / "dropped_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry", "candidate", "reason"])
        w.writeheader()
        w.writerows(report["dropped_nonfinite"])
    counts = {}
    for r in rows:
        counts[r["entry"]] = counts.get(r["entry"], 0) + 1
    k_max = max(counts.values())
    short = {k: v for k, v in counts.items() if v < k_max}
    if short:
        print(f"WARNING: proteins with fewer plotted candidates than the maximum ({k_max}): {short}", file=sys.stderr)
    def inputs(paths):
        return [{"path": str(p), "sha1": hashlib.sha1(Path(p).read_bytes()).hexdigest()} for p in input_paths(paths)]
    (out / "plot_settings.json").write_text(json.dumps({
        "accessibility_inputs": inputs(acc_csv), "structure_inputs": inputs(st_csv),
        "x_column": a.x, "y_metric": a.y, "x_meaning": "accessibility error against the reference computed from the experimental structure",
        "subset_rmsd_range_A": a.subset_rmsd_range, "only_splits": a.only_splits, "n_boot": a.n_boot, "strict": a.strict,
        "rows_accessibility_table": report["rows_accessibility_table"], "rows_structure_table": report["rows_structure_table"],
        "matched": report["matched"], "plotted": len(rows), "only_in_accessibility_table": report["only_in_accessibility_table"],
        "only_in_structure_table": report["only_in_structure_table"], "dropped_nonfinite": len(report["dropped_nonfinite"]),
        "candidates_per_protein": {"min": min(counts.values()), "max": k_max}}, indent=1))
    print(f"matched {report['matched']} of {report['rows_accessibility_table']} (accessibility table) / {report['rows_structure_table']} (structure table) rows; "
          f"plotted {len(rows)}; dropped for missing values {len(report['dropped_nonfinite'])}; candidates per protein {min(counts.values())}..{k_max}")

    with open(out / "bin_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry", "split", "bin", "n", "mean_y_rank", "mean_y"])
        w.writeheader()
        w.writerows({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in bt)
    with open(out / "plotted_points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "split", "candidate", "x_accessibility_error", "y_structural_error", "x_rank_within_protein", "y_rank_within_protein"])
        for r in rows:
            w.writerow([r["entry"], r["split"], r["candidate"], f"{r['x']:.6g}", f"{r['y']:.6g}", f"{r['x_rank']:.4f}", f"{r['y_rank']:.4f}"])
    cols = ["entry", "split", "n_candidates", "rmsd_range_A", "x_min", "x_max", "y_min", "y_max", "spearman_rho", "in_summary", "in_subset"]
    with open(out / "per_protein_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in stats:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items() if k in cols})

    allv = [s["spearman_rho"] for s in stats if s["in_summary"]]
    subv = [s["spearman_rho"] for s in stats if s["in_subset"]]
    summaries = [("ALL proteins", allv)]
    if a.subset_rmsd_range is not None:
        summaries.append((f"subset, RMSD range >= {a.subset_rmsd_range:g} A", subv))
    for label, v in summaries:
        med, lo, hi = bootstrap_median(v, a.n_boot)
        print(f"{label}: n = {len(v)}, median Spearman rho {med:+.3f}  95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]; rho > 0 in {sum(x > 0 for x in v)} of {len(v)}")
    undefined = [s["entry"] for s in stats if not s["in_summary"]]
    if undefined:
        print(f"rho undefined (fewer than 4 candidates or a constant axis), not in the summary: {undefined}")
    print(f"{len(rows)} candidates, {len(stats)} proteins; wrote {out}/fig1_accessibility_vs_accuracy.png|svg, "
          + (f"fig2_per_protein_raw_p1..p{n_pages}.png|svg, " if a.raw_pages else "")
          + "plotted_points.csv, per_protein_stats.csv, bin_summary.csv, dropped_rows.csv, plot_settings.json")


if __name__ == "__main__":
    main()
