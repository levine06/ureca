#!/usr/bin/env python
"""Stage 3 control figures: does the accessibility signal survive the controls, and does choosing by it help?

Reads the output folder of run_controls.py and draws, for one accessibility error (--error-metric mae|corr) and one structural
metric (--metric rmsd_ca primary), two figures:

  controls_<err>_<metric>_overview     A  per protein: the real rho (dot) against the range of that protein's own SHUFFLED rho
                                          (bar = 2.5-97.5 %, tick = mean); a real dot clearly right of its bar = the real
                                          reference profile carries information a shuffled one does not.
                                       B  across proteins: median rho under each control (dot + 95 % bootstrap interval over
                                          proteins). The grey band is the 95 % range of the protein-level shuffle null of the
                                          median (each draw takes one shuffle repetition per protein).
                                       C  top-1 selection: mean rank percentile of the candidate with the lowest accessibility
                                          error (0 = it was the most accurate candidate, 0.5 = random pick, 1 = the worst), under
                                          each control, next to the reference rules random / oracle / ranking_top1 (the
                                          candidate ranked first by the supplied ranking column). Band = 95 % range of the shuffle null.
  controls_<err>_<metric>_per_protein  one row per protein: real pick percentile (dot) against the range of the shuffled picks
                                       (bar), the model's own top-1 (cross), the oracle (0) and random (0.5) rules.

How to read it (the question is whether accessibility is informative data for structure prediction, not whether two numbers
correlate): the real values must sit clearly beyond the shuffle band, and the controls that keep some real information
(reverse, other_protein) should land between real and shuffle or at the null. Compare `real` with the SHUFFLE null, not with 0
(rho) or 0.5 (pick): neither null is guaranteed to be centred there (clearly not for "mae"; read the generated null for "corr" too).
All proteins form the summary; no RMSD-range subset is displayed. Panel A is sorted by real rho minus the mean shuffled rho,
with the largest advantage at the top. The reference accessibility here is COMPUTED from the experimental structure,
so passing these controls means the idea is not refuted on the best-case input, not that real dye data will work.

    python scripts/plot_controls.py --controls-dir controls_out --out figures_controls
Needs numpy, matplotlib (and the accessfold package for the protein-level null).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e1", band="#c9c8c2",
                  dev="#2a78d6", val="#eb6834", test_a="#16865c", test_b="#a345a0"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8c8b84", grid="#2e2e2c", band="#4a4a46",
                 dev="#3987e5", val="#d95926", test_a="#51c993", test_b="#d78bd0"),
}
CONTROL_ORDER = ["real", "shuffle", "reverse", "other_protein", "dropout:0.5", "dropout:0.2", "binary:0.2", "binary:median"]
CONTROL_LABEL = {"real": "real reference profile", "shuffle": "shuffle (residue values permuted)", "reverse": "reverse along the sequence",
                 "other_protein": "another protein's profile", "dropout:0.5": "keep 50 % of residues", "dropout:0.2": "keep 20 % of residues",
                 "binary:0.2": "buried / exposed (threshold 0.2)", "binary:median": "buried / exposed (median split)",
                 "random": "random pick (reference)", "oracle": "oracle: best candidate (reference)",
                 "ranking_top1": "supplied ranking: top-ranked (reference)"}
ERR_LABEL = {"mae": "mean absolute error of relative accessibility", "corr": "1 − correlation of the accessibility profiles"}
ERROR_NAME = {"mae": "accessibility MAE", "corr": "accessibility 1 − r"}
ERROR_EXPLANATION = {
    "mae": "MAE = mean absolute ACCESSIBILITY error: the average absolute difference between candidate and reference relative accessibility values.",
    "corr": "1 − Pearson r = accessibility pattern error: low when the same residues tend to be exposed or buried.",
}
PICK_RULE = {"mae": "lowest accessibility MAE", "corr": "lowest pattern error (1 − Pearson r)"}
METRIC_LABEL = {"rmsd_ca": "Cα RMSD", "lddt_ca": "1 − lDDT-Cα", "tm_score": "1 − TM-score"}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def colour_of(split, c):
    key = {"development": "dev", "validation": "val", "test_a": "test_a", "test_b": "test_b"}.get(split)
    return c[key] if key else c["muted"]


def split_handles(d, c, prefix):
    from matplotlib.lines import Line2D
    labels = {"development": "development", "validation": "validation", "test_a": "Test A", "test_b": "Test B"}
    present = set(d.split.values())
    order = [s for s in labels if s in present] + sorted(present - labels.keys())
    return [Line2D([], [], marker="o", ls="", color=colour_of(s, c),
                   label=f"{prefix}, {labels.get(s, s or 'unspecified')}") for s in order]


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


class Data:
    """Everything the figures need, with the same matching rule as run_controls.py."""

    def __init__(self, folder, em, metric):
        folder = Path(folder)
        self.em, self.metric = em, metric
        self.settings = json.loads((folder / "settings.json").read_text())
        self.per = [r for r in read(folder / "controls_per_entry.csv") if r["error_metric"] == em]
        self.summ = [r for r in read(folder / "controls_summary.csv") if r["error_metric"] == em and r["metric"] == metric]
        self.sel = [r for r in read(folder / "selection_summary.csv") if r["error_metric"] == em and r["metric"] == metric]
        self.null = np.load(folder / "shuffle_null.npz")
        rc = self.settings.get("rank_csv")
        self.rank_label = f"{Path(rc).name}:{self.settings.get('rank_column', 'rank')}" if rc else "the supplied ranking"
        if not self.per:
            sys.exit(f"no rows for error metric {em!r} in {folder}/controls_per_entry.csv (available: {sorted({r['error_metric'] for r in read(folder / 'controls_per_entry.csv')})})")
        self.entries = sorted({r["entry"] for r in self.per})
        self.split = {r["entry"]: r["split"] for r in self.per}
        self.in_subset = {r["entry"]: r["in_subset"] == "True" for r in self.per if r["control"] == "real"}
        self.value = {}
        for r in self.per:
            self.value[(r["control"], r["entry"])] = {"rho": fnum(r.get(f"rho_{metric}")), "pct": fnum(r.get(f"pick_pct_{metric}")),
                                                       "delta": fnum(r.get(f"pick_delta_{metric}"))}

    def null_of(self, entry, kind):
        k = f"{self.em}|{entry}|{self.metric}|{kind}"
        return self.null[k] if k in self.null.files else np.array([])

    def members(self, which):
        return [e for e in self.entries if which == "all" or self.in_subset.get(e, False)]

    def matched(self, which, kind):
        """Proteins with a defined real value and (when a shuffle was run) a defined shuffle null: the run_controls.py rule."""
        shuffled = any(self.null_of(e, kind).size for e in self.entries)
        out = []
        for e in self.members(which):
            if not np.isfinite(self.value.get(("real", e), {}).get(kind, np.nan)):
                continue
            if shuffled and not np.isfinite(self.null_of(e, kind)).any():
                continue
            out.append(e)
        return out

    def row(self, table, which, control):
        for r in table:
            if r["set"] == which and r["control"] == control:
                return r
        return None

    def null_band(self, which, kind, statistic, n_draw=10000, seed=0):
        from accessfold.controls import protein_level_null
        ents = [e for e in self.matched(which, kind) if self.null_of(e, kind).size]
        arr = {e: self.null_of(e, kind) for e in ents}
        if not arr:
            return None
        nm = protein_level_null(arr, np.random.default_rng(seed), n_draw, statistic=statistic)
        return (float(np.quantile(nm, 0.025)), float(np.mean(nm)), float(np.quantile(nm, 0.975))) if nm.size else None


def overview(d, a, plt, c):
    n = len(d.entries)
    fig = plt.figure(figsize=(15.0, max(9.5, 0.23 * n + 3.4)))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0], wspace=0.55, hspace=0.38, left=0.07, right=0.985,
                          top=0.86, bottom=0.115)
    m = d.metric

    # ---------------------------------------------------------------- A: per-protein real rho vs its shuffle range
    ax = fig.add_subplot(gs[:, 0])
    def advantage(entry):
        real = d.value[("real", entry)]["rho"]
        null = d.null_of(entry, "rho")
        null = null[np.isfinite(null)]
        return float(real - null.mean()) if np.isfinite(real) and null.size else -np.inf

    order = sorted(d.entries, key=lambda e: (advantage(e), e))
    for i, e in enumerate(order):
        nul = d.null_of(e, "rho")
        nul = nul[np.isfinite(nul)]
        if nul.size:
            lo, hi = np.quantile(nul, [0.025, 0.975])
            ax.plot([lo, hi], [i, i], color=c["band"], lw=6, solid_capstyle="butt", zorder=1)
            ax.plot([nul.mean()] * 2, [i - 0.32, i + 0.32], color=c["text2"], lw=1.2, zorder=2)
        v = d.value[("real", e)]["rho"]
        if np.isfinite(v):
            ax.scatter([v], [i], s=36, color=colour_of(d.split[e], c), zorder=3,
                       edgecolors=c["surface"], linewidths=0.6)
    ax.axvline(0, color=c["muted"], lw=1, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{e}  (Δρ {advantage(e):+.2f})" if np.isfinite(advantage(e)) else f"{e}  (Δρ n/a)" for e in order], fontsize=8)
    ax.set_ylim(-1, n)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=c["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Spearman ρ across candidates: {ERROR_NAME[d.em]} vs {METRIC_LABEL[m]}\n"
                  "Positive = smaller accessibility error tends to accompany smaller structural error")
    ax.set_title("A  Real vs shuffled ρ: largest advantage at top")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c["band"], lw=6, label="95 % range of shuffled ρ (tick = mean)"),
                       *split_handles(d, c, "real ρ")],
              loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, labelcolor=c["text2"], ncol=2)

    # ---------------------------------------------------------------- B and C: across proteins, one row per control
    def across(ax, table, controls, key, lo_key, hi_key, band, null_kind, xlabel, title, refline=None, xlim=None, extra=2.4, band_sub=None):
        ys = {ctl: i for i, ctl in enumerate(controls)}
        if band is not None:
            ax.axvspan(band[0], band[2], color=c["band"], alpha=0.55, lw=0, zorder=0)
            ax.axvline(band[1], color=c["text2"], lw=1, ls=(0, (3, 3)), zorder=1)
        if band_sub is not None:
            ax.axvspan(band_sub[0], band_sub[2], facecolor="none", edgecolor=c["text2"], ls=(0, (4, 3)), lw=1.1, zorder=1)
            ax.axvline(band_sub[1], color=c["text2"], lw=1, ls=(0, (1, 2)), zorder=1)
        if refline is not None:
            ax.axvline(refline, color=c["muted"], lw=1, zorder=1)
        for which, marker, dy in (("all", "o", 0.0),):
            for ctl in controls:
                r = d.row(table, which, ctl)
                if r is None or not np.isfinite(fnum(r.get(key))):
                    continue
                v, lo, hi = fnum(r[key]), fnum(r[lo_key]), fnum(r[hi_key])
                y = ys[ctl] + dy
                real = ctl == "real"
                col = c["text"] if real else c["text2"]
                ax.plot([lo, hi], [y, y], color=col, lw=2.0 if which == "all" else 1.3, alpha=0.9 if which == "all" else 0.7, zorder=2)
                ax.scatter([v], [y], s=(52 if real else 38) if which == "all" else 34, marker=marker, zorder=3,
                           facecolor=col if which == "all" else c["surface"], edgecolor=col, linewidths=1.3)
        ax.set_yticks(range(len(controls)))
        def lab(x):
            ra = d.row(table, "all", x)
            ns = [ra["n_proteins"]] if ra else []
            return CONTROL_LABEL.get(x, x) + (f"  (n = {' / '.join(ns)})" if ns else "")
        ax.set_yticklabels([lab(x) for x in controls], fontsize=8)
        ax.set_ylim(len(controls) - 0.4 + extra, -0.7)          # `extra` blank rows at the bottom hold the summary box
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(axis="x", color=c["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_title(title)

    present = [x for x in CONTROL_ORDER if d.row(d.summ, "all", x)]
    bx = fig.add_subplot(gs[0, 1])
    band = d.null_band("all", "rho", "median")
    across(bx, d.summ, present, "median_rho", "ci95_lo", "ci95_hi", band, "rho",
           "median ρ across proteins (dot; line = 95 % bootstrap interval over proteins)",
           f"B  Ranking agreement: {ERROR_NAME[d.em]} vs {METRIC_LABEL[m]}", refline=0.0, extra=3.4)
    real_row = d.row(d.summ, "all", "real")
    if real_row and band:
        txt = f"real {fnum(real_row['median_rho']):+.2f}; shuffle null {band[1]:+.2f} [{band[0]:+.2f}, {band[2]:+.2f}]"
        if real_row.get("p_median_vs_shuffle"):
            txt += f"\np = {fnum(real_row['p_median_vs_shuffle']):.4f}  (n = {real_row['n_proteins']} matched proteins)"
        bx.text(0.99, 0.02, txt, transform=bx.transAxes, ha="right", va="bottom", fontsize=8, color=c["text"], linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.35", fc=c["surface"], ec=c["grid"], lw=0.8), zorder=6)

    cx = fig.add_subplot(gs[1, 1])
    rules = [x for x in ("ranking_top1", "random", "oracle") if d.row(d.sel, "all", x)]
    sel_controls = [x for x in CONTROL_ORDER if d.row(d.sel, "all", x)] + rules
    band_s = d.null_band("all", "pct", "mean")
    across(cx, d.sel, sel_controls, "mean_pick_pct", "ci95_lo", "ci95_hi", band_s, "pct",
           "mean rank percentile of the picked candidate  (0 = best, 0.5 = random, 1 = worst)",
           f"C  Select by {ERROR_NAME[d.em]}; evaluate by structural rank", refline=0.5, xlim=(-0.02, 1.02), extra=4.4)
    rs = d.row(d.sel, "all", "real")
    if rs and band_s:
        txt = f"real {fnum(rs['mean_pick_pct']):.2f}; shuffle null {band_s[1]:.2f} [{band_s[0]:.2f}, {band_s[2]:.2f}]"
        if rs.get("p_mean_pct_vs_shuffle"):
            txt += f"\np = {fnum(rs['p_mean_pct_vs_shuffle']):.4f}  (n = {rs['n_proteins']} matched proteins)"
        if rs.get("frac_real_better_than_ranking"):
            txt += f"\nvs {d.rank_label}: {100 * fnum(rs['frac_real_better_than_ranking']):.0f}% win rate (ties count half; n = {rs['n_proteins_vs_ranking']})"
        cx.text(0.01, 0.02, txt, transform=cx.transAxes, ha="left", va="bottom", fontsize=8, color=c["text"], linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.35", fc=c["surface"], ec=c["grid"], lw=0.8), zorder=6)
    from matplotlib.lines import Line2D
    cx.legend(handles=[Line2D([], [], marker="o", ls="", color=c["text2"], label="all proteins"),
                       Line2D([], [], color=c["band"], lw=8, label="shuffle null, all proteins (95 % range)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8, labelcolor=c["text2"], ncol=2)

    nn = d.settings.get("n_analysed", n)
    fig.suptitle(f"Controls using {ERROR_NAME[d.em]} against the experimental-structure reference", x=0.07, y=0.985, ha="left",
                 fontsize=14, fontweight="bold", color=c["text"])
    fig.text(0.07, 0.925, ERROR_EXPLANATION[d.em] + " Lower is better; scores are averaged over four probe radii.\n"
             f"{nn} proteins. Accessibility selects/scores candidates; {METRIC_LABEL.get(m, m)} independently evaluates structural accuracy. All proteins are shown.\n"
             "Panel A labels: Δρ = real minus mean shuffled ρ. Compare `real` with the shuffle band, not with 0 or 0.5.\n"
             "Summaries are over matched proteins (defined real value, defined shuffle null and, per control, a defined control value; n in brackets); the bootstrap and the null treat proteins, not candidates, as independent.",
             fontsize=9, color=c["text2"], ha="left", va="center", linespacing=1.5)
    return fig


def per_protein(d, a, plt, c):
    n = len(d.entries)
    fig, ax = plt.subplots(figsize=(10.5, max(7.5, 0.24 * n + 2.6)))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.12)
    order = sorted(d.entries, key=lambda e: (np.inf if not np.isfinite(d.value[("real", e)]["pct"]) else d.value[("real", e)]["pct"]))
    for i, e in enumerate(order):
        nul = d.null_of(e, "pct")
        nul = nul[np.isfinite(nul)]
        if nul.size:
            lo, hi = np.quantile(nul, [0.025, 0.975])
            ax.plot([lo, hi], [i, i], color=c["band"], lw=6, solid_capstyle="butt", zorder=1)
            ax.plot([nul.mean()] * 2, [i - 0.32, i + 0.32], color=c["text2"], lw=1.2, zorder=2)
        of = d.value.get(("ranking_top1", e), {}).get("pct", np.nan)
        if np.isfinite(of):
            ax.scatter([of], [i], marker="x", s=34, color=c["text"], linewidths=1.4, zorder=4)
        v = d.value[("real", e)]["pct"]
        if np.isfinite(v):
            ax.scatter([v], [i], s=36, color=colour_of(d.split[e], c), zorder=3,
                       edgecolors=c["surface"], linewidths=0.6)
    ax.axvline(0.5, color=c["muted"], lw=1, zorder=0)
    ax.axvline(0.0, color=c["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{e}  ({d.value[('real', e)]['delta']:+.2f})" for e in order], fontsize=8)
    ax.set_ylim(-1, n)
    ax.set_xlim(-0.03, 1.03)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=c["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Selected candidate's rank percentile by {METRIC_LABEL[d.metric]} (not its accessibility score)\n"
                  "0 = best structural accuracy, 0.5 = expected random rank, 1 = worst")
    from matplotlib.lines import Line2D
    ax.legend(handles=[*split_handles(d, c, f"pick by {ERROR_NAME[d.em]}"),
                       Line2D([], [], marker="x", ls="", color=c["text"], label=f"top-ranked by {d.rank_label}"),
                       Line2D([], [], color=c["band"], lw=6, label="picks under shuffled reference, 95 % range (tick = mean)")],
              loc="upper center", bbox_to_anchor=(0.4, -0.07), fontsize=8, labelcolor=c["text2"], ncol=2)
    unit = "Å" if d.metric == "rmsd_ca" else "metric units"
    fig.suptitle(f"Select by {PICK_RULE[d.em]}; evaluate by {METRIC_LABEL[d.metric]}", x=0.02, y=0.975, ha="left",
                 fontsize=12.5, fontweight="bold", color=c["text"])
    fig.text(0.02, 0.915, ERROR_EXPLANATION[d.em] + "\n"
             "Select the candidate with the lowest error averaged over four probe radii, then check its structural rank.\n"
             f"Brackets = selected structural error minus the protein's mean ({unit}; negative = better than average). All proteins are shown.",
             fontsize=8.5, color=c["text2"], ha="left", va="center", linespacing=1.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-dir", required=True, help="run_controls.py output folder")
    ap.add_argument("--out", default="figures_controls")
    ap.add_argument("--error-metric", nargs="+", help="mae and/or corr (default: those present)")
    ap.add_argument("--metric", default="rmsd_ca", choices=["rmsd_ca", "lddt_ca", "tm_score"], help="structural metric (RMSD is primary)")
    ap.add_argument("--theme", default="light", choices=list(THEMES))
    a = ap.parse_args()
    folder = Path(a.controls_dir)
    for need in ("controls_per_entry.csv", "controls_summary.csv", "selection_summary.csv", "shuffle_null.npz", "settings.json"):
        if not (folder / need).exists():
            sys.exit(f"{folder / need} not found (run scripts/run_controls.py first; selection_summary.csv needs the current version)")
    avail = sorted({r["error_metric"] for r in read(folder / "controls_per_entry.csv")})
    ems = a.error_metric or avail
    plt, c = setup(a.theme)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for em in ems:
        d = Data(folder, em, a.metric)
        for fig, stem in ((overview(d, a, plt, c), f"controls_{em}_{a.metric}_overview"), (per_protein(d, a, plt, c), f"controls_{em}_{a.metric}_per_protein")):
            fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
            fig.savefig(out / f"{stem}.svg", bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
        print(f"wrote {out}/controls_{em}_{a.metric}_overview.png|svg and controls_{em}_{a.metric}_per_protein.png|svg")


if __name__ == "__main__":
    main()
