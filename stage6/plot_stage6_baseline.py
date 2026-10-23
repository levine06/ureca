#!/usr/bin/env python
"""Stage 6 baseline figure: A (normal OpenFold selection) versus B (accessibility reranking at the frozen lambda), from SAVED Stage 4 results.

Reads `rerank_per_protein.csv` written by `rerank_candidates.py evaluate`. Nothing is re-inferred, re-selected or re-shuffled: A and B are the
OpenFold choice and the combined-rule choice that Stage 4 already scored over the SAME 25 candidates per protein. Generation budget is shared;
accessibility scoring and reranking add overhead. These are reused baseline results, not a new Stage 6 run.

Figure: two panels, development + validation (lambda was fitted there) and Test A + Test B. One dot per protein = RMSD(B) - RMSD(A) in Angstrom
(negative = B better), a zero line for no change, a median marker (bar, with a bootstrap 95 % interval over proteins) and the improved / unchanged /
worsened counts. Not drawn: the lambda curve, accessibility-alone and shuffled results, the A/B/C-like classification. Oracle values stay in the table.

    python stage6/plot_stage6_baseline.py --results stage4/results/rerank_per_protein.csv --out stage6_baseline
Writes stage6_baseline.png|svg, stage6_baseline_per_protein.csv and stage6_baseline_summary.csv in --out.
The lambda in the legend is read from the results file (and checked to be one value); `--expect-lambda 0.1` stops if it differs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage4"))
import rerank_candidates as rr                                   # noqa: E402  (reuse: themes, plot setup, counts, csv writer)
from accessfold.controls import bootstrap_mean, bootstrap_median  # noqa: E402

PANELS = (("tuned", ("development", "validation"), "Development + validation", "lambda selected on validation only"),
          ("heldout", ("test_a", "test_b"), "Test A + Test B", "frozen lambda; inspected in Stage 3, never tuned on"))


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v) if v != "" else float("nan")
            except ValueError:
                d[k] = v
        rows.append(d)
    need = {"entry", "split", "rmsd_openfold", "rmsd_combined", "rmsd_oracle"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{path} lacks the columns {sorted(need - set(rows[0] if rows else []))}")
    for d in rows:
        if "delta_combined" not in d or not np.isfinite(d["delta_combined"]):
            d["delta_combined"] = d["rmsd_combined"] - d["rmsd_openfold"]       # B - A, derived from the saved RMSDs
        d["delta"] = d["rmsd_combined"] - d["rmsd_openfold"]
    return rows


def lambda_of(rows, expect):
    if "lambda" not in rows[0]:
        sys.exit("the results file has no `lambda` column, so the figure cannot state the lambda it shows")
    lams = {float(d["lambda"]) for d in rows}
    if len(lams) != 1:
        sys.exit(f"the results file mixes lambdas {sorted(lams)}")
    lam = lams.pop()
    if expect is not None and abs(lam - expect) > 1e-12:
        sys.exit(f"STOP: the saved results use lambda = {lam:g}, not the expected {expect:g}; check which Stage 4 run you are plotting")
    return lam


def summarise(rows, name, n_boot):
    d = np.array([r["delta"] for r in rows])
    m, lo, hi = bootstrap_mean(d, n_boot, np.random.default_rng(1))
    md, mlo, mhi = bootstrap_median(d, n_boot, np.random.default_rng(2))
    imp, same, wor = rr.counts(d)
    return {"group": name, "n_proteins": len(d), "median_B_minus_A_A": md, "median_lo": mlo, "median_hi": mhi, "mean_B_minus_A_A": m, "mean_lo": lo, "mean_hi": hi,
            "n_B_better": imp, "n_unchanged": same, "n_B_worse": wor, "n_choice_changed": int(sum(bool(r.get("changed_choice") in (True, "True")) for r in rows)),
            "mean_A_rmsd_A": float(np.mean([r["rmsd_openfold"] for r in rows])), "mean_B_rmsd_A": float(np.mean([r["rmsd_combined"] for r in rows])),
            "mean_oracle_rmsd_A": float(np.mean([r["rmsd_oracle"] for r in rows])),
            "mean_oracle_minus_A_A": float(np.mean([r["rmsd_oracle"] - r["rmsd_openfold"] for r in rows]))}


def figure(rows, lam, theme, n_boot):
    plt, c = rr.setup(theme)
    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08, left=0.09, right=0.98, top=0.80, bottom=0.18)
    rng = np.random.default_rng(0)
    axes, summary = [], []
    for k, (key, splits, title, note) in enumerate(PANELS):
        ax = fig.add_subplot(gs[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        rs = [r for r in rows if r["split"] in splits]
        ax.axhline(0, color=c["muted"], lw=1.2, zorder=1)
        if rs:
            d = np.array([r["delta"] for r in rs])
            ax.scatter(rng.uniform(-0.28, 0.28, len(d)), d, s=30, alpha=0.8, linewidths=0, color=[c[r["split"]] for r in rs], zorder=3)
            s = summarise(rs, title, n_boot)
            summary.append(s)
            ax.plot([-0.4, 0.4], [s["median_B_minus_A_A"]] * 2, color=c["text"], lw=3, zorder=5)
            ax.errorbar([0.0], [s["median_B_minus_A_A"]], yerr=[[s["median_B_minus_A_A"] - s["median_lo"]], [s["median_hi"] - s["median_B_minus_A_A"]]],
                        fmt="none", ecolor=c["text"], elinewidth=1.4, capsize=5, zorder=5)
        ax.set_yscale("symlog", linthresh=0.1)
        ticks = [-10, -3, -1, -0.3, -0.1, 0, 0.1, 0.3, 1, 3, 10]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in ticks])
        dv = np.array([r["delta"] for r in rows])
        ax.set_ylim(-max(1.0, float(-dv.min()) * 1.4), max(0.3, float(dv.max()) * 1.6))
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.grid(axis="y", color=c["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{title} (n = {len(rs)})", pad=10)
        if k == 0:
            ax.set_ylabel("RMSD(B) − RMSD(A) (Å)\nnegative = better; symlog scale")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    ax0 = axes[0]
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c[s], label=rr.SPLIT_LABEL[s]) for s in rr.SPLIT_LABEL if any(r["split"] == s for r in rows)]
    handles += [Line2D([], [], color=c["text"], lw=3, label="Median (95% bootstrap CI)")]
    ax0.legend(handles=handles, loc="upper center", bbox_to_anchor=(1.04, -0.09), ncol=6, fontsize=8.5, labelcolor=c["text2"])
    fig.suptitle("Accessibility reranking vs OpenFold", y=0.98, fontsize=14, fontweight="bold", color=c["text"])
    fig.text(0.535, 0.89, f"A: OpenFold confidence selection   |   B: accessibility reranking (λ = {lam:g})",
             fontsize=10, color=c["text2"], ha="center")
    return fig, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="rerank_per_protein.csv from `rerank_candidates.py evaluate`")
    ap.add_argument("--out", default="stage6_baseline")
    ap.add_argument("--expect-lambda", type=float, help="stop if the saved results use a different lambda (e.g. 0.1)")
    ap.add_argument("--theme", default="light", choices=list(rr.THEMES))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    rows = load(a.results)
    lam = lambda_of(rows, a.expect_lambda)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, summary = figure(rows, lam, a.theme, a.n_boot)
    fig.savefig(out / "stage6_baseline.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out / "stage6_baseline.svg", bbox_inches="tight", pad_inches=0.15)
    per = [{"entry": r["entry"], "split": r["split"], "lambda": lam, "rmsd_A_openfold": r["rmsd_openfold"], "rmsd_B_reranked": r["rmsd_combined"],
            "B_minus_A_A": r["delta"], "rmsd_oracle": r["rmsd_oracle"], "oracle_minus_A_A": r["rmsd_oracle"] - r["rmsd_openfold"],
            "choice_changed": r.get("changed_choice", ""),
            "B_vs_A": "B better" if r["delta"] < -rr.TOL else ("B worse" if r["delta"] > rr.TOL else "unchanged")} for r in rows]
    rr.write_csv(out / "stage6_baseline_per_protein.csv", per)
    rr.write_csv(out / "stage6_baseline_summary.csv", summary)
    for s in summary:
        print(f"{s['group']:<26} n={s['n_proteins']:>3}  median {s['median_B_minus_A_A']:+.3f} [{s['median_lo']:+.3f}, {s['median_hi']:+.3f}] A  "
              f"mean {s['mean_B_minus_A_A']:+.3f}  B better/unchanged/worse {s['n_B_better']}/{s['n_unchanged']}/{s['n_B_worse']}  "
              f"(oracle - A: {s['mean_oracle_minus_A_A']:+.3f} A on average)")
    print(f"lambda = {lam:g}; wrote {out}/stage6_baseline.png|svg, stage6_baseline_per_protein.csv, stage6_baseline_summary.csv")


if __name__ == "__main__":
    main()
