"""Calibrate on the development split, validate against Stage 3 conventional relative SASA (probe 1.4 A)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "stage3/truth_accessibility"
sys.path[:0] = [str(TRUTH / "src")]
from accessfold.structures.mmcif import load_chain_from_mmcif  # noqa: E402

from differentiable_accessibility import accessibility  # noqa: E402

BURIED = 0.2  # relative SASA below this is buried (as in Stage 3)
OUT = Path(__file__).parent


def load(entry):
    """Return (CA coords of resolved residues, reference rel. SASA clipped to [0, 1], valid mask) or None."""
    d = np.load(TRUTH / "results" / f"{entry}.npz")
    meta = json.loads(str(d["meta"]))
    s, _, _ = load_chain_from_mmcif(TRUTH / "cif_cache" / f"{meta['pdb_id']}.cif.gz", meta["label_chain_id"], meta["sequence"])
    ca = s.atom_names == "CA"
    xyz = np.full((len(d["mask"]), 3), np.nan)
    xyz[s.residue_index[ca]] = s.coords[ca]
    res = ~np.isnan(xyz[:, 0])  # all resolved C-alphas contribute neighbours
    ref = np.clip(d["rel_1.4"][res], 0, 1)
    valid = d["mask"][res] & ~np.isnan(ref)
    return torch.tensor(xyz[res], dtype=torch.float32), torch.tensor(np.nan_to_num(ref), dtype=torch.float32), valid


INIT = dict(r0=8.0, b=10.0, s=3.0, tau=1.0)


def fit(data, steps=1500):
    """Fit r0, b, s (tau fixed) by minimising MSE over the given proteins."""
    p = {k: torch.tensor(float(v), requires_grad=True) for k, v in INIT.items() if k != "tau"}
    opt = torch.optim.Adam(p.values(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss = sum(((accessibility(x, tau=INIT["tau"], **p) - r)[v] ** 2).mean() for x, r, v in data) / len(data)
        loss.backward()
        opt.step()
    return {**{k: float(v) for k, v in p.items()}, "tau": INIT["tau"]}


def estimate(x, ref, valid, params):
    with torch.no_grad():
        return accessibility(x, **params).numpy()[valid], ref.numpy()[valid]


def metrics(entry, split, a, r):
    pred_b, true_b = a < BURIED, r < BURIED
    tpr, tnr = (pred_b & true_b).sum() / max(true_b.sum(), 1), (~pred_b & ~true_b).sum() / max((~true_b).sum(), 1)
    return dict(entry=entry, split=split, n=len(r), pearson=pearsonr(a, r)[0], spearman=spearmanr(a, r)[0],
                mae=np.abs(a - r).mean(), accuracy=(pred_b == true_b).mean(), balanced_accuracy=(tpr + tnr) / 2)


if __name__ == "__main__":
    summary = pd.read_csv(TRUTH / "results/summary.csv")[["entry", "split"]]
    data = {e: load(e) for e in summary.entry}
    params = fit([data[e] for e in summary.entry[summary.split == "development"]])
    print("calibrated:", {k: round(v, 3) for k, v in params.items()})
    json.dump(params, open(OUT / "results/calibrated_params.json", "w"), indent=1)
    est = {e: estimate(*data[e], params) for e in summary.entry}
    rows = [metrics(e, s, *est[e]) for e, s in zip(summary.entry, summary.split)]
    for s in ("development", "validation"):  # all residues of a split pooled into one vector
        pooled = [np.concatenate(v) for v in zip(*(est[e] for e in summary.entry[summary.split == s]))]
        rows.append(metrics("pooled", f"{s}_pooled", *pooled))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "results/validation_metrics.csv", index=False)
    print(df.groupby("split")[["pearson", "spearman", "mae", "accuracy", "balanced_accuracy"]].agg(["mean", "std"]).round(3).T)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    splits = ("development", "validation")
    pool = {s: [np.concatenate(v) for v in zip(*(est[e] for e in summary.entry[summary.split == s]))] for s in splits}
    grid = np.linspace(0, 1, 200)

    # Scatter (reference vs estimate) with marginal densities; both splits overlaid
    fig = plt.figure(figsize=(11, 6.5))
    outer = fig.add_gridspec(1, 2, width_ratios=(3, 1.3), wspace=0.3)
    gs = outer[0].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    ax, top, right = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])
    box = fig.add_subplot(outer[1])
    for k, s in enumerate(splits):
        ref_v, est_v = pool[s][1], pool[s][0]
        ax.scatter(ref_v, est_v, s=3, alpha=0.25 if k == 0 else 0.35, color=f"C{k}", rasterized=True,
                   label=f"{s} (n={len(ref_v)}, R={pearsonr(est_v, ref_v)[0]:.2f})")
        top.fill_between(grid, gaussian_kde(ref_v)(grid), alpha=0.35, color=f"C{k}")
        right.fill_betweenx(grid, gaussian_kde(est_v)(grid), alpha=0.35, color=f"C{k}")
    ax.plot([0, 1], [0, 1], "k", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="conventional relative SASA (Stage 3)", ylabel="differentiable accessibility")
    ax.legend(loc="upper left", markerscale=4, frameon=False)
    top.set(xlim=(0, 1), title="Relative accessibility")
    right.set(ylim=(0, 1))
    for a in (top, right):
        a.axis("off")
    # Per-protein Pearson r: is the agreement consistent across proteins (each protein counts once)?
    per_protein = [df[df.split == s].pearson.values for s in splits]
    box.boxplot(per_protein, tick_labels=[f"{s}\n({len(v)} proteins)" for s, v in zip(splits, per_protein)], widths=0.5,
                showfliers=False)
    rng = np.random.default_rng(0)
    for k, v in enumerate(per_protein):
        box.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=f"C{k}", zorder=3)
    box.set(ylabel="per-protein Pearson r", title="Consistency across proteins")
    fig.savefig(OUT / "plots/accessibility_comparison.png", dpi=200, bbox_inches="tight")

    # Buried/exposed confusion matrices (pooled residues, row-normalised) per split
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    for a, s in zip(axes, splits):
        ref_v, est_v = pool[s][1] < BURIED, pool[s][0] < BURIED
        cm = np.array([[(~ref_v & ~est_v).sum(), (~ref_v & est_v).sum()], [(ref_v & ~est_v).sum(), (ref_v & est_v).sum()]])
        im = a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            a.text(c, r, f"{v}\n({v / cm[r].sum():.0%})", ha="center", va="center", color="white" if r == c else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["exposed", "buried"], yticklabels=["exposed", "buried"],
              xlabel="predicted (differentiable)",
              title=f"{s}\naccuracy {np.trace(cm) / cm.sum():.2f}, balanced {np.mean(np.diag(cm) / cm.sum(1)):.2f}")
    axes[0].set_ylabel("conventional SASA")
    fig.colorbar(im, ax=axes, fraction=0.025, label="fraction of true class")
    fig.savefig(OUT / "plots/buried_exposed_accuracy.png", dpi=200, bbox_inches="tight")
