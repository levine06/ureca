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
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for e, s in zip(summary.entry, summary.split):
        if s != "validation":
            continue
        ax[0].scatter(est[e][1], est[e][0], s=3, alpha=0.3, color="C0")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="conventional relative SASA", ylabel="differentiable accessibility", title="Validation residues (all proteins)")
    ax[1].boxplot([df[df.split == s].pearson for s in ("development", "validation")], tick_labels=["development", "validation"])
    ax[1].set(ylabel="per-protein Pearson r", title="Agreement per protein")
    fig.tight_layout()
    fig.savefig(OUT / "plots/accessibility_comparison.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    for k, col in enumerate(["accuracy", "balanced_accuracy"]):
        ax.bar(np.arange(2) + (k - 0.5) * 0.4, [df[df.split == s][col].mean() for s in ("development", "validation")], 0.4,
               label=col.replace("_", " "), color=f"C{k}")
    ax.set(xticks=[0, 1], xticklabels=["development", "validation"], ylabel="mean over proteins", ylim=(0, 1),
           title="Buried/exposed (rel. SASA < 0.2)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "plots/buried_exposed_accuracy.png", dpi=150)
