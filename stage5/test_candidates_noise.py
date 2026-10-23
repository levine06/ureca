"""Does the differentiable estimate keep the Stage 3 accessibility signal on OpenFold candidates and under coordinate noise?

Development + validation proteins only (the held-out test splits are not read).
  candidate test: per protein, Spearman rho between accessibility error and C-alpha RMSD over its 25 candidates,
                  for the differentiable estimate and for accessfold (probe 1.4 A), plus a shuffled-reference control.
  noise test:     Gaussian noise added to native C-alphas; agreement with the clean estimate and reference, and
                  cosine similarity of the loss gradient with the clean gradient.
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

from differentiable_accessibility import accessibility, accessibility_loss
from validate_accessibility import OUT, ROOT, TRUTH, load  # also puts accessfold on sys.path
from accessfold.structures.mmcif import load_chain_from_mmcif  # noqa: E402
from accessfold.structures.predicted import load_predicted_chain  # noqa: E402

CAND = ROOT / "stage3/candidate_generation/candidates"
RUNS = ROOT / "stage3/scoring/runs/dev_val"
SCORES = pd.read_csv(ROOT / "stage3/scoring/results/development_validation/candidate_scores.csv", usecols=["entry", "candidate", "rank"])
MANIFEST = pd.read_csv(ROOT / "stage3/candidate_generation/candidates_manifest.csv")
SIGMAS, DRAWS, N_SHUFFLE = [0, 0.5, 1, 2, 3, 5, 8], 5, 20
torch.set_num_threads(1)


def ca_by_position(structure, n):
    """C-alpha coordinates indexed by sequence position (NaN where absent)."""
    ca = structure.atom_names == "CA"
    xyz = np.full((n, 3), np.nan)
    xyz[structure.residue_index[ca]] = structure.coords[ca]
    return xyz


def estimate(xyz):
    """Differentiable accessibility per sequence position (NaN where the C-alpha is absent)."""
    present = ~np.isnan(xyz[:, 0])
    out = np.full(len(xyz), np.nan)
    with torch.no_grad():
        out[present] = accessibility(torch.tensor(xyz[present], dtype=torch.float32)).numpy()
    return out


def load_candidate(args):
    entry, path, chain, seq = args
    s, _, _ = load_predicted_chain(CAND / path, chain, seq)
    return ca_by_position(s, len(seq))


def pct(rmsd, k):
    """Structural rank percentile of candidate k within its protein (0 = best RMSD, 1 = worst)."""
    return (rankdata(rmsd)[k] - 1) / (len(rmsd) - 1)


def mae(a, b):
    return np.abs(a - b).mean()


def corr_err(a, b):
    return 1 - pearsonr(a, b)[0]


def candidate_test(entry, split, pool, rng):
    run = np.load(RUNS / f"{entry}.npz", allow_pickle=True)
    truth_meta = json.loads(str(np.load(TRUTH / "results" / f"{entry}.npz")["meta"]))
    paths = json.loads(str(run["meta"]))["candidates"]
    seq = truth_meta["sequence"]
    chain = MANIFEST.loc[MANIFEST.query_name == paths[0].split("/")[1], "chain_id"].iloc[0]
    sc = SCORES[SCORES.entry == entry].set_index("candidate").loc[paths]
    rmsd, rank = run["rmsd_ca"], sc["rank"].to_numpy()
    mask, ref_af = run["mask"], run["truth_rel_1.4"]
    ref = np.clip(ref_af, 0, 1)
    af = run["rel_1.4"]  # accessfold candidate accessibility, (K, N)

    coords = pool.map(load_candidate, [(entry, p, chain, seq) for p in paths])
    est = np.stack([estimate(c) for c in coords])  # (K, N)
    native, _, _ = load_chain_from_mmcif(TRUTH / "cif_cache" / f"{truth_meta['pdb_id']}.cif.gz", truth_meta["label_chain_id"], seq)
    est_native = estimate(ca_by_position(native, len(seq)))

    m = mask & ~np.isnan(est).any(0)
    err = {
        "diff_ref_mae": [mae(e[m], ref[m]) for e in est],
        "diff_ref_corr": [corr_err(e[m], ref[m]) for e in est],
        "af_ref_mae": [mae(a[m], ref_af[m]) for a in af],
        "af_ref_corr": [corr_err(a[m], ref_af[m]) for a in af],
        "diff_native_mae": [mae(e[m], est_native[m]) for e in est],
    }
    row = dict(entry=entry, split=split, n_candidates=len(paths), rmsd_min=rmsd.min(), rmsd_max=rmsd.max())
    for k, v in err.items():
        row[f"rho_{k}"] = spearmanr(v, rmsd)[0]
    shuf = []
    for _ in range(N_SHUFFLE):  # control: the same error against a residue-permuted reference
        perm = rng.permutation(np.where(m)[0])
        shuf.append(spearmanr([mae(e[m], ref[perm]) for e in est], rmsd)[0])
    row["rho_shuffled_mean"] = np.nanmean(shuf)
    row["agree_pearson_mean"] = np.mean([pearsonr(e[m], a[m])[0] for e, a in zip(est, af)])  # estimate vs accessfold, same candidate
    row["pct_openfold_top1"] = pct(rmsd, int(np.argmin(rank)))
    row["pct_select_diff"] = pct(rmsd, int(np.argmin(err["diff_ref_mae"])))
    row["pct_select_af"] = pct(rmsd, int(np.argmin(err["af_ref_mae"])))
    return row


def noise_test(entries, rng):
    rows = []
    for entry in entries:
        x, ref, valid = load(entry)
        v = torch.tensor(valid)
        clean = x.clone().requires_grad_()
        accessibility_loss_valid = lambda c: ((accessibility(c) - ref)[v] ** 2).mean()  # noqa: E731
        accessibility_loss_valid(clean).backward()
        g0 = clean.grad.flatten()
        a0 = accessibility(x).detach().numpy()[valid]
        for sigma in SIGMAS:
            for _ in range(DRAWS):
                noisy = (x + sigma * torch.randn_like(x)).requires_grad_()
                accessibility_loss_valid(noisy).backward()
                a = accessibility(noisy).detach().numpy()[valid]
                rows.append(dict(entry=entry, sigma=sigma, rmsd_equiv=sigma * 3 ** 0.5,
                                 pearson_ref=pearsonr(a, ref.numpy()[valid])[0] if sigma else pearsonr(a0, ref.numpy()[valid])[0],
                                 pearson_clean=pearsonr(a, a0)[0] if sigma else 1.0,
                                 grad_cosine=float(torch.nn.functional.cosine_similarity(noisy.grad.flatten(), g0, dim=0)) if sigma else 1.0))
    return pd.DataFrame(rows)


def summarize(df, col, b=2000, seed=0):
    """Median over proteins with a 95% bootstrap interval, and the number of proteins with rho > 0."""
    v = df[col].dropna().to_numpy()
    r = np.random.default_rng(seed)
    meds = [np.median(r.choice(v, len(v))) for _ in range(b)]
    return f"{np.median(v):+.2f} [{np.percentile(meds, 2.5):+.2f}, {np.percentile(meds, 97.5):+.2f}]  >0 in {(v > 0).sum()}/{len(v)}"


if __name__ == "__main__":
    summary = pd.read_csv(TRUTH / "results/summary.csv")[["entry", "split"]]
    rng = np.random.default_rng(0)
    with Pool(16) as pool:
        cand = pd.DataFrame([candidate_test(e, s, pool, rng) for e, s in zip(summary.entry, summary.split)])
    cand.round(4).to_csv(OUT / "results/candidate_test.csv", index=False)
    print(f"candidates: {len(cand)} proteins")
    print("per-protein Spearman rho (accessibility error vs C-alpha RMSD), median [95% CI]")
    for col in ["rho_af_ref_mae", "rho_diff_ref_mae", "rho_af_ref_corr", "rho_diff_ref_corr", "rho_diff_native_mae", "rho_shuffled_mean"]:
        print(f"  {col:22s} {summarize(cand, col)}")
    print(f"  estimate vs accessfold on candidates: mean Pearson {cand.agree_pearson_mean.mean():.3f}")
    print("median structural rank percentile of selected candidate (0 best, 0.5 chance):")
    for col in ["pct_openfold_top1", "pct_select_af", "pct_select_diff"]:
        print(f"  {col:20s} {cand[col].median():.3f}   mean {cand[col].mean():.3f}")

    noise = noise_test(summary.entry, rng)
    tab = noise.groupby("sigma")[["rmsd_equiv", "pearson_ref", "pearson_clean", "grad_cosine"]].mean().round(3)
    tab.to_csv(OUT / "results/noise_test.csv")
    print(tab)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    cols = [c for c in ["rho_af_ref_mae", "rho_diff_ref_mae", "rho_diff_native_mae", "rho_shuffled_mean"]]
    labels = ["accessfold\nvs reference", "differentiable\nvs reference", "differentiable\nvs its own native", "differentiable,\nshuffled reference"]
    ax[0].boxplot([cand[c].dropna() for c in cols], tick_labels=labels, showfliers=False, widths=0.5)
    for k, c in enumerate(cols):
        ax[0].scatter(k + 1 + np.random.default_rng(1).uniform(-0.12, 0.12, cand[c].notna().sum()), cand[c].dropna(), s=10, zorder=3)
    ax[0].axhline(0, c="k", lw=0.8)
    ax[0].set(ylabel="per-protein Spearman rho (error vs RMSD)", title="A. Does lower accessibility error mean lower RMSD?")
    ax[0].tick_params(axis="x", labelsize=8)
    ax[1].boxplot([cand[c] for c in ["pct_openfold_top1", "pct_select_af", "pct_select_diff"]],
                  tick_labels=["OpenFold\ntop-ranked", "lowest accessfold\nerror", "lowest differentiable\nerror"], showfliers=False, widths=0.5)
    ax[1].axhline(0.5, ls="--", c="k", lw=1)
    ax[1].set(ylabel="structural rank percentile of selected candidate\n(0 = best RMSD)", title="B. Candidate selection (dashed = chance)")
    ax[1].tick_params(axis="x", labelsize=8)
    for c, lab in [("pearson_ref", "Pearson with reference RSA"), ("pearson_clean", "Pearson with clean estimate"), ("grad_cosine", "gradient cosine vs clean")]:
        g = noise.groupby("rmsd_equiv")[c]
        ax[2].errorbar(g.mean().index, g.mean(), g.std(), marker="o", capsize=3, label=lab)
    ax[2].set(xlabel="noise, equivalent RMSD (A) = sigma * sqrt(3)", ylabel="value (mean +/- sd over proteins)", title="C. Robustness to coordinate noise")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots/candidate_noise_test.png", dpi=200)
