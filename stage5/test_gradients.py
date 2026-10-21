"""Sanity checks and the buried-residue gradient test."""
import numpy as np
import pandas as pd
import torch

from differentiable_accessibility import PARAMS, accessibility
from validate_accessibility import OUT, load

ENTRY, TARGET, LR, STEPS = "1AYD_A", 0.65, 5.0, 300

x, ref, valid = load(ENTRY)

# 1. shape, finite gradients, rotation/translation invariance
c = x.clone().requires_grad_()
a = accessibility(c)
((a - ref) ** 2).mean().backward()
assert a.shape == (len(x),) and c.grad.shape == x.shape and torch.isfinite(c.grad).all()
q, _ = torch.linalg.qr(torch.randn(3, 3))
a2 = accessibility(x @ q.T + torch.randn(3) * 10)
print(f"max |a - a_transformed| = {(a.detach() - a2).abs().max():.2e}")
assert torch.allclose(a.detach(), a2, atol=1e-4)

# 2. gradient test: push the most buried residue (lowest estimate among truly buried ones) towards TARGET
i = int(torch.where(torch.tensor(valid) & (ref < 0.2))[0][a.detach()[torch.tensor(valid) & (ref < 0.2)].argmin()])
c = x.clone().requires_grad_()
opt = torch.optim.SGD([c], lr=LR)
rows = []
for step in range(STEPS + 1):
    est = accessibility(c)
    rows.append(dict(step=step, target_accessibility=float(est[i]), mean_accessibility=float(est.mean()),
                     rmsd_from_start=float(((c - x) ** 2).sum(-1).mean().sqrt())))
    opt.zero_grad()
    ((est[i] - TARGET) ** 2).backward()
    opt.step()
df = pd.DataFrame(rows)
df.round(4).to_csv(OUT / "results/gradient_test.csv", index=False)
print(f"residue {i + 1}: {df.target_accessibility.iloc[0]:.3f} -> {df.target_accessibility.iloc[-1]:.3f} (target {TARGET})")
assert df.target_accessibility.iloc[-1] > df.target_accessibility.iloc[0]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
disp = (c.detach() - x).norm(dim=-1).numpy()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(df.step, df.target_accessibility)
ax[0].axhline(TARGET, ls="--", c="k", lw=1)
ax[0].set(xlabel="gradient step", ylabel=f"estimated accessibility, residue {i + 1}", title="Buried residue becomes exposed")
ax[1].bar(range(1, len(x) + 1), disp)
ax[1].axvline(i + 1, c="r", lw=1)
ax[1].set(xlabel="residue", ylabel="C-alpha displacement (A)", title="Coordinate change (red = target)")
from sklearn.decomposition import PCA
p0 = PCA(2).fit(x.numpy())
a0, a1 = p0.transform(x.numpy()), p0.transform(c.detach().numpy())
ax[2].scatter(*a0.T, s=8, c="gray", label="before")
ax[2].scatter(*a1.T, s=8, c="C3", label="after")
ax[2].annotate("", a1[i], a0[i], arrowprops=dict(arrowstyle="->", lw=2))
ax[2].scatter(*a0[i], s=120, marker="*", c="k", label=f"residue {i + 1} (arrow: move)")
ax[2].set(title="C-alpha trace (PCA projection)", xlabel="PC1 (A)", ylabel="PC2 (A)")
ax[2].legend()
cen = x.mean(0)
print(f"neighbours within r0: {int((torch.cdist(x[i:i+1], x) < PARAMS['r0']).sum()) - 1} -> {int((torch.cdist(c.detach()[i:i+1], c.detach()) < PARAMS['r0']).sum()) - 1}; "
      f"distance from centroid: {float((x[i] - cen).norm()):.1f} -> {float((c.detach()[i] - c.detach().mean(0)).norm()):.1f} A")
fig.tight_layout()
fig.savefig(OUT / "plots/gradient_test.png", dpi=150)
