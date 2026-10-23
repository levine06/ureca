#!/usr/bin/env python
"""Stage 6, step 1: compute and log the accessibility loss during OpenFold3 diffusion WITHOUT changing the structures.

A forward hook on OpenFold3's DiffusionModule sees, at every diffusion step and for every sample, the noisy input x_t and the
denoised estimate x0 (the module's output). We take the C-alpha of each, compute the Stage 5 loss

    L_access = 1/N sum_i (a_hat_i - y_i)^2      a_hat = stage5 accessibility(C-alpha),  y = Stage 3 reference relative SASA

and append it to a CSV. The hook returns None, so the output is unchanged and the sampling loop is not touched or copied. This only
checks that the Stage 5 function works inside OpenFold3 (shapes, residue mapping, loss trend over the steps).

    python stage6/log_access_loss.py --entry 1L66_A --seeds 42 --num-diffusion-samples 5 --out stage6/runs/step1

Writes <out>/access_loss_<entry>.csv: seed, step, sigma, sample, loss_x0, loss_xt. Run in the openfold3 environment (CPU, cpu_inference.yml).

Step 2 (`--gradient`): in the final 30% of the steps, also compute the gradient of L_access with respect to the C-alpha of the denoised
estimate x0 and test it (still nothing is applied to the structures):
    * the gradient is (N, 3), finite, and zero for residues that are not neighbours (they do not enter the loss);
    * a small move against the gradient lowers the loss, in line with the first-order prediction, and a random move of the same size does not;
    * the inference-mode status of the hook's inputs is recorded, to show how the gradient is obtained.
Written to <out>/gradient_check_<entry>.csv. `--no-hook` runs the same prediction without any hook, as the control.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage5"))
from differentiable_accessibility import accessibility  # noqa: E402

TRUTH = ROOT / "stage3" / "truth_accessibility"
MSA_GLOB = str(ROOT / "stage3/candidate_generation/candidates/*/msas/*/main/{sha}.npz")


def find_entry(name):
    for split, folder in (("development", "results"), ("validation", "results"), ("test_a", "results_test"), ("test_b", "results_test")):
        df = pd.read_csv(ROOT / "stage1/datasets" / f"{split}.csv")
        hit = df[(df.pdb_id + "_" + df.label_chain_id) == name]
        if len(hit):
            return hit.iloc[0], TRUTH / folder / f"{name}.npz"
    sys.exit(f"{name} not found in the Stage 1 datasets")


class AccessLossLogger:
    """Forward hook on DiffusionModule: records L_access of x_t and of the denoised estimate; changes nothing."""

    def __init__(self, ref, loss_mask, contrib_mask, path, gradient=False, n_steps=200, fraction=0.3, sequence=None):
        self.ref, self.lmask, self.cmask = ref, loss_mask, contrib_mask
        self.rows, self.path = [], path
        self.last_t, self.step, self.ca_idx = None, 0, None
        self.gradient, self.first_step, self.sequence = gradient, int(round((1 - fraction) * n_steps)), sequence
        self.grad_rows, self.info = [], {}

    def loss_t(self, ca):
        """Same loss as `loss`, as a tensor (so it can be differentiated)."""
        a = accessibility(ca[self.cmask])
        idx = torch.cumsum(self.cmask.long(), 0) - 1
        sel = self.lmask & self.cmask
        return ((a[idx[sel]] - self.ref[sel]) ** 2).mean()

    def check_gradient(self, ca, sample, seed, sigma):
        """Gradient of L_access with respect to the C-alpha of x0, and the small-perturbation tests. Nothing is returned to OpenFold3."""
        with torch.inference_mode(False), torch.enable_grad():   # the hook runs under Lightning's inference mode, where autograd is off
            x = ca.detach().clone().double().requires_grad_(True)
            loss = self.loss_t(x)
            (g,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            g = g.detach()
            rms = g.pow(2).sum(-1).mean().sqrt()
            row = dict(seed=seed, step=self.step, sample=sample, sigma=sigma, loss=float(loss), shape_ok=tuple(g.shape) == (len(self.ref), 3),
                       finite=bool(torch.isfinite(g).all()), grad_rms=float(rms), grad_max=float(g.norm(dim=-1).max()),
                       grad_nonneighbour_max=float(g[~self.cmask].abs().max()) if (~self.cmask).any() else 0.0,
                       grad_neighbour_zero_frac=float((g[self.cmask].norm(dim=-1) == 0).float().mean()))
            gen = torch.Generator().manual_seed(1000 * seed + self.step * 8 + sample)
            rand = torch.randn(g.shape, generator=gen, dtype=g.dtype)
            for eps in (0.01, 0.03, 0.1):                         # RMS move per residue, in Angstrom
                d = -eps * g / rms
                r = eps * rand / rand.pow(2).sum(-1).mean().sqrt()
                row[f"dL_{eps}"] = float(self.loss_t(x.detach() + d) - loss)                # move against the gradient
                row[f"dL_random_{eps}"] = float(self.loss_t(x.detach() + r) - loss)         # random move of the same size
                row[f"dL_predicted_{eps}"] = float((g * d).sum())                           # first-order prediction
        self.grad_rows.append(row)

    def loss(self, ca):
        a = accessibility(ca[self.cmask])
        idx = torch.cumsum(self.cmask.long(), 0) - 1
        sel = self.lmask & self.cmask
        return float(((a[idx[sel]] - self.ref[sel]) ** 2).mean())

    def ca_index(self, batch):
        """Atom index of each token's C-alpha. One C-alpha per token (single protein chain)."""
        n_atom = batch["atom_mask"].shape[-1]
        chars = batch["ref_atom_name_chars"].reshape(-1, n_atom, 4, 64)[0].argmax(-1) + 32
        names = ["".join(chr(int(c)) for c in row).strip() for row in chars]
        is_ca = torch.tensor([n == "CA" for n in names]) & batch["atom_mask"].reshape(-1, n_atom)[0].bool().cpu()
        tok = batch["atom_to_token_index"].reshape(-1, n_atom)[0].cpu()[is_ca]
        n_tok = int(batch["token_mask"].reshape(-1, batch["token_mask"].shape[-1])[0].sum())
        assert len(tok) == n_tok == len(self.ref), f"C-alpha/token/reference mismatch: {len(tok)}, {n_tok}, {len(self.ref)}"
        out = torch.empty(n_tok, dtype=torch.long)
        out[tok] = is_ca.nonzero().flatten()
        return out

    def __call__(self, module, args, kwargs, output):
        batch, xt, t = kwargs["batch"], kwargs["xl_noisy"], float(kwargs["t"])
        if self.last_t is None or t > self.last_t:        # sigma falls during a rollout, so a rise means a new rollout (new seed)
            self.step, self.ca_idx = 0, self.ca_index(batch)
        self.last_t = t
        seed = batch["seed"][0] if isinstance(batch.get("seed"), list) else -1
        if not self.info:
            self.info = dict(inference_mode=torch.is_inference_mode_enabled(), grad_mode=torch.is_grad_enabled(),
                             output_requires_grad=output.requires_grad, output_is_inference_tensor=output.is_inference(),
                             output_grad_fn=str(output.grad_fn), output_shape=tuple(output.shape), n_atom=int(batch["atom_mask"].shape[-1]),
                             n_residues=len(self.ref), n_resolved=int(self.cmask.sum()), n_scored=int((self.lmask & self.cmask).sum()))
            if self.sequence:
                self.info.update(self.check_mapping(batch))
        for s in range(output.shape[1]):
            self.rows.append(dict(seed=seed, step=self.step, sigma=t, sample=s,
                                  loss_x0=self.loss(output[0, s][self.ca_idx].detach()),
                                  loss_xt=self.loss(xt[0, s][self.ca_idx].detach())))
            if self.gradient and self.step >= self.first_step:
                self.check_gradient(output[0, s][self.ca_idx], s, seed, t)
        self.step += 1
        return None                                        # output untouched

    def check_mapping(self, batch):
        """Residue mapping: C-alpha token order, residue types against the Stage 1 sequence, reference lengths."""
        n_atom = batch["atom_mask"].shape[-1]
        tok = batch["atom_to_token_index"].reshape(-1, n_atom)[0].cpu()[self.ca_idx]
        restype = batch["restype"].reshape(-1, batch["restype"].shape[-2], batch["restype"].shape[-1])[0].argmax(-1).cpu()
        order = "ARNDCQEGHILKMFPSTWYV"                       # OpenFold3 / AF3 residue-type order (alphabetical by three-letter code)
        types_ok = all(order.index(c) == int(restype[i]) for i, c in enumerate(self.sequence))
        return dict(ca_token_order_ok=bool((tok == torch.arange(len(tok))).all()), residue_types_match_sequence=bool(types_ok),
                    reference_length_matches=len(self.ref) == len(self.sequence) == len(self.ca_idx))

    def save(self):
        for path, rows in ((self.path, self.rows), (self.path.with_name(self.path.name.replace("access_loss", "gradient_check")), self.grad_rows)):
            if rows:
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        (self.path.parent / "hook_info.json").write_text(json.dumps(self.info, indent=1, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry", required=True, help="<PDB>_<label chain>, e.g. 1L66_A")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--num-diffusion-samples", type=int, default=5)
    p.add_argument("--runner-yaml", default=str(ROOT / "cpu_inference.yml"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--gradient", action="store_true", help="step 2: gradient of L_access with respect to x0 over the final 30%% of steps, plus tests")
    p.add_argument("--no-hook", action="store_true", help="control: same run without the hook, to confirm the hook does not change the structures")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    row, truth = find_entry(a.entry)
    d = np.load(truth, allow_pickle=True)
    rel = d["rel_1.4"].astype(float)
    ref = torch.tensor(np.clip(np.nan_to_num(rel), 0, 1), dtype=torch.float32)
    lmask = torch.tensor(d["mask"].astype(bool) & ~np.isnan(rel))
    cmask = torch.tensor(d["resolved"].astype(bool))            # resolved C-alphas are the neighbours, as in the Stage 5 calibration
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    msa = glob.glob(MSA_GLOB.format(sha=hashlib.sha256(row.sequence.encode()).hexdigest()))
    if not msa:
        sys.exit("no cached Stage 3 MSA for this sequence")
    query = out / f"query_{a.entry}.json"
    query.write_text(json.dumps({"queries": {a.entry: {"chains": [{"molecule_type": "protein", "chain_ids": ["A"],
                                                                  "sequence": row.sequence, "main_msa_file_paths": [msa[0]]}]}}}))
    runner = yaml.safe_load(Path(a.runner_yaml).read_text())
    runner.setdefault("experiment_settings", {}).update(seeds=a.seeds, skip_existing=False, use_msa_server=False, use_templates=False)
    runner_path = out / "runner.yml"
    runner_path.write_text(yaml.safe_dump(runner, sort_keys=False))

    torch.set_num_threads(a.threads)
    logger = AccessLossLogger(ref, lmask, cmask, out / f"access_loss_{a.entry}.csv", gradient=a.gradient, sequence=row.sequence)
    from openfold3.core.model.structure.diffusion_module import DiffusionModule
    original_init = DiffusionModule.__init__

    def init_with_hook(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.register_forward_hook(logger, with_kwargs=True)

    if not a.no_hook:
        DiffusionModule.__init__ = init_with_hook                   # the hook is attached when the model is built; nothing else changes
    from openfold3.run_openfold import cli
    cli.main(args=["predict", "--query-json", str(query), "--output-dir", str(out / "structures"), "--runner-yaml", str(runner_path),
                   "--use-msa-server", "False", "--use-templates", "False",
                   "--num-diffusion-samples", str(a.num_diffusion_samples)], standalone_mode=False)
    if a.no_hook:
        return
    logger.save()
    df = pd.DataFrame(logger.rows)
    last = df.step.max()
    print(f"{len(df)} loss values; mean L_access of the denoised estimate: first step {df[df.step == 0].loss_x0.mean():.4f}, "
          f"last step {df[df.step == last].loss_x0.mean():.4f} -> {logger.path}")


if __name__ == "__main__":
    main()
