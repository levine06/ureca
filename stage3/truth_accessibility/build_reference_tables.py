"""Rebuild the bundled Gly-X-Gly maximum-ASA tables (Tien et al. 2013 method; see tripeptide.py).

    python build_reference_tables.py [--radii 1.4 2.5 4.0 6.0] [--clash-scale 0.80] [--workers 2]
                                             [--step 10] [--out src/accessfold/data/references] [--suffix ""]
"""
import argparse
import time
from pathlib import Path

from accessfold.accessibility.tripeptide import TripeptideConfig, build_tripeptide_references

ap = argparse.ArgumentParser()
ap.add_argument("--radii", type=float, nargs="+", default=[1.4, 2.5, 4.0, 6.0])
ap.add_argument("--clash-scale", type=float, default=0.80)
ap.add_argument("--step", type=float, default=10.0)
ap.add_argument("--n-points", type=int, default=1000)
ap.add_argument("--n-points-refine", type=int, default=10000)
ap.add_argument("--top-m", type=int, default=24)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--out", default="src/accessfold/data/references")
ap.add_argument("--suffix", default="")
a = ap.parse_args()

cfg = TripeptideConfig(phi_psi_step=a.step, clash_scale=a.clash_scale, n_points=a.n_points,
                       n_points_refine=a.n_points_refine, top_m=a.top_m)
t0 = time.time()
tables = build_tripeptide_references(
    a.radii, config=cfg, n_workers=a.workers,
    progress=lambda sc: print(f"{sc.resname}: {sc.n_conformers} conformers, {sc.n_clash_rejected} rejected, "
                              f"{sc.seconds:.0f}s, A_max@{min(sc.max_total)}={sc.max_total[min(sc.max_total)]:.1f}", flush=True))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for r, table in tables.items():
    path = out / f"gxg_staggered_probe{r:g}{a.suffix}.json"
    table.to_json(path)
    print("wrote", path)
print(f"done in {time.time() - t0:.0f}s")
