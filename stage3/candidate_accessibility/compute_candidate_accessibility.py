#!/usr/bin/env python
"""Stage 3, step 3: accessibility of every OpenFold3 candidate structure.

Reads ``stage3/candidate_generation/candidates_manifest.csv`` and, for every candidate, computes the same
accessibility quantities as the truth pipeline (``stage3/truth_accessibility``): relative / absolute SASA, whole
residue and side chain only, at the probe radii and number of sampling points stored in the entry's truth ``.npz``.
Each value is kept only on residues that are in the saved truth ``mask`` (and valid in the candidate), so truth and
candidate are always compared on identical residues.

Output: one ``<entry>.npz`` per protein in ``--out``, with arrays stacked over its candidates (K = candidates):

    candidate_id [K], seed [K], sample [K], residue_names [N]
    mask [K, N]                      truth mask AND candidate-valid residues
    abs_<r>, rel_<r>, sc_abs_<r>, sc_rel_<r>   [K, N]   (NaN outside mask), r = probe radius formatted ``:g``
    meta                             JSON: settings, truth file used, package version

Examples
--------
    # one protein (test)
    python stage3/candidate_accessibility/compute_candidate_accessibility.py --only 1L66_A

    # everything, 4 worker processes
    python stage3/candidate_accessibility/compute_candidate_accessibility.py --workers 4

    # one shard of four (PBS array element)
    python stage3/candidate_accessibility/compute_candidate_accessibility.py --shard 0/4 --workers 4

Re-running is safe: entries whose ``.npz`` already exists are skipped (``--overwrite`` recomputes).
"""

from __future__ import annotations

import os

# one BLAS thread per worker process: many small processes are faster than many threads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

URECA_ROOT = Path(__file__).resolve().parents[2]
TRUTH_SRC = URECA_ROOT / "stage3" / "truth_accessibility" / "src"
DEFAULT_MANIFEST = URECA_ROOT / "stage3" / "candidate_generation" / "candidates_manifest.csv"
DEFAULT_TRUTH_DIR = URECA_ROOT / "stage3" / "truth_accessibility" / "results"
DEFAULT_OUT = URECA_ROOT / "stage3" / "candidate_accessibility" / "results"
CANDIDATE_CHAIN = "A"   # OpenFold3 writes the single predicted chain as label_asym_id A


def _import_accessfold():
    """Use the installed accessfold if there is one, otherwise the copy in stage3/truth_accessibility/src."""
    try:
        import accessfold  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(TRUTH_SRC))


def process_entry(task):
    """Runs in a worker process. Returns (entry, None) on success or (entry, error string)."""
    entry, rows, truth_dir, out_dir, overwrite = task
    target = Path(out_dir) / f"{entry}.npz"
    try:
        if target.exists() and not overwrite:
            return entry, None
        _import_accessfold()
        from accessfold.structures.mmcif import load_chain_from_mmcif
        from accessfold.truth import truth_arrays

        t0 = time.time()
        truth = np.load(Path(truth_dir) / f"{entry}.npz", allow_pickle=True)
        tmeta = json.loads(str(truth["meta"]))
        radii, n_points, sequence = tmeta["probe_radii"], tmeta["n_points"], tmeta["sequence"]
        truth_mask = truth["mask"]

        keys = [k for k in truth.files if k.split("_")[0] in ("abs", "rel", "sc")]
        stacked = {k: [] for k in keys}
        masks, ids, seeds, samples = [], [], [], []
        for row in rows:
            cif = URECA_ROOT / row["model_path"]
            structure, complete, _qc = load_chain_from_mmcif(cif, CANDIDATE_CHAIN, sequence)
            if structure.n_residues != len(truth_mask):
                raise ValueError(f"{row['candidate_id']}: {structure.n_residues} residues, truth has {len(truth_mask)}")
            if not np.array_equal(structure.residue_names.astype("<U3"), truth["residue_names"]):
                raise ValueError(f"{row['candidate_id']}: residue names differ from the truth entry")
            arrays = truth_arrays(structure, complete, radii, n_points)   # no `modified`: candidates have none
            mask = truth_mask & arrays["mask"]
            for k in keys:
                stacked[k].append(np.where(mask, arrays[k], np.nan))
            masks.append(mask)
            ids.append(row["candidate_id"])
            seeds.append(int(row["seed"]))
            samples.append(int(row["sample"]))

        try:
            from importlib.metadata import version
            pkg_version = version("accessfold")
        except Exception:  # noqa: BLE001
            pkg_version = "unknown"
        meta = {
            "entry": entry, "n_candidates": len(ids), "probe_radii": radii, "n_points": n_points,
            "radii_table": tmeta.get("radii_table"), "reference": tmeta.get("reference"),
            "truth_file": f"{entry}.npz", "mask": "truth mask AND candidate-valid residues",
            "sequence": sequence, "seconds": round(time.time() - t0, 2), "package_version": pkg_version,
        }
        out = {k: np.stack(v) for k, v in stacked.items()}
        tmp = Path(out_dir) / f".{entry}.tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), candidate_id=np.array(ids), seed=np.array(seeds),
                            sample=np.array(samples), residue_names=truth["residue_names"], mask=np.stack(masks), **out)
        tmp.replace(target)
        return entry, None
    except Exception as exc:  # noqa: BLE001 - a bad entry must not kill the batch
        return entry, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--truth-dir", default=str(DEFAULT_TRUTH_DIR), help="folder with the truth <entry>.npz files")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=1, help="processes; must not exceed allocated CPUs")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th entry starting at i")
    ap.add_argument("--only", nargs="+", help="restrict to these entries, e.g. 1L66_A (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(args.manifest)
    entries = sorted(manifest["query_name"].unique())
    if args.only:
        entries = [e for e in entries if e in set(args.only)]
    i, n = (int(x) for x in args.shard.split("/"))
    entries = entries[i::n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(e, manifest[manifest["query_name"] == e].sort_values(["seed", "sample"]).to_dict("records"),
              args.truth_dir, str(out_dir), args.overwrite) for e in entries]
    print(f"{len(tasks)} entries, {sum(len(t[1]) for t in tasks)} candidates -> {out_dir}", flush=True)

    failures = []
    t0 = time.time()
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(process_entry, tasks)
    else:   # no pool for a single worker: simpler and easier to debug
        pool, results = None, map(process_entry, tasks)
    for k, (entry, err) in enumerate(results, 1):
        print(f"[{k}/{len(tasks)}] {entry}: {'ok' if err is None else 'FAILED ' + err}", flush=True)
        if err:
            failures.append((entry, err))
    if pool:
        pool.shutdown()
    print(f"done in {time.time() - t0:.0f}s, {len(failures)} failed")
    if failures:
        pd.DataFrame(failures, columns=["entry", "error"]).to_csv(out_dir / "failures.csv", index=False)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
