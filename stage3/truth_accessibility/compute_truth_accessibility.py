#!/usr/bin/env python
"""Calculate true synthetic accessibility from experimental structures.

Input: development/validation CSVs and experimental PDB mmCIF structures.
Output: per-chain NPZ accessibility arrays, summary.csv, residues.csv.gz and
failure reports. Values include absolute, relative and side-chain accessibility
at the selected probe radii, with masks and quality-control metadata.

"True" means calculated from experimental coordinates, not measured in solution.
This script does not generate or score OpenFold3 predictions.

Run from this package folder:
    python -m pip install -e ".[biotite]"
    python compute_truth_accessibility.py --csv development.csv validation.csv \\
        --cache-dir cif_cache --out truth_out --download-only --workers 4
    python compute_truth_accessibility.py --csv development.csv validation.csv \\
        --cache-dir cif_cache --out truth_out --offline --workers 4

On PBS, workers must not exceed allocated CPUs. --shard i/n supports batch
partitioning; --collect-only assembles saved outputs after all shards finish.
Existing NPZs are skipped: use a separate output directory or --overwrite when
changing inputs or settings.
"""

from __future__ import annotations

import os

# one BLAS thread per worker process: the SASA kernel uses small matmuls, many processes are faster than many threads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

def raise_csv_field_limit():
    """Long sequences exceed csv's default field limit; sys.maxsize overflows a C long on Windows, so back off."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


raise_csv_field_limit()

DEFAULT_URLS = (
    "https://files.rcsb.org/download/{ID}.cif.gz",
    "https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/{id}.cif.gz",
)


# --------------------------------------------------------------------------------------------- input
def read_rows(csv_paths):
    rows, seen = [], {}
    for p in csv_paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["pdb_id"].strip().upper(), r["label_chain_id"].strip())
                if key in seen:
                    print(f"WARNING duplicate entry {key} in {p} (first seen in {seen[key]}); keeping the first", file=sys.stderr)
                    continue
                seen[key] = Path(p).name
                r["_key"] = key
                r["_source_csv"] = Path(p).name
                rows.append(r)
    return rows


def entry_name(key):
    return f"{key[0]}_{key[1]}"


# ----------------------------------------------------------------------------------------- download
def fetch_cif(pdb_id: str, cache_dir: Path, urls=DEFAULT_URLS, retries: int = 4) -> Path:
    dest = cache_dir / f"{pdb_id.upper()}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    last = None
    for template in urls:
        url = template.format(ID=pdb_id.upper(), id=pdb_id.lower())
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "accessfold-truth/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                gzip.decompress(data)  # integrity check: raises on a truncated/garbled download
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return dest
            except urllib.error.HTTPError as exc:
                last = f"{url}: HTTP {exc.code}"
                if exc.code in (404, 403):  # will not fix itself: try the next mirror
                    break
            except Exception as exc:  # network hiccup, truncated gzip, ...
                last = f"{url}: {type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {pdb_id}: {last}")


# ------------------------------------------------------------------------------------------ compute
def process_entry(task):
    """Runs in a worker process. Returns (entry, None) on success or (entry, error string)."""
    row, cache_dir, out_dir, radii, n_points, gap_flank, offline, download_only, overwrite, shadow_threshold = task
    key = row["_key"]
    name = entry_name(key)
    target = Path(out_dir) / f"{name}.npz"
    try:
        if target.exists() and not overwrite and not download_only:
            return name, None
        cache = Path(cache_dir)
        cif = cache / f"{key[0]}.cif.gz"
        if not cif.exists() or cif.stat().st_size == 0:
            if offline:
                raise FileNotFoundError(f"{cif} missing and --offline given")
            cif = fetch_cif(key[0], cache)
        if download_only:
            return name, None

        from accessfold.structures.mmcif import load_chain_from_mmcif
        from accessfold.truth import truth_arrays

        t0 = time.time()
        _h = hashlib.sha1()
        with gzip.open(cif, "rb") as _fh:
            for _block in iter(lambda: _fh.read(1 << 20), b""):
                _h.update(_block)
        cif_sha1 = _h.hexdigest()
        structure, complete, qc = load_chain_from_mmcif(cif, key[1], row["sequence"].strip())
        # True synthetic accessibility from experimental coordinates, with validity masks.
        arrays = truth_arrays(structure, complete, radii, n_points, modified=qc["modified_residues"],
                              shadow_threshold=shadow_threshold)
        try:  # Stage 1's coverage vs usable-coordinate coverage
            csv_frac = float(row.get("resolved_fraction", ""))
            if abs(csv_frac - qc["resolved_fraction"]) > 1e-3:
                qc["warnings"].append(f"Stage 1 resolved_fraction {csv_frac:.4f} differs from usable-coordinate coverage "
                                      f"{qc['resolved_fraction']:.4f}")
        except ValueError:
            pass
        mask = arrays["mask"].copy()
        if gap_flank > 0:
            mask &= arrays["gap_distance"] > gap_flank
            for k in list(arrays):
                if k.split("_")[0] in ("abs", "rel", "sc"):
                    arrays[k] = np.where(mask, arrays[k], np.nan)
            arrays["mask"] = mask
    except Exception as exc:  # noqa: BLE001 - a bad entry must not kill the batch
        return name, f"{type(exc).__name__}: {exc}"

    try:
        from importlib.metadata import version
        pkg_version = version("accessfold")
    except Exception:  # noqa: BLE001
        pkg_version = "unknown"
    meta = {
        "entry": name, "pdb_id": key[0], "label_chain_id": key[1], "source_csv": row["_source_csv"],
        "split": row.get("training_or_test_split", ""), "release_date": row.get("release_date", ""),
        "method": row.get("experimental_method", ""), "resolution": row.get("resolution", ""),
        "csv_resolved_fraction": row.get("resolved_fraction", ""), "csv_length": row.get("length", ""),
        "number_of_chains_in_assembly": row.get("number_of_chains", ""), "screening_decision": row.get("screening_decision", ""),
        "passes_standard_filters": row.get("passes_standard_filters", ""), "forced_split": row.get("forced_split", ""),
        "probe_radii": list(radii), "n_points": n_points, "radii_table": "bondi", "gap_flank": gap_flank,
        "mask_incomplete_residues": True, "shadow_threshold": shadow_threshold,
        "masking_policy": "unresolved | incomplete | modified/mismatching (selenomethionine excepted) | shadow of modification",
        "reference": "gxg_staggered_chi_grid_v1 (clash_scale 0.80)",
        "sequence": row["sequence"].strip(), "seconds": round(time.time() - t0, 2), "qc": qc,
        "package_version": pkg_version,
        # hash of the DECOMPRESSED structure the arrays were computed from; score_candidates.py refuses to score against a
        # cache whose content has changed since (a re-download with a different gzip header hashes the same)
        "source_cif_name": cif.name, "source_cif_sha1": cif_sha1,
    }
    tmp = Path(out_dir) / f".{name}.tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(target)
    return name, None


# ------------------------------------------------------------------------------------------ collect
def _differs(csv_value, usable) -> bool:
    try:
        return abs(float(csv_value) - float(usable)) > 1e-3
    except ValueError:
        return False


def collect(out_dir: Path, radii):
    files = sorted(p for p in out_dir.glob("*.npz") if not p.name.startswith("."))
    summary_rows, long_rows = [], []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            names, mask, resolved = z["residue_names"], z["mask"], z["resolved"]
            gap, complete = z["gap_distance"], z["complete"]
            modified = z["modified"] if "modified" in z.files else np.zeros_like(mask)
            shadow = z["shadow"] if "shadow" in z.files else np.zeros_like(mask)
            rel = {r: z[f"rel_{r:g}"] for r in radii}
            ab = {r: z[f"abs_{r:g}"] for r in radii}
            sc_ab = {r: z[f"sc_abs_{r:g}"] for r in radii}
            sc_rel = {r: z[f"sc_rel_{r:g}"] for r in radii}
        qc = meta["qc"]
        row = {k: meta[k] for k in ("entry", "pdb_id", "label_chain_id", "split", "release_date", "method", "resolution",
                                    "number_of_chains_in_assembly", "screening_decision", "passes_standard_filters", "forced_split")}
        row.update({
            "n_sequence": qc["n_sequence"],
            "n_with_coordinate_records": qc.get("n_with_coordinate_records", ""), "n_usable": qc["n_resolved"],
            "records_fraction": round(qc.get("records_fraction", float("nan")), 5),
            "usable_fraction": round(qc["resolved_fraction"], 5),
            "csv_resolved_fraction": meta["csv_resolved_fraction"],
            "coverage_differs_from_stage1": _differs(meta["csv_resolved_fraction"], qc["resolved_fraction"]),
            "usable_ge_0.90": qc["resolved_fraction"] >= 0.90,
            "n_masked_out": int((~mask).sum()), "n_incomplete_residues": qc["n_incomplete_residues"],
            "n_modified_masked": int(modified.sum()), "n_shadow_masked": int(shadow.sum()),
            "modified_residues": "; ".join(f"{m['position']}:{m['comp_id']}" + (f"(parent {m['parent']})" if m.get("parent") else "")
                                           for m in qc.get("modified_residues", [])),
            "n_selenomethionine": qc.get("n_selenomethionine", 0), "longest_internal_gap": qc["longest_internal_gap"],
            "n_residues_with_altloc": qc["n_residues_with_altloc"], "n_residues_altloc_tie": qc.get("n_residues_altloc_tie", ""),
            "n_models_in_file": qc["n_models_in_file"], "warnings": "; ".join(qc["warnings"]),
        })
        for r in radii:
            v = rel[r][mask]
            row[f"mean_rsa_{r:g}"] = round(float(np.nanmean(v)), 4) if v.size else ""
            row[f"frac_rsa_gt1_{r:g}"] = round(float(np.mean(v > 1.0)), 4) if v.size else ""
            row[f"frac_buried_rsa_lt0.2_{r:g}"] = round(float(np.mean(v < 0.2)), 4) if v.size else ""
        summary_rows.append(row)
        for i in range(len(names)):
            long_rows.append([meta["entry"], meta["split"], i + 1, names[i], int(resolved[i]), int(complete[i]), int(modified[i]),
                              int(shadow[i]), int(mask[i]), int(gap[i]), *[("" if np.isnan(ab[r][i]) else f"{ab[r][i]:.3f}") for r in radii],
                              *[("" if np.isnan(rel[r][i]) else f"{rel[r][i]:.4f}") for r in radii],
                              *[("" if np.isnan(sc_ab[r][i]) else f"{sc_ab[r][i]:.3f}") for r in radii],
                              *[("" if np.isnan(sc_rel[r][i]) else f"{sc_rel[r][i]:.4f}") for r in radii]])
    if summary_rows:
        with open(out_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)
        with gzip.open(out_dir / "residues.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["entry", "split", "position", "residue", "resolved", "complete", "modified", "shadow", "in_mask", "gap_distance",
                        *[f"abs_{r:g}" for r in radii], *[f"rel_{r:g}" for r in radii],
                        *[f"sc_abs_{r:g}" for r in radii], *[f"sc_rel_{r:g}" for r in radii]])
            w.writerows(long_rows)
    return len(summary_rows)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="Stage 1 dataset CSV file(s)")
    ap.add_argument("--out", default="truth_out")
    ap.add_argument("--cache-dir", default="cif_cache")
    ap.add_argument("--radii", type=float, nargs="+", default=[1.4, 2.5, 4.0, 6.0])
    ap.add_argument("--n-points", type=int, default=1000, help="must equal the value used for candidate structures")
    ap.add_argument("--gap-flank", type=int, default=0, help="also mask this many residues next to INTERNAL gaps "
                    "(gap_distance is saved, so this can also be applied later)")
    ap.add_argument("--shadow-threshold", type=float, default=5.0, help="Angstrom^2: mask residues whose area changes by "
                    "more than this when a chemical modification's extra atoms are removed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th entry starting at i (SLURM array)")
    ap.add_argument("--offline", action="store_true", help="never touch the network; require files in --cache-dir")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", nargs="+", help="restrict to these PDB ids (testing)")
    ap.add_argument("--limit", type=int, help="process only the first N entries (testing)")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.collect_only:
        n = collect(out_dir, a.radii)
        print(f"collected {n} entries -> {out_dir/'summary.csv'}, {out_dir/'residues.csv.gz'}")
        return 0

    rows = read_rows(a.csv)
    if a.only:
        wanted = {x.upper() for x in a.only}
        rows = [r for r in rows if r["_key"][0] in wanted]
    if a.limit:
        rows = rows[: a.limit]
    i, n = (int(x) for x in a.shard.split("/"))
    rows = rows[i::n]
    print(f"{len(rows)} entries (shard {i}/{n}); workers={a.workers}; radii={a.radii}; n_points={a.n_points}", flush=True)

    tasks = [(r, str(a.cache_dir), str(out_dir), tuple(a.radii), a.n_points, a.gap_flank, a.offline, a.download_only, a.overwrite, a.shadow_threshold)
             for r in rows]
    failures = []
    t0 = time.time()
    if a.workers <= 1:
        results = (process_entry(t) for t in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=a.workers)
        results = (f.result() for f in as_completed([pool.submit(process_entry, t) for t in tasks]))
    for done, (name, err) in enumerate(results, 1):
        if err:
            failures.append((name, err))
            print(f"[{done}/{len(tasks)}] FAILED {name}: {err}", flush=True)
        elif done % 10 == 0 or done == len(tasks):
            print(f"[{done}/{len(tasks)}] ok  ({time.time() - t0:.0f}s)", flush=True)
    if a.workers > 1:
        pool.shutdown()

    if failures:
        fp = out_dir / (f"failures_shard{i}of{n}.csv" if n > 1 else "failures.csv")
        with open(fp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["entry", "error"])
            w.writerows(failures)
        print(f"{len(failures)} failure(s) written to {fp}")
    if not a.download_only and n == 1:
        print(f"collected {collect(out_dir, a.radii)} entries")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
