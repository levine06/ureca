#!/usr/bin/env python
"""Calculate candidate accessibility, accessibility error and structural accuracy.

Predicted accessibility: calculate accessibility for each candidate.
Accessibility error: compare with the experimental reference, using
        matched atoms and valid reference positions by default. Reports include
        MAE, RMSE and profile-correlation error at the saved probe radii.
Structural accuracy: compare with experimental coordinates:
        aligned C-alpha RMSD, lDDT-C-alpha and TM-score, plus pairwise RMSD.

Choose one mode:
    --accessibility-only  candidate accessibility and accessibility errors
    --structure-only      structural accuracy
    neither flag          both calculations (combined workflow)
The flags are mutually exclusive. This script never generates predictions.

Input for all modes: dataset CSVs, saved experimental-reference NPZs, their original CIF cache
and candidate CIF/PDB files. --ranking-csv is optional and only supplies model
ranking information; it does not measure structural accuracy.

    python stage3/scoring/score_candidates.py --csv development.csv validation.csv \\
        --truth-dir truth_out --cache-dir cif_cache \\
        --candidates "preds/*/{entry}/seed_*/*_model.cif*" \\
        --accessibility-only --out accessibility_out --workers 8 --verify-truth

Output: candidate_scores.csv, entry_summary.csv and one NPZ per entry.
Single-mode outputs omit skipped calculations. Accessibility NPZs save candidate
profiles as c[candidate,residue,radius], reference profiles as y, and identifiers
and radii in JSON meta. --kind selects rel (default), abs or sc_rel.
Use separate output folders for modes/runs. Combine matching entry/candidate
identifiers later for accessibility-versus-accuracy analysis.

Candidate correspondence and coverage are checked before scoring. Default
--min-coverage 1.0 requires reference C-alpha coverage; accessibility modes also
require complete heavy atoms at reference-mask positions. Failed candidates are
recorded, not promoted into the original top ranks. Top-N summaries require
valid ranking scores and all candidates relevant to that top N.

--verify-truth recomputes the reference accessibility (including on reuse when
not yet verified); it cannot be used with --structure-only. Input/settings/mode
fingerprints control reuse. Failed entry outputs are quarantined.
--shard i/n supports partitioning; --collect-only assembles saved outputs for
the selected mode. Exit status 1 reports entry failures or conflicting rankings.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCHEMA = 4                      # bump when the stored arrays or their meaning change (invalidates old results)
THRESHOLDS_A = (1.0, 2.0, 4.0)


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


def read_rows(csv_paths):
    rows, seen = [], set()
    for p in csv_paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["pdb_id"].strip().upper(), r["label_chain_id"].strip())
                if key in seen:
                    continue
                seen.add(key)
                r["_key"] = key
                rows.append(r)
    return rows


def entry_name(key):
    return f"{key[0]}_{key[1]}"


ALIASES = {
    "entry": ["entry", "entry_id", "pdb_entry", "target", "target_id", "name", "pdb_id", "pdb"],
    "candidate": ["candidate", "candidate_file", "file", "filename", "file_name", "path", "structure", "structure_path",
                  "model", "model_path", "cif", "pdb_file"],
    "score": ["score", "ranking_score", "rank_score", "sample_ranking_score", "confidence", "ranking_confidence"],
    "rank": ["rank", "rank_position"],
}


def _norm(value) -> str:
    return re.sub(r"^\./", "", str(value).strip().replace("\\", "/"))


def _stem(name: str) -> str:
    for ext in (".cif.gz", ".pdb.gz", ".mmcif.gz", ".cif", ".pdb", ".mmcif", ".ent"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


class Ranking:
    """The manifest as a list of records {entry, cand (normalised path as written), score}; nothing is merged or overwritten."""

    def __init__(self, records=None):
        self.records = list(records or [])

    def __bool__(self):
        return bool(self.records)


def read_ranking(path, columns=None) -> Ranking:
    """Read a ranking table / candidate manifest (higher score = better). Columns are auto-detected when unambiguous.

    `columns`: optional {"entry": col, "candidate": col, "score": col | "rank": col}. A `rank` column (1 = best) becomes
    score = -rank. Every row is kept as written; the matching to scored candidates happens per entry in `resolve_ranking`."""
    if not path:
        return Ranking()
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        lower = {f.lower().strip(): f for f in fields}
        use = dict(columns or {})
        for role in ("entry", "candidate"):
            if role not in use:
                hits = [lower[a] for a in ALIASES[role] if a in lower]
                if len(hits) != 1:
                    raise SystemExit(f"{path}: cannot tell which column is the {role} ({'candidates: ' + ', '.join(hits) if hits else 'none matched'}); "
                                     f"columns are {fields}. Pass --ranking-columns {role}=<column>")
                use[role] = hits[0]
        if "score" not in use and "rank" not in use:
            hits = [lower[a] for a in ALIASES["score"] if a in lower]
            rhits = [lower[a] for a in ALIASES["rank"] if a in lower]
            if len(hits) == 1:
                use["score"] = hits[0]
            elif not hits and len(rhits) == 1:
                use["rank"] = rhits[0]
            else:
                raise SystemExit(f"{path}: cannot tell which column is the score or rank (score-like: {hits}, rank-like: {rhits}); "
                                 f"columns are {fields}. Pass --ranking-columns score=<column> (higher = better) or rank=<column> (1 = best)")
        for col in use.values():
            if col not in fields:
                raise SystemExit(f"{path}: no column {col!r}; columns are {fields}")
        print(f"ranking from {path}: entry={use['entry']!r} candidate={use['candidate']!r} " +
              (f"score={use['score']!r} (higher = better)" if "score" in use else f"rank={use['rank']!r} (1 = best)"), flush=True)
        records = []
        for r in rd:
            try:
                v = float(r[use["score"]]) if "score" in use else -float(r[use["rank"]])
            except (TypeError, ValueError):
                v = float("nan")
            records.append({"entry": r[use["entry"]].strip(), "cand": _norm(r[use["candidate"]]), "score": v})
    return Ranking(records)


def _cand_matches(p: str, cid: str) -> bool:
    """Does a manifest path `p` denote the scored candidate `cid` (path relative to the candidate root)?"""
    if p == cid or p.endswith("/" + cid):
        return True
    base = cid.rsplit("/", 1)[-1]
    if "/" not in p:
        return p == base or _stem(p) == _stem(base)
    return _stem(p) == _stem(cid) or _stem(p).endswith("/" + _stem(cid))


def resolve_ranking(ranking: Ranking, meta: dict) -> dict:
    """Match the manifest rows of one entry to its scored candidates, WITHOUT merging or overwriting anything.

    Returns {"scores": [K] (NaN = no score), "manifest": [(cand, score, matched_id | None)], "problems": [str]}.
    A row that matches no scored candidate is a candidate that failed or is missing (kept: it still occupies its rank).
    Problems: a bare file name that matches several scored candidates (ambiguous), one row matching several, or two rows
    with different scores for the same candidate (conflict)."""
    keys = {meta["entry"].upper(), meta["pdb_id"].upper()}
    ids = meta["candidates"]
    scores = np.full(len(ids), np.nan)
    manifest, problems = [], []
    for rec in ranking.records:
        if rec["entry"].upper() not in keys:
            continue
        hit = [i for i, cid in enumerate(ids) if _cand_matches(rec["cand"], cid)]
        if len(hit) > 1:
            problems.append(f"ambiguous: manifest row {rec['cand']!r} matches {len(hit)} candidates ({', '.join(ids[i] for i in hit[:3])}...)")
            manifest.append((rec["cand"], rec["score"], None))
            continue
        if not hit:
            manifest.append((rec["cand"], rec["score"], None))
            continue
        i = hit[0]
        if np.isfinite(scores[i]) and np.isfinite(rec["score"]) and scores[i] != rec["score"]:
            problems.append(f"conflict: candidate {ids[i]!r} has two different scores in the manifest ({scores[i]} and {rec['score']})")
        elif np.isfinite(rec["score"]):
            scores[i] = rec["score"]
        manifest.append((rec["cand"], rec["score"], ids[i]))
    return {"scores": scores, "manifest": manifest, "problems": problems}


def top_n_status(resolved: dict, n_scored: int, n_top: int):
    """(valid, reason). Top-N of the ORIGINAL ranking is usable only if no unscored manifest candidate could belong to it."""
    scores = resolved["scores"]
    if not np.isfinite(scores).all():
        return False, f"{int((~np.isfinite(scores)).sum())} scored candidate(s) have no manifest score"
    unscored = [sc for _, sc, mid in resolved["manifest"] if mid is None]
    if any(not np.isfinite(sc) for sc in unscored):
        return False, "a manifest row without a numeric score does not match any scored candidate"
    if not unscored:
        return True, ""
    allv = sorted([sc for _, sc, _ in resolved["manifest"]], reverse=True)
    thr = allv[n_top - 1] if len(allv) >= n_top else -np.inf
    if any(sc >= thr for sc in unscored):
        return False, f"an unscored (failed/missing) candidate belongs to the original top {n_top}"
    return True, ""


def sha1_of(path, chunk=1 << 20):
    """SHA-1 of a file's content; .gz files are hashed DECOMPRESSED (a re-download with a different gzip header is the same data)."""
    h = hashlib.sha1()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def candidate_root(pattern: str) -> str:
    """Fixed directory part of a glob pattern: candidate ids are paths relative to it."""
    m = re.search(r"[*?\[]", pattern)
    prefix = pattern[: m.start()] if m else pattern
    if prefix.endswith(("/", "\\")):
        return prefix
    return os.path.dirname(prefix) or "."


def candidate_id(file: str, root: str) -> str:
    try:
        return os.path.relpath(file, root).replace(os.sep, "/")
    except ValueError:                                         # different drive on Windows
        return Path(file).name


def fingerprint(ids_and_files, truth_path, cif_path, environment, kind, chain, min_coverage):
    from accessfold import __version__
    h = hashlib.sha1()
    h.update(repr((SCHEMA, __version__, environment, kind, chain, float(min_coverage), sha1_of(truth_path),
                   sha1_of(cif_path), [(cid, sha1_of(f)) for cid, f in ids_and_files])).encode())
    return h.hexdigest()


def quarantine(out_dir, name):
    """Move an entry's previous result out of the way so it can never be collected after a failed or rejected rerun."""
    target = Path(out_dir) / f"{name}.npz"
    if target.exists():
        qdir = Path(out_dir) / "_quarantine"
        qdir.mkdir(exist_ok=True)
        shutil.move(str(target), str(qdir / f"{name}.{time.strftime('%Y%m%dT%H%M%S')}.npz"))


def check_truth_against_structure(truth, tmeta, tstruct, tcomplete, tqc, radii, n_points, mask):
    """Recompute the truth accessibility from the cached structure and compare with the saved truth file (--verify-truth)."""
    from accessfold.truth import truth_arrays
    ref = truth_arrays(tstruct, tcomplete, radii, n_points, modified=tqc["modified_residues"],
                       shadow_threshold=tmeta.get("shadow_threshold", 5.0))
    for r in radii:
        if not np.allclose(ref[f"rel_{r:g}"][mask], truth[f"rel_{r:g}"][mask], atol=1e-6, equal_nan=True):
            raise RuntimeError(f"--verify-truth: recomputed accessibility at {r:g} A differs from the saved truth file")
    return "truth recomputed from the cached structure and matched"


def verify_reused(target, tpath, cif, key, old, out_dir):
    """--verify-truth on a REUSED result: check the truth file against the cached structure now (without rescoring) and
    record the outcome in the stored meta. Raises if the recomputation disagrees (the caller then quarantines the result)."""
    from accessfold.structures.mmcif import load_chain_from_mmcif
    with np.load(tpath, allow_pickle=False) as z:
        truth = {k: z[k] for k in z.files if k != "meta"}
        tmeta = json.loads(str(z["meta"]))
    if tmeta.get("source_cif_sha1") and tmeta["source_cif_sha1"] != sha1_of(cif):
        raise RuntimeError("the cached structure is not the one the truth accessibility was computed from (content hash differs)")
    tstruct, tcomplete, tqc = load_chain_from_mmcif(cif, key[1], tmeta["sequence"])
    note = check_truth_against_structure(truth, tmeta, tstruct, tcomplete, tqc, tmeta["probe_radii"], int(tmeta["n_points"]),
                                         truth["mask"].astype(bool))
    with np.load(target, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
    old = dict(old, truth_check=note)
    tmp = Path(out_dir) / f".{target.stem}.tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(old)), **arrays)
    tmp.replace(target)


# ------------------------------------------------------------------------------------------------ compute
def process_entry(task):
    """Returns (entry, error or None, status) with status in {"scored", "reused", "reused+verified", "stale->rescored", "failed"}."""
    mode = task[12] if len(task) > 12 else "combined"
    do_accessibility = mode != "structure"
    do_structure = mode != "accessibility"
    (row, truth_dir, cache_dir, pattern, cand_chain, out_dir, environment, kind, overwrite,
     min_candidates, min_coverage, verify_truth) = task[:12]
    key = row["_key"]
    name = entry_name(key)
    target = Path(out_dir) / f"{name}.npz"
    status = "scored"
    try:
        from accessfold.scoring import candidate_profile, corr_error_matrix, error_matrix, truth_profile
        from accessfold.structure_metrics import ca_coordinates, pairwise_rmsd, structural_metrics
        from accessfold.structures.mmcif import load_chain_from_mmcif
        from accessfold.structures.predicted import load_predicted_chain

        t0 = time.time()
        tpath = Path(truth_dir) / f"{name}.npz"
        if not tpath.exists():
            raise FileNotFoundError(f"no truth file {tpath} (run compute_truth_accessibility.py first)")
        cif = Path(cache_dir) / f"{key[0]}.cif.gz"
        pat = pattern.format(entry=name, pdb_id=key[0], chain=key[1])
        files = sorted(glob.glob(pat))
        if not files:
            raise FileNotFoundError(f"no candidates match {pat!r}")
        root = candidate_root(pat)
        pairs = [(candidate_id(f, root), f) for f in files]
        if len({c for c, _ in pairs}) != len(pairs):
            raise RuntimeError("candidate identifiers (paths relative to the candidate root) are not unique")
        fp = fingerprint(pairs, tpath, cif, environment, kind, cand_chain, min_coverage) + ":" + mode
        if target.exists() and not overwrite:
            try:
                with np.load(target, allow_pickle=False) as z:
                    old = json.loads(str(z["meta"]))
                same = old.get("fingerprint") == fp
            except Exception:  # noqa: BLE001 - unreadable old result: recompute
                same = False
            if same:
                k_old = len(old["candidates"])
                if k_old < min_candidates:                       # the stored result must satisfy the CURRENT requirement
                    raise RuntimeError(f"stored result has only {k_old} usable candidate(s) (<{min_candidates} required now)")
                if verify_truth and do_accessibility and "matched" not in str(old.get("truth_check", "")):
                    verify_reused(target, tpath, cif, key, old, out_dir)      # reuse must not skip the requested check
                    return name, None, "reused+verified"
                return name, None, "reused"
            status = "stale->rescored"

        with np.load(tpath, allow_pickle=False) as z:
            truth = {k: z[k] for k in z.files if k != "meta"}
            tmeta = json.loads(str(z["meta"]))
        radii, n_points, sequence = tmeta["probe_radii"], int(tmeta["n_points"]), tmeta["sequence"]
        src_hash = sha1_of(cif)
        if tmeta.get("source_cif_sha1"):
            if tmeta["source_cif_sha1"] != src_hash:
                raise RuntimeError("the cached structure is not the one the truth accessibility was computed from "
                                   "(content hash differs); recompute the truth for this entry")
        truth_note = "" if tmeta.get("source_cif_sha1") else "truth file has no source-structure hash (made by an older script); only residue coverage was checked"
        tstruct, tcomplete, tqc = load_chain_from_mmcif(cif, key[1], sequence)
        if not np.array_equal(tstruct.resolved_mask, truth["resolved"]):
            raise RuntimeError("truth file does not match the structure in the cache (different loader version or structure?)")
        mask = truth["mask"].astype(bool)
        if verify_truth and do_accessibility:
            truth_note = check_truth_against_structure(truth, tmeta, tstruct, tcomplete, tqc, radii, n_points, mask)
        t_ca, t_has = ca_coordinates(tstruct)
        y = truth_profile(truth, radii, kind) if do_accessibility else None
        needed = mask[:, None] & np.isfinite(y) if do_accessibility else None           # truth-DEFINED entries only (e.g. glycine has no side-chain value)

        # Everything for one candidate is computed first and appended TOGETHER, so a failure at any step can never leave
        # one candidate's metrics paired with another candidate's identifier.
        names, mets, profiles, cas, has_l, covs, failed, warns = [], [], [], [], [], [], [], {}
        for cid, f in pairs:
            try:
                cand, cand_complete, qc = load_predicted_chain(f, cand_chain, sequence)
                c_ca, c_has = ca_coordinates(cand)
                cov_ca = float((c_has & t_has).sum() / t_has.sum())
                # Predicted accessibility for this candidate.
                prof = candidate_profile(cand, tstruct, truth, radii, n_points, environment, kind) if do_accessibility else None
                usable = ((~needed | np.isfinite(prof)).all(axis=1) & cand_complete) if do_accessibility else cand_complete
                cov_mask = float(usable[mask].mean()) if mask.any() else 1.0
                coverage = min(cov_ca, cov_mask) if do_accessibility else cov_ca
                if coverage < min_coverage:
                    raise ValueError(f"incomplete candidate: Ca present for {cov_ca:.3f} of the truth-resolved residues, complete "
                                     f"atoms and defined accessibility for {cov_mask:.3f} of the truth-mask residues "
                                     f"(required: {min_coverage})")
                # Structural accuracy against experimental geometry.
                met = structural_metrics(t_ca, t_has, c_ca, c_has) if do_structure else None
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not lose the entry
                failed.append(f"{cid}: {type(exc).__name__}: {exc}")
                continue
            names.append(cid)
            mets.append(met)
            profiles.append(prof)
            cas.append(c_ca)
            has_l.append(c_has)
            covs.append((cov_ca, cov_mask))
            if qc.get("warnings"):
                warns[cid] = qc["warnings"]
        if len(names) < min_candidates:
            raise RuntimeError(f"only {len(names)} usable candidate(s) (<{min_candidates}); failures: {failed[:3]}")

        if do_accessibility:
            c_all = np.stack(profiles).astype(np.float32)
            # Accessibility errors against the saved experimental reference.
            mae, rmse, n_ac = error_matrix(y, c_all.astype(float))
            corr_err, _ = corr_error_matrix(y, c_all.astype(float))
        if do_structure:
            # Structural spread between candidates.
            common = t_has & np.logical_and.reduce(has_l)
            if common.sum() >= 3:
                pair = pairwise_rmsd(np.stack([c[common] for c in cas])).astype(np.float32)
            else:
                pair = np.full((len(names), len(names)), np.nan, np.float32)
        meta = {"entry": name, "pdb_id": key[0], "chain": key[1], "split": tmeta.get("split", ""),
                "sequence_length": len(sequence), "radii": radii, "n_points": n_points, "environment": environment,
                "kind": kind, "mode": mode, "min_coverage": min_coverage, "candidates": names, "failed_candidates": failed,
                "candidate_warnings": warns, "n_residues_structural": int(t_has.sum()),
                "n_residues_accessibility": int(mask.sum()), "fingerprint": fp, "schema": SCHEMA,
                "truth_source_sha1": src_hash, "truth_check": truth_note, "seconds": round(time.time() - t0, 2)}
        arrays = {
            "coverage_ca": np.array([c[0] for c in covs]),
            "coverage_mask": np.array([c[1] for c in covs]),
        }
        if do_structure:
            arrays.update(
                rmsd_ca=np.array([m["rmsd_ca"] for m in mets]), lddt_ca=np.array([m["lddt_ca"] for m in mets]),
                tm_score=np.array([m["tm_score"] for m in mets]), n_common=np.array([m["n_common"] for m in mets]),
                n_reference=np.array([m["n_reference"] for m in mets]), pair_rmsd=pair)
        if do_accessibility:
            arrays.update(ac_mae=mae, ac_rmse=rmse, ac_corr=corr_err, n_ac=n_ac, y=y, c=c_all,
                          mask=truth["mask"], gap_distance=truth["gap_distance"])
        tmp = Path(out_dir) / f".{name}.tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(target)
        return name, None, status
    except Exception as exc:  # noqa: BLE001
        try:
            quarantine(out_dir, name)                          # an earlier result no longer matches the current inputs/requirements
        except OSError:
            pass
        return name, f"{type(exc).__name__}: {exc}", "failed"


# ------------------------------------------------------------------------------------------------ collect
def collect(out_dir: Path, top_n=(1, 5, 20), ranking=None, entries=None, mode="combined"):
    """Build candidate_scores.csv / entry_summary.csv from the results of `entries` (a set of entry names; None = all).
    Returns (n_entries, n_ranking_problems)."""
    from accessfold.controls import spearman
    from accessfold.scoring import aggregate_radii

    ranking = ranking or Ranking()
    cand_rows, entry_rows, notes = [], [], []
    n_problem = 0
    for f in sorted(out_dir.glob("*.npz")):
        if f.name.startswith("."):
            continue
        with np.load(f, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        if entries is not None and meta["entry"] not in entries:
            notes.append(f"ignored {f.name}: not part of this run's entries")
            continue
        if meta.get("schema") != SCHEMA:
            print(f"WARNING {f.name}: written by an older scoring schema ({meta.get('schema')}); rerun with --overwrite", file=sys.stderr)
            continue
        stored_mode = meta.get("mode", "combined")
        if stored_mode != mode:
            notes.append(f"ignored {f.name}: mode {stored_mode}, requested {mode}")
            continue
        radii = meta["radii"]
        k = len(meta["candidates"])
        # Placeholders are only for the shared collector, never saved or exported.
        if mode == "accessibility":
            for key in ("rmsd_ca", "lddt_ca", "tm_score", "n_common", "n_reference"):
                a[key] = np.zeros(k)
            a["pair_rmsd"] = np.zeros((k, k))
        elif mode == "structure":
            for key in ("ac_mae", "ac_rmse", "ac_corr"):
                a[key] = np.full((k, len(radii)), np.nan)
            a["n_ac"] = np.zeros((k, len(radii)), int)
        e = aggregate_radii(a["ac_mae"], a["n_ac"]) if mode != "structure" else np.full(k, np.nan)
        e_rmse = aggregate_radii(a["ac_rmse"], a["n_ac"]) if mode != "structure" else np.full(k, np.nan)
        e_corr = aggregate_radii(a["ac_corr"], a["n_ac"]) if mode != "structure" else np.full(k, np.nan)
        rm = a["rmsd_ca"]
        fin = np.isfinite(rm)

        status, sc, top_ok = "no ranking", np.full(k, np.nan), {}
        res = None
        if ranking:
            res = resolve_ranking(ranking, meta)
            sc = res["scores"]
            n_unscored = sum(1 for _, _, mid in res["manifest"] if mid is None)
            if not res["manifest"]:
                status = "entry not in manifest"
            elif res["problems"]:
                status = "conflict: " + "; ".join(res["problems"][:2])
                n_problem += 1
            else:
                status = "complete" if (np.isfinite(sc).all() and n_unscored == 0) else (
                    f"incomplete: {int((~np.isfinite(sc)).sum())} scored candidate(s) without score, {n_unscored} manifest candidate(s) not scored")
            for n in (1, *top_n):
                ok, why = (False, status) if res["problems"] or not res["manifest"] else top_n_status(res, k, n)
                top_ok[n] = (ok, why)
        # original ranks within the manifest (gaps where candidates failed): 1 = best
        rank = [""] * k
        if res and not res["problems"] and res["manifest"]:
            allv = sorted(((sc_, mid) for _, sc_, mid in res["manifest"] if np.isfinite(sc_)), key=lambda t: -t[0])
            order_ids = [mid for _, mid in allv]
            for i, cid in enumerate(meta["candidates"]):
                if cid in order_ids and np.isfinite(sc[i]):
                    rank[i] = 1 + sum(1 for sc_, _ in allv if sc_ > sc[i])
        for i, cid in enumerate(meta["candidates"]):
            r = {"entry": meta["entry"], "candidate": cid, "rank_score": sc[i], "rank": rank[i],
                 "rmsd_ca": a["rmsd_ca"][i], "lddt_ca": a["lddt_ca"][i], "tm_score": a["tm_score"][i],
                 "n_reference": a["n_reference"][i], "n_common": a["n_common"][i], "coverage_ca": a["coverage_ca"][i],
                 "coverage_mask": a["coverage_mask"][i], "ac_error": e[i], "ac_rmse": e_rmse[i],
                 "ac_corr_error": e_corr[i], "n_ac_residues": int(a["n_ac"][i].min())}
            for j, rad in enumerate(radii):
                r[f"ac_mae_{rad:g}"] = a["ac_mae"][i, j]
            cand_rows.append(mode_columns(r, mode))

        iu = np.triu_indices(k, 1)
        pr = a["pair_rmsd"][iu]
        s = {"entry": meta["entry"], "split": meta["split"], "n_candidates": k, "n_failed_candidates": len(meta["failed_candidates"]),
             "min_coverage_ca": float(a["coverage_ca"].min()), "min_coverage_mask": float(a["coverage_mask"].min()),
             "n_residues_structural": meta["n_residues_structural"], "n_residues_accessibility": meta["n_residues_accessibility"],
             "rmsd_min": np.nanmin(rm), "rmsd_median": np.nanmedian(rm), "rmsd_max": np.nanmax(rm),
             "rmsd_spread": np.nanmax(rm) - np.nanmin(rm), "pair_rmsd_median": np.nanmedian(pr) if pr.size else np.nan,
             "pair_rmsd_max": np.nanmax(pr) if pr.size else np.nan,
             "ranking_status": status if ranking else "",
             "n_manifest_candidates": len(res["manifest"]) if res else "",
             "n_manifest_not_scored": sum(1 for _, _, mid in res["manifest"] if mid is None) if res else "",
             "truth_check": meta.get("truth_check", "")}
        for t in THRESHOLDS_A:
            s[f"n_within_{t:g}A"] = int((rm[fin] < t).sum())
        by_score = np.argsort(-np.where(np.isfinite(sc), sc, -np.inf), kind="stable")
        ok1 = top_ok.get(1, (False, ""))[0]
        s["rmsd_top1"] = rm[by_score[0]] if ok1 else np.nan
        s["n_tied_at_top1"] = int((sc == sc[by_score[0]]).sum()) if ok1 else ""
        for n in top_n:
            ok = top_ok.get(n, (False, ""))[0]
            s[f"rmsd_best_top{n}"] = np.nanmin(rm[by_score[:n]]) if ok else np.nan
        s["top1_minus_best"] = s["rmsd_top1"] - s["rmsd_min"] if ok1 else np.nan
        s["topN_unavailable_reason"] = "; ".join(f"top{n}: {why}" for n, (ok, why) in top_ok.items() if not ok and why and n in (1, *top_n))[:300] if ranking else ""
        sel = int(np.nanargmin(e)) if np.isfinite(e).any() else -1
        s["rmsd_acc_selected"] = rm[sel] if sel >= 0 else np.nan             # candidate that agrees best with accessibility
        for m, label in (("rmsd_ca", "rmsd"), ("lddt_ca", "lddt"), ("tm_score", "tm")):
            err = rm if m == "rmsd_ca" else 1.0 - a[m]
            s[f"rho_ac_vs_{label}"] = spearman(e, err)
            s[f"rho_accorr_vs_{label}"] = spearman(e_corr, err)
        entry_rows.append(mode_columns(s, mode))
    for n in notes[:10]:
        print("NOTE:", n, file=sys.stderr)
    bad = [r for r in entry_rows if ranking and r["ranking_status"] not in ("complete",)]
    if bad:
        print(f"WARNING: ranking not complete for {len(bad)} entr(y/ies); affected top-ranked / best-of-top-N values are left empty: "
              + "; ".join(f"{r['entry']} [{r['ranking_status'][:80]}]" for r in bad[:6]), file=sys.stderr)

    def write(path, rows):
        if not rows:
            if path.exists():
                path.unlink()                  # nothing valid this run: an old table must not survive and look current
            return
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if (isinstance(r.get(c), float) and np.isnan(r[c])) else r.get(c, "")) for c in cols})

    write(out_dir / "candidate_scores.csv", cand_rows)
    write(out_dir / "entry_summary.csv", entry_rows)
    return len(entry_rows), n_problem


def mode_columns(row, mode):
    if mode == "combined":
        return row
    structural = lambda key: key.startswith(("rmsd", "lddt", "tm_", "pair_rmsd", "n_within", "rho_", "top1_")) or key in {
        "n_reference", "n_common", "n_residues_structural", "n_tied_at_top1", "topN_unavailable_reason"}
    accessibility = lambda key: key.startswith(("ac_", "n_ac")) or key in {"n_residues_accessibility"}
    if mode == "accessibility":
        return {k: v for k, v in row.items() if not structural(k)}
    return {k: v for k, v in row.items() if not accessibility(k) and not k.startswith("rho_")}


def parse_columns(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--ranking-columns expects role=column, got {it!r}")
        role, col = it.split("=", 1)
        if role not in ("entry", "candidate", "score", "rank"):
            raise SystemExit(f"--ranking-columns role must be entry, candidate, score or rank, got {role!r}")
        out[role] = col
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--truth-dir", default="truth_out")
    ap.add_argument("--cache-dir", default="cif_cache")
    ap.add_argument("--candidates", help="glob with {entry}/{pdb_id}/{chain} placeholders")
    ap.add_argument("--candidate-chain", default="A")
    ap.add_argument("--ranking-csv", help="candidate manifest / ranking table (applied when tables are collected)")
    ap.add_argument("--ranking-columns", nargs="+", metavar="ROLE=COLUMN",
                    help="override column detection: entry=.. candidate=.. score=.. (higher = better) or rank=.. (1 = best)")
    ap.add_argument("--out", default="scores_out")
    ap.add_argument("--environment", default="truth_atoms", choices=["truth_atoms", "truth_residues", "all_atoms"])
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--accessibility-only", action="store_true", help="skip structural metrics and pairwise RMSD")
    modes.add_argument("--structure-only", action="store_true", help="skip accessibility calculations and errors")
    ap.add_argument("--kind", default="rel", choices=["rel", "abs", "sc_rel"])
    ap.add_argument("--min-candidates", type=int, default=2)
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="a candidate must have Ca for this fraction of the truth-resolved residues AND complete atoms / defined "
                         "accessibility (where the truth value is defined) for this fraction of the truth-mask residues, else it is "
                         "rejected (default 1.0 = fully complete)")
    ap.add_argument("--verify-truth", action="store_true",
                    help="recompute the truth accessibility from the cached structure and compare it with the truth file")
    ap.add_argument("--top-n", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    mode = "accessibility" if a.accessibility_only else "structure" if a.structure_only else "combined"
    if a.structure_only and a.verify_truth:
        ap.error("--verify-truth recalculates accessibility; omit it with --structure-only")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking = read_ranking(a.ranking_csv, parse_columns(a.ranking_columns))
    rows = read_rows(a.csv)
    if a.only:
        want = {x.upper() for x in a.only}
        rows = [r for r in rows if r["_key"][0] in want or entry_name(r["_key"]) in want]
    if a.limit:
        rows = rows[: a.limit]
    names = {entry_name(r["_key"]) for r in rows}
    if a.collect_only:
        n, bad = collect(out_dir, tuple(a.top_n), ranking, names, mode)
        print(f"collected {n} entries -> {out_dir}")
        if bad:
            sys.exit(1)
        return
    if not a.candidates:
        ap.error("--candidates is required unless --collect-only")
    i, n = (int(x) for x in a.shard.split("/"))
    rows = rows[i::n]
    tasks = [(r, a.truth_dir, a.cache_dir, a.candidates, a.candidate_chain, str(out_dir), a.environment, a.kind,
              a.overwrite, a.min_candidates, a.min_coverage, a.verify_truth, mode) for r in rows]
    fails, t0 = [], time.time()
    if a.workers <= 1:
        results = (process_entry(t) for t in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=a.workers)
        results = (f.result() for f in as_completed([pool.submit(process_entry, t) for t in tasks]))
    for done, (name, err, status) in enumerate(results, 1):
        if err:
            fails.append((name, err))
            print(f"[{done}/{len(tasks)}] FAILED {name}: {err}", flush=True)
        else:
            print(f"[{done}/{len(tasks)}] {status} {name}", flush=True)
    fp = out_dir / (f"failures_shard{i}of{n}.csv" if n > 1 else "failures.csv")
    if fails:
        with open(fp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["entry", "error"])
            w.writerows(fails)
        print(f"{len(fails)} failure(s) -> {fp} (their previous results, if any, were moved to {out_dir / '_quarantine'})")
    elif fp.exists():
        fp.unlink()                                  # a clean run must not leave a stale failure list behind
    bad = 0
    if n == 1:
        cnt, bad = collect(out_dir, tuple(a.top_n), ranking, {entry_name(r["_key"]) for r in rows}, mode)
        print(f"collected {cnt} entries; {time.time() - t0:.0f}s")
    if fails or bad:
        sys.exit(1)                                  # schedulers (PBS/SLURM) must see an incomplete run as a failure


if __name__ == "__main__":
    main()
