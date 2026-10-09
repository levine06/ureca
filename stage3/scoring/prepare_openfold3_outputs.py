#!/usr/bin/env python
"""Prepare existing OpenFold3 outputs; no accessibility or accuracy calculations.

Candidate OpenFold3 structure generation belongs to
../candidate_generation/generate_candidates.py, not this script.

Input: existing OpenFold3 model files and the development/validation CSVs.
This script checks candidate sequences/chains, matches files to dataset entries,
and reads ranking scores from sibling *_confidences_aggregated.json files.
Output: ranking.csv, a validation report and the actual --candidates path to use
for score_candidates.py. Missing confidence files leave scores blank; structural
and accessibility calculations can proceed without model-ranking comparisons.

    python stage3/scoring/prepare_openfold3_outputs.py --csv development.csv validation.csv \\
        --predictions of3_out --out ranking.csv

--chain is the model's label_asym_id for CIFs (default A); auto requires one
polymer chain. Higher ranking scores are treated as better.
Exit status 1 means a candidate failed validation or a dataset entry had no
predictions; --allow-missing permits entries that have not been generated yet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


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

MODEL_RE = re.compile(r"^(?P<query>.+)_seed_(?P<seed>\d+)_sample_(?P<sample>\d+)_model\.(?P<ext>cif|pdb)(?P<gz>\.gz)?$")
FIELDS = ["avg_plddt", "ptm", "iptm", "gpde", "disorder", "has_clash"]


def read_entries(csv_paths):
    rows, seen = [], set()
    for p in csv_paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["pdb_id"].strip().upper(), r["label_chain_id"].strip())
                if key not in seen:
                    seen.add(key)
                    rows.append({"entry": f"{key[0]}_{key[1]}", "pdb_id": key[0], "chain": key[1], "sequence": r["sequence"].strip()})
    return rows


def polymer_chains(path: Path):
    """Sorted chain ids that carry polymer residues in a model file (mmCIF: label_asym_id with a label_seq_id; PDB: column 22)."""
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    if ".cif" in name:
        from biotite.structure.io.pdbx import CIFFile
        with opener(path, "rt") as fh:
            cif = CIFFile.read(fh)
        atom = cif[next(iter(cif.keys()))]["atom_site"]
        chain = atom["label_asym_id"].as_array(str)
        seq = atom["label_seq_id"].as_array(str)
        return sorted({str(c) for c, s in zip(chain, seq) if s not in (".", "?")})
    out = set()
    with opener(path, "rt") as fh:
        for line in fh:
            if line[:6] in ("ATOM  ", "HETATM"):
                out.add(line[21])
    return sorted(out)


def candidate_patterns(root: Path, matched, by_name):
    """--candidates pattern(s) that reach every matched model file, built from where the files really are.

    Layout assumed below the root: <subfolders...>/<query>/seed_<n>/<file>. Folders that are the same for all files at a
    given position are kept literally (e.g. development/), those that differ become `*` (development/ vs validation/),
    and seed_<n> becomes seed_*. The query folder is written {entry} or {pdb_id} when it is named like that. Files with a
    different layout (depth, query-folder naming or extension) get their own pattern line. Returns (patterns, notes)."""
    groups, notes = defaultdict(list), []
    for m, entry, q in matched:
        rel = m.relative_to(root).parts
        idx = max((i for i, part in enumerate(rel[:-1]) if part == q), default=None)
        if idx is None:                                   # no folder named like the query: assume <query>/seed_<n>/file
            idx = max(len(rel) - 3, 0)
            notes.append(f"{m.name}: no folder named {q!r} on its path; assumed {rel[idx]!r} is the query folder")
        ext = "pdb" if ".pdb" in m.name.lower() else "cif"
        groups[(len(rel[:idx]), len(rel[idx + 1:-1]), q.upper() in by_name, ext)].append((rel[:idx], rel[idx + 1:-1]))
    pats = []
    for (n_pre, n_tail, is_entry, ext), items in sorted(groups.items()):
        pre = [vals.pop() if len(vals := {p[i] for p, _ in items}) == 1 else "*" for i in range(n_pre)]
        tail = []
        for i in range(n_tail):
            vals = {t[i] for _, t in items}
            tail.append("seed_*" if all(v.startswith("seed_") for v in vals) else (next(iter(vals)) if len(vals) == 1 else "*"))
        pats.append("/".join([root.as_posix(), *pre, "{entry}" if is_entry else "{pdb_id}", *tail, f"*_model.{ext}*"]))
    if len(pats) > 1:
        notes.append("the model files are not all laid out the same way, so there is one pattern per layout; run "
                     "score_candidates.py once per pattern with the matching --csv rows, or reorganise the folders")
    return pats, notes


def read_confidences(model: Path):
    js = model.with_name(MODEL_RE.sub(lambda m: f"{m['query']}_seed_{m['seed']}_sample_{m['sample']}_confidences_aggregated.json", model.name))
    if not js.exists():
        return None, f"missing {js.name}"
    try:
        data = json.loads(js.read_text())
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable {js.name}: {exc}"
    return data, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="Stage 1 dataset CSV file(s)")
    ap.add_argument("--predictions", required=True, help="OpenFold3 output directory")
    ap.add_argument("--out", default="ranking.csv")
    ap.add_argument("--chain", default="A", help="chain id in the model files, or 'auto'")
    ap.add_argument("--ranking-field", default="sample_ranking_score")
    ap.add_argument("--no-validate", action="store_true", help="skip the sequence/numbering check (faster; not recommended)")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail when a Stage 1 entry has no predictions")
    a = ap.parse_args()

    entries = read_entries(a.csv)
    by_name = {e["entry"].upper(): e for e in entries}
    by_pdb = defaultdict(list)
    for e in entries:
        by_pdb[e["pdb_id"]].append(e)

    root = Path(a.predictions)
    models = sorted(p for p in root.rglob("*") if p.is_file() and MODEL_RE.match(p.name))
    if not models:
        sys.exit(f"no files named <query>_seed_<n>_sample_<i>_model.cif|pdb found under {root}")

    rows, per_entry, unmatched, problems, matched = [], defaultdict(list), defaultdict(int), [], []
    for m in models:
        g = MODEL_RE.match(m.name)
        q = g["query"]
        entry = by_name.get(q.upper())
        if entry is None and len(by_pdb.get(q.upper(), [])) == 1:
            entry = by_pdb[q.upper()][0]
        if entry is None:
            unmatched[q] += 1
            continue
        conf, conf_err = read_confidences(m)
        row = {"entry": entry["entry"], "candidate": m.name, "score": "", "seed": g["seed"], "sample": g["sample"], "query": q,
               "validated": "skipped" if a.no_validate else "ok", "note": conf_err or ""}
        if conf is not None:
            if a.ranking_field in conf and isinstance(conf[a.ranking_field], (int, float)):
                row["score"] = conf[a.ranking_field]
            else:
                row["note"] = f"no numeric {a.ranking_field!r} in the aggregated json"
            for f in FIELDS:
                v = conf.get(f)
                row[f] = v if isinstance(v, (int, float)) else ""
        if not a.no_validate:
            try:
                from accessfold.structures.predicted import load_predicted_chain
                chain = a.chain
                if chain == "auto":
                    found = polymer_chains(m)
                    if len(found) != 1:
                        raise ValueError(f"--chain auto needs exactly one polymer chain, found {found}")
                    chain = found[0]
                load_predicted_chain(m, chain, entry["sequence"])
            except Exception as exc:  # noqa: BLE001
                found = ""
                try:
                    found = f" (polymer chains in the file: {polymer_chains(m)})"
                except Exception:  # noqa: BLE001
                    pass
                row["validated"] = f"FAILED: {type(exc).__name__}: {str(exc)[:220]}{found}"
                problems.append((m.name, row["validated"]))
        rows.append(row)
        matched.append((m, entry, q))
        per_entry[entry["entry"]].append(row)

    cols = ["entry", "candidate", "score"] + FIELDS + ["seed", "sample", "query", "validated", "note"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"{len(models)} model files, {len(rows)} matched to Stage 1 entries -> {a.out}")
    missing = [e["entry"] for e in entries if e["entry"] not in per_entry]
    print(f"{'entry':<10}{'models':>7}{'seeds':>6}{'scored':>7}{'valid':>6}  ranking score range")
    for name in sorted(per_entry):
        rs = per_entry[name]
        sc = [float(r["score"]) for r in rs if r["score"] != ""]
        ok = sum(1 for r in rs if r["validated"] in ("ok", "skipped"))
        rng = f"{min(sc):.3f} .. {max(sc):.3f}" + ("  (constant!)" if len(sc) > 1 and min(sc) == max(sc) else "") if sc else "none"
        print(f"{name:<10}{len(rs):>7}{len({r['seed'] for r in rs}):>6}{len(sc):>7}{ok:>6}  {rng}")
    if unmatched:
        print("\nWARNING: query names that match no Stage 1 entry (renamed queries?): " + ", ".join(f"{k} ({v} files)" for k, v in list(unmatched.items())[:10]))
    if missing:
        print(f"\n{len(missing)} Stage 1 entr(y/ies) have no predictions yet: " + ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else ""))
    if problems:
        print(f"\n{len(problems)} model(s) FAILED validation, first few:")
        for n, msg in problems[:8]:
            print(f"  {n}: {msg}")
    patterns, notes = candidate_patterns(root, matched, by_name)
    chain_txt = a.chain if a.chain != "auto" else "<chain printed in the failures above, or A>"
    print("\nuse (one --candidates pattern; the folders between the predictions root and the query folder are included):")
    for pat in patterns:
        print(f"  --candidates \"{pat}\" --candidate-chain {chain_txt} --ranking-csv {a.out}")
    for n in notes:
        print("  NOTE: " + n)
    if problems or (missing and not a.allow_missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
