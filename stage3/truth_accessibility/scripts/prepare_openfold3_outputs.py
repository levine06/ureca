#!/usr/bin/env python
"""Stage 3 pre-flight for OpenFold3 predictions: check them against Stage 1, and write the ranking table that
score_candidates.py reads.

OpenFold3 writes (see https://openfold-3.readthedocs.io/en/latest/inference.html)
    <output_dir>/<query>/seed_<seed>/<query>_seed_<seed>_sample_<i>_model.cif          (or .pdb)
    <output_dir>/<query>/seed_<seed>/<query>_seed_<seed>_sample_<i>_confidences_aggregated.json
This script walks that tree and, for every model
  * maps the query name to a Stage 1 entry (`2ID7_A` exactly, or the bare PDB id when that is unambiguous);
  * VERIFIES the model against the Stage 1 sequence with the same reader score_candidates.py uses (chain present, residue
    names equal to the sequence, numbering 1..N, no duplicate atoms), so problems show up now and not after hours of scoring;
  * reads `sample_ranking_score` (and avg_plddt, ptm, iptm, gpde, has_clash) from the sibling aggregated JSON.
It writes ranking.csv (columns entry, candidate, score, ...) which score_candidates.py accepts as --ranking-csv with no
extra options, prints a per-entry report, and prints the --candidates pattern to use. Nothing is scored here.

    python scripts/prepare_openfold3_outputs.py --csv development.csv validation.csv --predictions of3_out --out ranking.csv
    python scripts/score_candidates.py --csv development.csv validation.csv --truth-dir truth_out --cache-dir cif_cache \\
        --candidates "of3_out/{entry}/seed_*/*_model.cif*" --candidate-chain A --ranking-csv ranking.csv --out scores_out

--chain: the chain id inside the OpenFold3 files (default A; `auto` takes the single polymer chain of each file). Whether
OpenFold3 keeps your query chain id, and numbers residues 1..N, is checked here, not assumed.
Direction of the ranking score: higher is assumed better (as for AlphaFold3's ranking score); the OpenFold3 page does not
say so explicitly and gives no weights, so confirm it on one entry before relying on top-ranked results.
Exit status 1 if any model failed validation or any Stage 1 entry has no predictions (use --allow-missing to accept the latter).
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

    rows, per_entry, unmatched, problems = [], defaultdict(list), defaultdict(int), []
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
    queries = {r["query"] for r in rows}
    dirs = "{entry}" if all(q.upper() in by_name for q in queries) else "{pdb_id}"
    print(f"\nuse:  --candidates \"{root.as_posix()}/{dirs}/seed_*/*_model.cif*\" --candidate-chain {a.chain if a.chain != 'auto' else '<chain printed in the failures above, or A>'} --ranking-csv {a.out}")
    if problems or (missing and not a.allow_missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
