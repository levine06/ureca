#!/usr/bin/env python3
"""
retry_seqsearch.py
Retry RCSB pre-cutoff sequence similarity searches that failed with api_error.

Background
----------
  compute_similarity.py uses the RCSB BLAST API to classify each post-cutoff
  candidate as Test A (pre-cutoff similar) or Test B (novel).  Occasionally the
  API returns HTTP 200 with an empty or malformed body, probably due to transient
  server load.  Those rows receive precutoff_search_status == "api_error" and
  training_or_test_split == "post_cutoff_candidate" (pending review).

  This script:
    1. Reads master_candidates_clustered.csv.
    2. Finds rows where precutoff_search_status == "api_error" AND
       training_or_test_split == "post_cutoff_candidate".
    3. Retries the sequence search for each such row (up to --max-retries each,
       with increasing back-off between rows).
    4. Writes the updated assignments back to the CSV (or a new --out-csv path).
    5. Applies MANUAL_SPLIT_OVERRIDES after the search.

Usage
-----
  python retry_seqsearch.py --csv output/master_candidates_clustered.csv

  Optional flags:
    --out-csv PATH   Write to a different file (default: overwrite --csv)
    --max-retries N  Per-row retry attempts (default: 5, more than compute's 3)
    --sleep SECS     Base sleep between rows in seconds (default: 1.0)

Requirements
------------
  Python packages: requests
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constants (must match compute_similarity.py) ───────────────────────────────
CUTOFF_DATE        = "2021-09-30"
TESTB_MAX_PIDENT   = 0.30
TESTB_MIN_COVERAGE = 0.80
MAX_SEARCH_RESULTS = 1000

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"

# ── Manual split overrides (must match compute_similarity.py) ─────────────────
MANUAL_SPLIT_OVERRIDES: dict[str, str] = {
    "8EXF_B": "test_set_B_candidate",
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "stage1-retry-seqsearch/1.0 (NTU research)"


# ── Sequence search (duplicated from compute_similarity.py for standalone use) ─

def _alignment_metrics(result: dict) -> list[dict]:
    """Extract explicit identity/coverage values from an RCSB search result."""
    alignments = []
    for service in result.get("services", []):
        if service.get("service_type") != "sequence":
            continue
        for node in service.get("nodes", []):
            for match in node.get("match_context", []):
                q_len = int(match.get("query_length") or 0)
                s_len = int(match.get("subject_length") or 0)
                q_span = max(
                    0,
                    int(match.get("query_end") or 0)
                    - int(match.get("query_beg") or 0)
                    + 1,
                )
                s_span = max(
                    0,
                    int(match.get("subject_end") or 0)
                    - int(match.get("subject_beg") or 0)
                    + 1,
                )
                if not q_len or not s_len:
                    continue
                alignments.append({
                    "hit": result.get("identifier", ""),
                    "identity": float(match.get("sequence_identity") or 0.0),
                    "coverage": q_span / q_len,
                    "query_coverage": q_span / q_len,
                    "subject_coverage": s_span / s_len,
                    "evalue": float(match.get("evalue") or 0.0),
                })
    return alignments


def query_precutoff_similarity(
    sequence: str, retries: int = 5
) -> tuple[str, float, float, float, str]:
    """
    Query RCSB sequence-similarity search for the best pre-cutoff PDB hit.

    Returns (best_entity_id, identity, coverage, evalue, status).
    Status values: "similar_hit", "no_similar_hit", "incomplete_search", "api_error".
    """
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 1000,
                        "identity_cutoff": TESTB_MAX_PIDENT,
                        "sequence_type": "protein",
                        "value": sequence,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less_or_equal",
                        "value": f"{CUTOFF_DATE}T00:00:00Z",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": MAX_SEARCH_RESULTS},
            "scoring_strategy": "sequence",
            "results_verbosity": "verbose",
        },
    }
    for attempt in range(retries):
        try:
            resp = SESSION.post(RCSB_SEARCH, json=payload, timeout=60)
            if not resp.ok:
                print(f"    [seqsearch] HTTP {resp.status_code}: {resp.text[:200]}",
                      file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            data = resp.json()
            results = data.get("result_set", [])
            total_count = int(data.get("total_count") or len(results))
            alignments = [
                a for result in results for a in _alignment_metrics(result)
            ]
            if not alignments:
                return ("", 0.0, 0.0, 0.0, "no_similar_hit")

            global_hits = [a for a in alignments if a["coverage"] >= TESTB_MIN_COVERAGE]
            if not global_hits:
                best_local = max(alignments, key=lambda a: (a["coverage"], a["identity"]))
                status = "incomplete_search" if total_count > len(results) else "no_similar_hit"
                return (
                    best_local["hit"], best_local["identity"],
                    best_local["coverage"], best_local["evalue"],
                    status,
                )

            best = max(global_hits, key=lambda a: (a["identity"], a["coverage"]))
            status = "similar_hit" if best["identity"] >= TESTB_MAX_PIDENT else "no_similar_hit"
            return (best["hit"], best["identity"], best["coverage"], best["evalue"], status)

        except Exception as exc:
            print(f"    [seqsearch] attempt {attempt+1}/{retries} failed: {exc}",
                  file=sys.stderr)
            time.sleep(2 ** attempt)

    return ("", 0.0, 0.0, 0.0, "api_error")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry api_error rows from master_candidates_clustered.csv"
    )
    parser.add_argument("--csv", required=True,
                        help="Path to master_candidates_clustered.csv")
    parser.add_argument("--out-csv", default=None,
                        help="Output path (default: overwrite --csv)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Per-row retry attempts (default: 5)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Base sleep between rows in seconds (default: 1.0)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out_csv) if args.out_csv else csv_path

    # ── Load CSV ───────────────────────────────────────────────────────────────
    print(f"\n=== Loading {csv_path.name} ===")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # ── Identify rows to retry ─────────────────────────────────────────────────
    retry_rows = [
        r for r in all_rows
        if r.get("precutoff_search_status") == "api_error"
        and r.get("training_or_test_split") == "post_cutoff_candidate"
    ]
    print(f"  Total rows: {len(all_rows)}")
    print(f"  Rows to retry: {len(retry_rows)} (precutoff_search_status == api_error)")

    if not retry_rows:
        print("  Nothing to retry — all rows are already resolved.")
        return

    print(f"\n  Expected time: ~{len(retry_rows) * args.sleep / 60:.1f}–"
          f"{len(retry_rows) * args.max_retries / 60:.1f} min\n")

    # ── Retry each row ─────────────────────────────────────────────────────────
    n_resolved = 0
    n_still_error = 0
    n_incomplete = 0

    # Build a fast lookup: key → row object (rows are mutable dicts in all_rows)
    row_by_key = {f"{r['PDB_ID']}_{r['chain_ID']}": r for r in all_rows}

    for i, row in enumerate(retry_rows):
        key = f"{row['PDB_ID']}_{row['chain_ID']}"
        seq = row.get("sequence", "").strip()

        if not seq:
            print(f"  [{i+1:3d}/{len(retry_rows)}] {key:12s}  SKIP — no sequence in CSV",
                  flush=True)
            n_still_error += 1
            continue

        print(f"  [{i+1:3d}/{len(retry_rows)}] {key:12s}  retrying …", end="  ", flush=True)
        hit, pid, coverage, evalue, status = query_precutoff_similarity(
            seq, retries=args.max_retries
        )

        # Update the row in all_rows (same object via row_by_key)
        target_row = row_by_key[key]
        target_row["best_precutoff_hit"]      = hit
        target_row["best_precutoff_pident"]   = f"{pid:.4f}" if hit else ""
        target_row["best_precutoff_coverage"] = f"{coverage:.4f}" if hit else ""
        target_row["best_precutoff_evalue"]   = f"{evalue:.4g}" if hit else ""
        target_row["precutoff_search_status"] = status

        if key in MANUAL_SPLIT_OVERRIDES:
            target_row["training_or_test_split"] = MANUAL_SPLIT_OVERRIDES[key]
            target_row["split_assignment_reason"] = (
                "documented_structure_exception: manual override applied after retry"
            )
            label = f"→ {MANUAL_SPLIT_OVERRIDES[key]} (manual override)"
            n_resolved += 1
        elif status == "no_similar_hit":
            target_row["training_or_test_split"] = "test_set_B_candidate"
            target_row["split_assignment_reason"] = (
                f"no_pre_cutoff_hit_at_{TESTB_MAX_PIDENT:.0%}_identity_"
                f"and_{TESTB_MIN_COVERAGE:.0%}_coverage_retry"
            )
            label = f"Test B ({pid:.1%} id, {coverage:.1%} cov)"
            n_resolved += 1
        elif status == "similar_hit":
            target_row["training_or_test_split"] = "test_set_A_candidate"
            target_row["split_assignment_reason"] = (
                f"pre_cutoff_hit_at_or_above_{TESTB_MAX_PIDENT:.0%}_identity_"
                f"and_{TESTB_MIN_COVERAGE:.0%}_coverage_retry"
            )
            label = f"Test A ({pid:.1%} id, {coverage:.1%} cov)"
            n_resolved += 1
        elif status == "incomplete_search":
            # Conservative: leave as post_cutoff_candidate for manual review
            target_row["training_or_test_split"] = "post_cutoff_candidate"
            target_row["split_assignment_reason"] = "precutoff_similarity_incomplete_search"
            label = "⚠ incomplete search — left pending"
            n_incomplete += 1
        else:  # still api_error
            target_row["training_or_test_split"] = "post_cutoff_candidate"
            target_row["split_assignment_reason"] = "precutoff_similarity_unknown"
            label = "⚠ still api_error — left pending"
            n_still_error += 1

        best_str = f"best hit: {hit or 'none':20s}  id={pid:.3f}  cov={coverage:.3f}"
        print(f"[{status}]  {best_str}  {label}", flush=True)
        time.sleep(args.sleep)

    # ── Mutual exclusivity check ───────────────────────────────────────────────
    testa_keys = {f"{r['PDB_ID']}_{r['chain_ID']}"
                  for r in all_rows if r.get("training_or_test_split") == "test_set_A_candidate"}
    testb_keys = {f"{r['PDB_ID']}_{r['chain_ID']}"
                  for r in all_rows if r.get("training_or_test_split") == "test_set_B_candidate"}
    overlap = testa_keys & testb_keys
    if overlap:
        print(f"\nFATAL: {len(overlap)} protein(s) appear in both Test A and Test B "
              f"after retry: {overlap}", file=sys.stderr)
        sys.exit(1)
    print(f"\n  Mutual exclusivity check: Test A={len(testa_keys)}, "
          f"Test B={len(testb_keys)}, overlap=0  ✓")

    # ── Write updated CSV ──────────────────────────────────────────────────────
    # Ensure any new fields are present (fieldnames from the original CSV should
    # already include the clustered columns, but guard anyway).
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Written: {out_path} ({len(all_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    still_pending = sum(
        1 for r in all_rows if r.get("training_or_test_split") == "post_cutoff_candidate"
    )
    print(f"""
=== Retry summary ===
  Rows attempted    : {len(retry_rows)}
  Resolved          : {n_resolved}  (assigned to Test A or Test B)
  Incomplete search : {n_incomplete}  (still pending — check manually)
  Still api_error   : {n_still_error}  (still pending — check manually)
  Total pending     : {still_pending}
""")
    if still_pending:
        pending = [
            r for r in all_rows
            if r.get("training_or_test_split") == "post_cutoff_candidate"
        ]
        print("  Rows still pending manual Test A/B assignment:")
        for r in pending:
            print(f"    {r['PDB_ID']}_{r['chain_ID']}  "
                  f"({r.get('precutoff_search_status', 'unknown')})")
        print()
        print("  Manual search: https://www.rcsb.org/search")
        print("  (Sequence tab → filter by release date ≤ 2021-09-30)")
    else:
        print("  All api_error rows resolved  ✓")


if __name__ == "__main__":
    main()
