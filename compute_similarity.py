#!/usr/bin/env python3
"""
compute_similarity.py
Stage 1 sequence clustering and pre-cutoff similarity labelling.

Two tasks
---------
  1. Within-dataset clustering
       Cluster all passing candidates at 30% sequence identity / 80% coverage
       using MMseqs2 easy-cluster.  Assigns a cluster_id and cluster_rep to
       each row in master_candidates.csv.

  2. Pre-cutoff PDB similarity search
       For each candidate, query the RCSB sequence-similarity search API
       (filtered to entries released <= 2021-09-30) and read identity and
       coverage from the returned alignment metadata.  A post-cutoff candidate
       is Test B only when no pre-cutoff alignment reaches both 30% identity
       and 80% coverage of the post-cutoff candidate sequence; the remaining
       post-cutoff candidates form the mutually exclusive Test A set.

       8EXF_B is a documented structural exception: it is forced into Test B
       even though its sequence is similar to pre-cutoff BCCIPbeta.  The actual
       high-similarity alignment is still recorded for auditability.

  3. Dev/val split (Issue 6 fix)
       Pre-cutoff candidates (screening_status == "dev_screen_pass") are
       clustered together with post-cutoff candidates.  After clustering, a
       cluster-aware 80/20 split assigns ~20% of pre-cutoff clusters to
       "validation_candidate" and the rest to "development_candidate".  This
       prevents sequence leakage across dev/val boundaries.

Output
------
  master_candidates_clustered.csv   — original CSV + cluster_id, cluster_rep,
                                      best_precutoff_hit, identity, coverage,
                                      updated training_or_test_split

Usage
-----
  python compute_similarity.py --csv output/master_candidates.csv

Requirements
------------
  MMseqs2 must be on PATH.  Install with:
    conda install -c conda-forge -c bioconda mmseqs2
  or download from https://github.com/soedinglab/MMseqs2/releases

  Python packages: requests
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# Keep progress output reliable in Windows terminals whose inherited encoding
# cannot represent symbols already used by this script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ─────────────────────────────────────────────────────────────────
CUTOFF_DATE       = "2021-09-30"
TESTB_MAX_PIDENT  = 0.30   # paired with TESTB_MIN_COVERAGE below
CLUSTER_PIDENT    = 0.30   # 30% identity for within-dataset clustering
CLUSTER_COV       = 0.80   # 80% sequence coverage (longer sequence)
TESTB_MIN_COVERAGE = 0.80  # fraction of the post-cutoff candidate covered
MAX_SEARCH_RESULTS = 1000  # inspect alignments, not the aggregate result score

# *** REQUIRE EXPLICIT SCIENTIFIC APPROVAL before changing ***
# Fraction of pre-cutoff (dev) clusters assigned to validation.
# Every (1/VAL_FRACTION)-th cluster (1-indexed, sorted by cluster_id) → validation.
VAL_FRACTION = 0.20   # 20% validation, 80% development

RCSB_SEARCH  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_ENTRY_REST = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
API_SLEEP    = 0.25   # seconds between RCSB sequence-search calls

# Statuses accepted as "passing" candidates that should be clustered.
# Issue 4 fix: dev_screen_pass rows (pre-cutoff) must be clustered so the
# dev/val split can be performed cluster-aware.
PASS_STATUSES = {"auto_screen_pass", "manual_screen_pass", "dev_screen_pass"}

# ── Manual inclusions ─────────────────────────────────────────────────────────
# Proteins that must appear in the dataset regardless of what build_dataset.py
# retrieved.  Fetched from RCSB if not already present in the input CSV.
# Verify chain_ID against the PDB entry before running.
MANUAL_INCLUSIONS: list[dict] = [
    {"PDB_ID": "8EXF", "chain_ID": "B"},   # BCCIPα (entity 2)
    {"PDB_ID": "8URV", "chain_ID": "A"},   # pro-IL-18
]

# ── Manual split overrides ────────────────────────────────────────────────────
# Force specific PDB_chain keys to a split label after sequence-similarity
# search.  Use only for documented scientific edge cases; the measured
# pre-cutoff similarity is retained in the output.
#
# 8EXF (BCCIPα): high sequence identity to pre-cutoff BCCIPβ but sufficiently
# different 3D fold to be a genuine test case → force Test Set B.
MANUAL_SPLIT_OVERRIDES: dict[str, str] = {
    "8EXF_B": "test_set_B_candidate",
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "stage1-clustering/1.0 (NTU research)"


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_mmseqs2(requested_path: str | None = None) -> str:
    """Return the native MMseqs2 executable, including the usual Windows install."""
    candidates: list[Path] = []
    if requested_path:
        candidates.append(Path(requested_path).expanduser())

    discovered = shutil.which("mmseqs")
    if discovered:
        candidates.append(Path(discovered))

    candidates.append(
        Path.home() / "Tools" / "MMseqs2" / "mmseqs" / "bin" / "mmseqs.exe"
    )

    for candidate in candidates:
        # The Windows .bat wrapper can make result export fail. Prefer its native exe.
        if candidate.suffix.lower() in {".bat", ".cmd"}:
            native_exe = candidate.parent / "bin" / "mmseqs.exe"
            if native_exe.is_file():
                return str(native_exe)
        if candidate.is_file():
            return str(candidate)

    print(
        "\nERROR: MMseqs2 was not found. Expected it at:\n"
        f"  {Path.home() / 'Tools' / 'MMseqs2' / 'mmseqs' / 'bin' / 'mmseqs.exe'}\n"
        "You can also provide its location with --mmseqs-exe.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def check_mmseqs2(mmseqs_exe: str) -> None:
    result = subprocess.run(
        [mmseqs_exe, "version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 could not start: {result.stderr.strip()}")
    version = result.stdout.strip() or result.stderr.strip()
    print(f"[MMseqs2] {mmseqs_exe}  version {version}", flush=True)


def run(cmd: list[str], desc: str) -> None:
    """Run a shell command, streaming last few lines of output."""
    print(f"\n[RUN] {desc}", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    if proc.stdout.strip():
        for line in proc.stdout.strip().splitlines()[-5:]:
            print(f"  {line}", flush=True)


def write_fasta(rows: list[dict], path: Path) -> None:
    """Write candidate sequences to a FASTA file."""
    with open(path, "w") as fh:
        for r in rows:
            pdb   = r["PDB_ID"]
            chain = r["chain_ID"]
            seq   = r.get("sequence", "").strip()
            if seq:
                fh.write(f">{pdb}_{chain}\n{seq}\n")
    n = sum(1 for r in rows if r.get("sequence"))
    print(f"  Wrote {path.name} ({n} sequences)", flush=True)


def parse_mmseqs_cluster(tsv_path: Path) -> dict[str, str]:
    """
    Parse MMseqs2 easy-cluster TSV output (rep\\tmember per line).
    Returns {member_id: rep_id}.
    """
    mapping: dict[str, str] = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                mapping[member] = rep
    return mapping


def fetch_sequence(pdb_id: str, chain_id: str) -> str:
    """
    Fetch the canonical sequence for a given PDB entry + chain from RCSB GraphQL.
    Returns the sequence string, or "" on failure.
    The entity matching chain_id is found by checking polymer_entity_instances.
    """
    GQL = """
    query($id: String!) {
      entry(entry_id: $id) {
        polymer_entities {
          entity_poly { pdbx_seq_one_letter_code_can }
          polymer_entity_instances {
            rcsb_polymer_entity_instance_container_identifiers { auth_asym_id }
          }
        }
      }
    }
    """
    try:
        resp = SESSION.post(RCSB_GRAPHQL,
                            json={"query": GQL, "variables": {"id": pdb_id}},
                            timeout=30)
        resp.raise_for_status()
        entities = ((resp.json().get("data") or {})
                    .get("entry") or {}).get("polymer_entities") or []
        for ent in entities:
            chains = [
                inst["rcsb_polymer_entity_instance_container_identifiers"]["auth_asym_id"]
                for inst in (ent.get("polymer_entity_instances") or [])
            ]
            if chain_id in chains:
                seq = (ent.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or ""
                return seq.replace("\n", "").replace(" ", "")
    except Exception as exc:
        print(f"  [fetch_sequence] {pdb_id}_{chain_id}: {exc}", file=sys.stderr)
    return ""


def fetch_release_date(pdb_id: str) -> str:
    """Fetch the initial PDB release date for a priority-row fallback."""
    try:
        resp = SESSION.get(RCSB_ENTRY_REST.format(pdb_id=pdb_id), timeout=30)
        resp.raise_for_status()
        date = (resp.json().get("rcsb_accession_info") or {}).get(
            "initial_release_date", ""
        )
        return date[:10]
    except Exception as exc:
        print(f"  [fetch_release_date] {pdb_id}: {exc}", file=sys.stderr)
        return ""


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
                    # The query is the post-cutoff candidate.  Query coverage
                    # prevents a nearly complete candidate match to one domain
                    # of a longer pre-cutoff protein being mislabelled novel.
                    "coverage": q_span / q_len,
                    "query_coverage": q_span / q_len,
                    "subject_coverage": s_span / s_len,
                    "evalue": float(match.get("evalue") or 0.0),
                })
    return alignments


def query_precutoff_similarity(
    sequence: str, retries: int = 3
) -> tuple[str, float, float, float, str]:
    """
    Query RCSB sequence-similarity search for the best pre-cutoff PDB hit.

    Returns (best_entity_id, identity, coverage, evalue, status), where
    status is one of:
      "similar_hit"       — at least one pre-cutoff alignment reaches both
                            TESTB_MAX_PIDENT and TESTB_MIN_COVERAGE
      "no_similar_hit"    — valid response, but no alignment reaches both
                            thresholds (eligible for ordinary Test B)
      "incomplete_search" — RCSB returned more matches than this run inspected;
                            do not assign Test B conservatively
      "api_error"         — similarity is unknown; do not assign A or B

    Important: result["score"] is an RCSB ranking score, not sequence identity.
    Identity and alignment coordinates are read from services[].nodes[].
    match_context[] in a verbose Search API response.
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
            resp = SESSION.post(RCSB_SEARCH, json=payload, timeout=45)
            if not resp.ok:
                print(f"    [seqsearch] HTTP {resp.status_code}: {resp.text[:200]}",
                      file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            data = resp.json()          # raises ValueError if body is empty/invalid
            results = data.get("result_set", [])
            total_count = int(data.get("total_count") or len(results))
            alignments = [
                alignment
                for result in results
                for alignment in _alignment_metrics(result)
            ]
            if not alignments:
                return ("", 0.0, 0.0, 0.0, "no_similar_hit")

            # Report the strongest global-enough alignment.  A high-identity
            # short motif does not disqualify a candidate from Test B.
            global_hits = [
                a for a in alignments
                if a["coverage"] >= TESTB_MIN_COVERAGE
            ]
            if not global_hits:
                best_local = max(
                    alignments,
                    key=lambda a: (a["coverage"], a["identity"]),
                )
                status = (
                    "incomplete_search"
                    if total_count > len(results)
                    else "no_similar_hit"
                )
                return (
                    best_local["hit"], best_local["identity"],
                    best_local["coverage"], best_local["evalue"],
                    status,
                )

            best = max(global_hits, key=lambda a: (a["identity"], a["coverage"]))
            status = (
                "similar_hit"
                if best["identity"] >= TESTB_MAX_PIDENT
                else "no_similar_hit"
            )
            return (
                best["hit"], best["identity"], best["coverage"],
                best["evalue"], status,
            )
        except Exception as exc:
            print(f"    [seqsearch] attempt {attempt+1} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    # All retries exhausted — we do NOT know the true identity
    return ("", 0.0, 0.0, 0.0, "api_error")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 sequence clustering")
    parser.add_argument("--csv", required=True,
                        help="Path to master_candidates.csv from build_dataset.py")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (defaults to same dir as --csv)")
    parser.add_argument("--tmp-dir", default=None,
                        help="Temp dir for MMseqs2 scratch files (default: system temp)")
    parser.add_argument("--mmseqs-exe", default=None,
                        help="Path to native mmseqs executable (normally auto-detected)")
    args = parser.parse_args()

    mmseqs_exe = resolve_mmseqs2(args.mmseqs_exe)
    check_mmseqs2(mmseqs_exe)

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    print(f"\n=== Loading {csv_path.name} ===")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader     = csv.DictReader(fh)
        all_rows   = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # ── Inject manual inclusions if missing ───────────────────────────────────
    existing_keys = {f"{r['PDB_ID']}_{r['chain_ID']}" for r in all_rows}
    for entry in MANUAL_INCLUSIONS:
        key = f"{entry['PDB_ID']}_{entry['chain_ID']}"
        if key in existing_keys:
            print(f"  [manual] {key} already in CSV — skipping injection", flush=True)
            continue
        print(f"  [manual] {key} not in CSV — fetching from RCSB …", flush=True)
        seq = fetch_sequence(entry["PDB_ID"], entry["chain_ID"])
        release_date = fetch_release_date(entry["PDB_ID"])
        if not seq:
            print(f"  [manual] WARNING: could not fetch sequence for {key}; "
                  f"verify PDB ID and chain ID.", file=sys.stderr)
        stub = {fn: "" for fn in fieldnames}
        stub.update({
            "PDB_ID":                 entry["PDB_ID"],
            "chain_ID":               entry["chain_ID"],
            "sequence":               seq,
            "length":                 str(len(seq)) if seq else "",
            "release_date":           release_date,
            "screening_status":       "manual_screen_pass",
            "training_or_test_split": "post_cutoff_candidate",
            "priority_case":          "YES",
            "review_notes":           (
                "Priority case injected by compute_similarity.py. Re-run "
                "build_dataset.py to populate its full biological metadata."
            ),
        })
        all_rows.append(stub)
        existing_keys.add(key)
        print(f"  [manual] {key} injected (seq len={len(seq)})", flush=True)

    # Issue 4 fix: include dev_screen_pass rows in clustering so that the
    # cluster-aware dev/val split (Issue 6) can be performed correctly.
    pass_rows = [r for r in all_rows if r.get("screening_status") in PASS_STATUSES]
    post_rows = [r for r in pass_rows if r.get("screening_status") != "dev_screen_pass"]
    dev_rows_all = [r for r in pass_rows if r.get("screening_status") == "dev_screen_pass"]
    print(
        f"  Total rows: {len(all_rows)} | "
        f"Passing rows to cluster: {len(pass_rows)} "
        f"(post-cutoff: {len(post_rows)}, pre-cutoff/dev: {len(dev_rows_all)})"
    )

    if not pass_rows:
        print("ERROR: No passing rows found.  Run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    pass_keys = [f"{r['PDB_ID']}_{r['chain_ID']}" for r in pass_rows]
    duplicate_keys = sorted({key for key in pass_keys if pass_keys.count(key) > 1})
    if duplicate_keys:
        print(
            "ERROR: duplicate passing PDB-chain rows would make split membership "
            f"ambiguous: {duplicate_keys}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing_priority = sorted(
        f"{e['PDB_ID']}_{e['chain_ID']}" for e in MANUAL_INCLUSIONS
        if f"{e['PDB_ID']}_{e['chain_ID']}" not in set(pass_keys)
    )
    if missing_priority:
        print(f"ERROR: required priority rows are missing: {missing_priority}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir, prefix="mmseqs_stage1_") as tmp:
        tmp         = Path(tmp)
        cand_fasta  = tmp / "candidates.fasta"
        sequence_db = tmp / "candidate_db"
        cluster_db  = tmp / "cluster_db"
        cluster_tsv = tmp / "cluster_cluster.tsv"

        # ── Write candidate FASTA ──────────────────────────────────────────────
        print("\n=== Writing candidate FASTA ===")
        write_fasta(pass_rows, cand_fasta)

        # ── Task 1: Within-dataset clustering ─────────────────────────────────
        print("\n=== Task 1: Within-dataset clustering (MMseqs2) ===")
        print(f"  Parameters: --min-seq-id {CLUSTER_PIDENT}  -c {CLUSTER_COV}  --cov-mode 0")
        run(
            [mmseqs_exe, "createdb", str(cand_fasta), str(sequence_db), "-v", "1"],
            "create MMseqs2 sequence database",
        )
        run(
            [
                mmseqs_exe, "cluster",
                str(sequence_db),
                str(cluster_db),
                str(tmp / "mmseqs_tmp"),
                "--min-seq-id", str(CLUSTER_PIDENT),
                "-c",           str(CLUSTER_COV),
                "--cov-mode",   "0",   # coverage of longer sequence
                "--cluster-mode", "1", # connected-component clustering
                "-v", "1",
            ],
            "within-dataset clustering",
        )
        run(
            [
                mmseqs_exe, "createtsv",
                str(sequence_db), str(sequence_db), str(cluster_db),
                str(cluster_tsv), "-v", "1",
            ],
            "export cluster membership TSV",
        )

        if not cluster_tsv.is_file():
            raise RuntimeError(f"MMseqs2 did not create expected output: {cluster_tsv}")

        cluster_map = parse_mmseqs_cluster(cluster_tsv)
        n_clusters  = len(set(cluster_map.values()))
        print(f"  {len(cluster_map)} sequences → {n_clusters} clusters at ≥30% identity")

    # ── Task 2: Pre-cutoff similarity via RCSB sequence search ────────────────
    # Only search for post-cutoff candidates (not dev rows — they ARE pre-cutoff).
    print(f"\n=== Task 2: Pre-cutoff PDB similarity search "
          f"({len(post_rows)} post-cutoff candidates via RCSB API) ===")
    print("  Querying RCSB and inspecting explicit alignment identity + coverage …")
    print(f"  Expected time: ~{len(post_rows) * API_SLEEP / 60:.1f}–"
          f"{len(post_rows) * 2 / 60:.1f} min\n")

    # {key: (best_hit_id, identity, coverage, evalue, status)}
    search_hits: dict[str, tuple[str, float, float, float, str]] = {}
    for i, row in enumerate(pass_rows):
        key = f"{row['PDB_ID']}_{row['chain_ID']}"

        release_date = row.get("release_date", "")[:10]
        current_split = row.get("training_or_test_split", "")
        screening_status = row.get("screening_status", "")

        # Pre-cutoff entries (dev rows) don't need the similarity search —
        # they are by definition pre-cutoff.
        is_pre_cutoff = (
            screening_status == "dev_screen_pass"
            or (release_date and release_date <= CUTOFF_DATE)
            or current_split == "development_candidate"
        )
        if is_pre_cutoff:
            search_hits[key] = ("", 0.0, 0.0, 0.0, "not_applicable_pre_cutoff")
            continue

        seq = row.get("sequence", "").strip()
        if not seq:
            search_hits[key] = ("", 0.0, 0.0, 0.0, "api_error")
            continue
        hit, pid, coverage, evalue, status = query_precutoff_similarity(seq)
        search_hits[key] = (hit, pid, coverage, evalue, status)
        if status in {"api_error", "incomplete_search"}:
            label = f"⚠ {status.replace('_', ' ')} — needs review"
        elif key in MANUAL_SPLIT_OVERRIDES:
            label = f"{MANUAL_SPLIT_OVERRIDES[key]} (documented exception)"
        elif status == "no_similar_hit":
            label = f"Test B ({pid:.1%} identity, {coverage:.1%} coverage)"
        else:
            label = f"Test A ({pid:.1%} identity, {coverage:.1%} coverage)"
        print(f"  [{i+1:3d}/{len(pass_rows)}] {key:12s}  best hit: {hit or 'none':20s}  "
              f"id={pid:.3f}  cov={coverage:.3f}  [{status}] → {label}", flush=True)
        time.sleep(API_SLEEP)

    n_testb   = sum(1 for _, _, _, _, s in search_hits.values()
                    if s == "no_similar_hit")
    n_errors  = sum(
        1 for _, _, _, _, s in search_hits.values()
        if s in {"api_error", "incomplete_search"}
    )
    print(f"\n  No pre-cutoff hit at >= {TESTB_MAX_PIDENT:.0%} identity and "
          f">= {TESTB_MIN_COVERAGE:.0%} coverage: "
          f"{n_testb} candidates → Test Set B")
    if n_errors:
        print(f"  ⚠  {n_errors} candidate(s) had API errors — split label left as "
              f"'post_cutoff_candidate' and flagged in 'precutoff_search_status' column.",
              file=sys.stderr)

    # ── Annotate rows ──────────────────────────────────────────────────────────
    print("\n=== Annotating CSV ===")

    cluster_by_key: dict[str, str] = {}
    for row in pass_rows:
        key = f"{row['PDB_ID']}_{row['chain_ID']}"
        cluster_by_key[key] = cluster_map.get(key, key)

    rep_to_id: dict[str, str] = {}
    for rep in sorted(set(cluster_by_key.values())):
        rep_to_id[rep] = f"cluster_{len(rep_to_id)+1:04d}"

    new_fields = [
        "cluster_id", "cluster_rep",
        "best_precutoff_hit", "best_precutoff_pident",
        "best_precutoff_coverage", "best_precutoff_evalue",
        "precutoff_search_status", "split_assignment_reason",
    ]

    out_rows = []
    for row in all_rows:
        key     = f"{row['PDB_ID']}_{row['chain_ID']}"
        is_pass = row.get("screening_status") in PASS_STATUSES

        if is_pass:
            rep      = cluster_by_key.get(key, key)
            clust_id = rep_to_id.get(rep, "")
            hit, pid, coverage, evalue, status = search_hits.get(
                key, ("", 0.0, 0.0, 0.0, "api_error")
            )

            row["cluster_id"]               = clust_id
            row["cluster_rep"]              = rep
            row["best_precutoff_hit"]       = hit
            row["best_precutoff_pident"]    = f"{pid:.4f}" if hit else ""
            row["best_precutoff_coverage"]  = f"{coverage:.4f}" if hit else ""
            row["best_precutoff_evalue"]    = f"{evalue:.4g}" if hit else ""
            row["precutoff_search_status"]  = status
            row["sequence_cluster"]         = clust_id

            current_split = row.get("training_or_test_split", "")
            release_date = row.get("release_date", "")[:10]
            screening_status = row.get("screening_status", "")

            if key in MANUAL_SPLIT_OVERRIDES:
                # Scientifically motivated override — takes precedence over
                # sequence similarity result (e.g. 8EXF: seq-similar to
                # pre-cutoff BCCIPβ but structurally distinct → Test B)
                row["training_or_test_split"] = MANUAL_SPLIT_OVERRIDES[key]
                row["split_assignment_reason"] = (
                    "documented_structure_exception: BCCIPalpha has a distinct "
                    "3D fold from pre-cutoff BCCIPbeta despite high sequence similarity"
                )
            elif status == "not_applicable_pre_cutoff" or screening_status == "dev_screen_pass":
                # Pre-cutoff rows: dev/val split assigned below after clustering.
                # Set a sentinel value — overwritten in the dev/val split pass.
                row["training_or_test_split"] = "development_candidate"
                row["split_assignment_reason"] = "released_on_or_before_cutoff_pending_dev_val_split"
            elif release_date > CUTOFF_DATE or current_split in {
                "post_cutoff_candidate",
                "test_set_A_candidate",
                "test_set_B_candidate",
            }:
                if status in {"api_error", "incomplete_search"}:
                    # Cannot confirm — leave label unchanged for manual review
                    row["training_or_test_split"] = "post_cutoff_candidate"
                    row["split_assignment_reason"] = "precutoff_similarity_unknown"
                elif status == "no_similar_hit":
                    row["training_or_test_split"] = "test_set_B_candidate"
                    row["split_assignment_reason"] = (
                        f"no_pre_cutoff_hit_at_{TESTB_MAX_PIDENT:.0%}_identity_"
                        f"and_{TESTB_MIN_COVERAGE:.0%}_coverage"
                    )
                elif status == "similar_hit":
                    row["training_or_test_split"] = "test_set_A_candidate"
                    row["split_assignment_reason"] = (
                        f"pre_cutoff_hit_at_or_above_{TESTB_MAX_PIDENT:.0%}_identity_"
                        f"and_{TESTB_MIN_COVERAGE:.0%}_coverage"
                    )
                else:
                    row["training_or_test_split"] = "post_cutoff_candidate"
                    row["split_assignment_reason"] = f"unexpected_status:{status}"
        else:
            row["cluster_id"]               = ""
            row["cluster_rep"]              = ""
            row["best_precutoff_hit"]       = ""
            row["best_precutoff_pident"]    = ""
            row["best_precutoff_coverage"]  = ""
            row["best_precutoff_evalue"]    = ""
            row["precutoff_search_status"]  = ""
            row["split_assignment_reason"]  = ""

        out_rows.append(row)

    # ── Mutual exclusivity check ───────────────────────────────────────────────
    # Test A and Test B must be disjoint.  Since every row gets exactly one
    # split label this is guaranteed by construction, but we assert it
    # explicitly as a safeguard against future bugs.
    testa_keys = {f"{r['PDB_ID']}_{r['chain_ID']}"
                  for r in out_rows if r.get("training_or_test_split") == "test_set_A_candidate"}
    testb_keys = {f"{r['PDB_ID']}_{r['chain_ID']}"
                  for r in out_rows if r.get("training_or_test_split") == "test_set_B_candidate"}
    overlap = testa_keys & testb_keys
    if overlap:
        print(f"\nFATAL: {len(overlap)} protein(s) appear in both Test A and Test B: "
              f"{overlap}", file=sys.stderr)
        sys.exit(1)

    priority_splits = {
        f"{r['PDB_ID']}_{r['chain_ID']}": r.get("training_or_test_split", "")
        for r in out_rows
        if f"{r['PDB_ID']}_{r['chain_ID']}" in {
            f"{e['PDB_ID']}_{e['chain_ID']}" for e in MANUAL_INCLUSIONS
        }
    }
    if priority_splits.get("8EXF_B") != "test_set_B_candidate":
        print("FATAL: 8EXF_B was not assigned to Test Set B.", file=sys.stderr)
        sys.exit(1)
    if priority_splits.get("8URV_A") not in {
        "test_set_A_candidate", "test_set_B_candidate"
    }:
        print("FATAL: 8URV_A was not assigned to Test Set A or B.", file=sys.stderr)
        sys.exit(1)

    test_clusters_a = {
        r.get("sequence_cluster") for r in out_rows
        if r.get("training_or_test_split") == "test_set_A_candidate"
    } - {""}
    test_clusters_b = {
        r.get("sequence_cluster") for r in out_rows
        if r.get("training_or_test_split") == "test_set_B_candidate"
    } - {""}
    cluster_overlap = sorted(test_clusters_a & test_clusters_b)
    if cluster_overlap:
        print(
            "  WARNING: Test A and Test B contain different proteins from the "
            f"same within-dataset sequence cluster: {cluster_overlap}. Review "
            "these clusters before locking the benchmark (8EXF may create an "
            "intentional exception cluster).",
            file=sys.stderr,
        )
    print(f"  Mutual exclusivity check passed "
          f"(Test A: {len(testa_keys)}, Test B: {len(testb_keys)}, overlap: 0)")

    # ── Issue 6 fix: Cluster-aware dev/val split ───────────────────────────────
    # Every (1/VAL_FRACTION)-th cluster of dev rows, sorted by cluster_id,
    # → "validation_candidate".  All others → "development_candidate".
    # This groups related sequences together so no cluster spans the boundary.
    #
    # *** REQUIRE EXPLICIT SCIENTIFIC APPROVAL to change VAL_FRACTION ***
    print(f"\n=== Issue 6: Cluster-aware dev/val split "
          f"(VAL_FRACTION={VAL_FRACTION:.0%}) ===")

    dev_rows_out = [r for r in out_rows if r.get("screening_status") == "dev_screen_pass"]
    if dev_rows_out:
        dev_cluster_reps = sorted(
            set(r["cluster_rep"] for r in dev_rows_out if r.get("cluster_rep")),
            key=lambda rep: rep_to_id.get(rep, rep),  # sort by cluster_id string
        )
        # Assign every (1/VAL_FRACTION)-th cluster to validation (1-indexed).
        # With VAL_FRACTION=0.20 this is every 5th cluster.
        val_step = round(1.0 / VAL_FRACTION)
        val_reps: set[str] = set()
        for i, rep in enumerate(dev_cluster_reps):
            if (i + 1) % val_step == 0:
                val_reps.add(rep)

        n_dev_clust_val = len(val_reps)
        n_dev_clust_train = len(dev_cluster_reps) - n_dev_clust_val
        print(f"  Pre-cutoff clusters: {len(dev_cluster_reps)} total — "
              f"{n_dev_clust_train} development, {n_dev_clust_val} validation")

        n_val_rows = 0
        n_train_rows = 0
        for r in out_rows:
            if r.get("screening_status") == "dev_screen_pass":
                rep = r.get("cluster_rep", "")
                if rep and rep in val_reps:
                    r["training_or_test_split"] = "validation_candidate"
                    r["split_assignment_reason"] = (
                        f"dev_val_cluster_split_validation"
                        f"_every_{val_step}th_cluster"
                    )
                    n_val_rows += 1
                else:
                    r["training_or_test_split"] = "development_candidate"
                    r["split_assignment_reason"] = (
                        f"dev_val_cluster_split_development"
                        f"_every_{val_step}th_cluster"
                    )
                    n_train_rows += 1
        print(f"  Dev/val rows assigned: {n_train_rows} development, "
              f"{n_val_rows} validation")
    else:
        print("  No dev_screen_pass rows found — dev/val split skipped.")

    # ── Write output CSV ───────────────────────────────────────────────────────
    out_csv = out_dir / "master_candidates_clustered.csv"
    out_fieldnames = fieldnames.copy()
    for f in new_fields:
        if f not in out_fieldnames:
            out_fieldnames.append(f)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n  Written: {out_csv} ({len(out_rows)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────────
    testa      = sum(1 for r in out_rows if r.get("training_or_test_split") == "test_set_A_candidate")
    testb      = sum(1 for r in out_rows if r.get("training_or_test_split") == "test_set_B_candidate")
    dev_train  = sum(1 for r in out_rows if r.get("training_or_test_split") == "development_candidate")
    dev_val    = sum(1 for r in out_rows if r.get("training_or_test_split") == "validation_candidate")
    pending    = sum(1 for r in out_rows if r.get("training_or_test_split") == "post_cutoff_candidate")
    n_manual   = len(MANUAL_SPLIT_OVERRIDES)

    print(f"""
=== Clustering summary ===
  Passing candidates clustered : {len(pass_rows)}
  Unique clusters (≥30% id)    : {n_clusters}
  Development set              : {dev_train}
  Validation set               : {dev_val}  ({VAL_FRACTION:.0%} of pre-cutoff clusters)
  Test Set A (temporal)        : {testa}
  Test Set B (low similarity)  : {testb}   (incl. {n_manual} documented exception(s))
  Needs review (API error)     : {pending}
  Test A ∩ Test B              : 0  ✓

  Manual split overrides applied:""")
    for k, v in MANUAL_SPLIT_OVERRIDES.items():
        print(f"    {k} → {v}")
    print(f"\n  Output: {out_csv}\n")
    if pending:
        review = [r for r in out_rows if r.get("training_or_test_split") == "post_cutoff_candidate"]
        print("  ⚠  The following rows need manual Test A/B assignment")
        print("     (RCSB sequence search failed for them — re-run retry_seqsearch.py):")
        for r in review:
            print(f"     {r['PDB_ID']}_{r['chain_ID']}")
        print()

    print("""Next steps
  1. Resolve any 'needs review' rows above — run:
       python retry_seqsearch.py --csv output/master_candidates_clustered.csv
     or manually search https://www.rcsb.org/search (Sequence tab, filter by date).
  2. Review any reported cluster shared by Test A and Test B, especially the
     documented 8EXF structural-exception cluster.
  3. Treat the 30% identity and 80% candidate-coverage thresholds as fixed
     protocol choices. Change them only using the separate validation set,
     never after inspecting Test A or Test B outcomes.
  4. Lock the mutually exclusive test sets before OpenFold3 inference.
""")


if __name__ == "__main__":
    main()
