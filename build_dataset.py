#!/usr/bin/env python3
"""
build_dataset.py
Automated Stage 1 protein dataset builder for experiment-guided structure prediction.

Pipeline
--------
  1. RCSB Search API  → bulk candidate set (post-cutoff, 80-400 aa, X-ray ≤ 2.5 Å or NMR)
  2. RCSB GraphQL API → per-entry metadata (assembly, sequence, resolution, ligands, UniProt ID)
  3. UniProt REST API → membrane topology, lipidation, GPI anchor, subcellular location, disorder
  4. OPM check       → membrane protein cross-reference
  5. Screening logic → PASS / FAIL / FLAG per criterion
  6. CSV output      → master_candidates.csv + screening_detail.csv + review_queue.csv

OpenFold3 structural training cutoff: 2021-09-30
This script builds biologically screened post-cutoff candidates.  Final,
mutually exclusive Test A/Test B labels are assigned by compute_similarity.py
after sequence clustering and comparison with all pre-cutoff PDB structures.

Usage
-----
    python build_dataset.py [--limit 2000] [--out-dir ./output]

    --limit     Max candidates to pull from RCSB Search (default 2000).
                The script stops early once TARGET_AUTO_PASSES + review-queue
                candidates have been accumulated.
    --out-dir   Directory for CSV outputs (created if absent).
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Keep progress output reliable in Windows terminals whose inherited encoding
# cannot represent symbols already used by this script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────────
CUTOFF_DATE       = "2021-09-30"          # OpenFold3 training cutoff
MIN_LEN           = 80                    # residues
MAX_LEN           = 400
MIN_RESOLVED      = 0.90                  # fraction of residues with coordinates (test sets)
MAX_XRAY_RES      = 2.5                   # Å; NMR entries are exempt (test sets)
LARGE_LIGAND_MW   = 500.0                 # Da — auto-fail threshold
MEDIUM_LIGAND_MW  = 200.0                 # Da — flag-for-review threshold
TARGET_AUTO_PASSES = 200                  # aim for this many clean INCLUDEs (post-cutoff)

# ── Dev / Val set parameters (relaxed — pre-cutoff structures) ─────────────────
# The development set is used freely during method development and debugging.
# The validation set is a held-out 20% subset used only for threshold / hyper-
# parameter selection.  Both share the same looser screening criteria below.
DEV_MIN_RESOLVED  = 0.80                  # fraction; more permissive than test sets
DEV_MAX_XRAY_RES  = 3.5                   # Å; broader resolution envelope
TARGET_DEV_PASSES = 300                   # aim for this many dev+val candidates
VAL_FRACTION      = 0.20                  # 20% of accepted pre-cutoff → val set

# Required Stage 1 biological cases.  Prepending them to the RCSB result list
# guarantees that they receive the same metadata acquisition and eight-criterion
# screening as ordinary candidates, even when --limit is small.
PRIORITY_ENTITIES: list[tuple[str, str]] = [
    ("8EXF", "2"),  # BCCIPalpha, chain B (chain A is TENT5A)
    ("8URV", "1"),  # pro-IL-18, chain A
]

RCSB_SEARCH  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/{acc}.json"
OPM_API      = "https://opm-assets.storage.googleapis.com/api/proteins/{pdb}.json"

GRAPHQL_BATCH = 50       # entries per GraphQL request
API_SLEEP     = 0.12     # seconds between requests (be polite to public APIs)

# Persistent HTTP session
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "stage1-dataset-builder/1.0 (NTU research, contact: mlee116@e.ntu.edu.sg)"

# ── Known-trivial ligand CCD IDs ───────────────────────────────────────────────
# These are buffer salts, cryo-agents, and covalently modified residues that
# are never considered "large ligands" in the folding sense.
TRIVIAL_LIGANDS = {
    # Water
    "HOH", "DOD",
    # Common ions
    "NA", "K", "MG", "CA", "MN", "FE", "FE2", "ZN", "CU", "CO", "NI",
    "CL", "BR", "F", "IOD",
    # Crystallography additives
    "SO4", "PO4", "ACT", "GOL", "EDO", "MPD", "PEG", "PGE", "BME",
    "DTT", "DMS", "TRS", "MES", "IMD", "HEPES", "EPE", "ACY",
    # Modified amino acid residues (covalently part of the chain)
    "KCX",   # carbamylated Lys (e.g. 9HPT)
    "MSE",   # selenomethionine
    "SEP",   # phosphoserine
    "TPO",   # phosphothreonine
    "PTR",   # phosphotyrosine
    "CME",   # carboxymethyl Cys
    "CSO",   # S-hydroxycysteine
    "OCS",   # cysteinesulfonic acid
    "HIC",   # 4-methyl-histidine
    "MLY",   # N-dimethyl-lysine
    "M3L",   # trimethyl-lysine
    "LYZ",   # 5-hydroxylysine
}

# ── UniProt annotation patterns signalling membrane / lipid anchoring ──────────
LIPID_FEATURE_TYPES = {
    "LIPID",                          # generic lipid attachment
    "LIPIDATION",
    "LIPID MOIETY-BINDING REGION",
}
GPI_PATTERNS = re.compile(r"gpi.anch|gpi-anch|omega.site|omega-site", re.I)

LIPID_KEYWORDS = {
    "Lipoprotein", "GPI-anchor", "Lipid anchor",
    "Myristoylation", "Palmitoylation", "Prenylation",
    "N-myristoylation", "S-palmitoylation",
    "Geranylgeranylation", "Farnesylation",
}
MEMBRANE_SUBLOC_TOKENS = {
    "cell membrane", "plasma membrane", "membrane",
    "integral component of membrane",
    "cell surface",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. RCSB SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _build_method_group() -> dict:
    """OR group matching X-RAY DIFFRACTION or SOLUTION NMR."""
    return {
        "type": "group",
        "logical_operator": "or",
        "nodes": [
            {
                "type": "terminal", "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "X-RAY DIFFRACTION",
                },
            },
            {
                "type": "terminal", "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "SOLUTION NMR",
                },
            },
        ],
    }


def rcsb_search(limit: int) -> list[tuple[str, str]]:
    """
    Query RCSB Search API for POST-cutoff candidates.
    Returns list of (entry_id, entity_id) pairs.

    Hard filters applied here (all further biological checks in steps 3-5):
      - initial_release_date > cutoff  (post-cutoff structures)
      - polymer type = Protein
      - sequence length [80, 400]
      - experimental method = X-RAY DIFFRACTION or SOLUTION NMR
    """
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater",
                        "value": f"{CUTOFF_DATE}T00:00:00Z",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {
                            "from": MIN_LEN, "to": MAX_LEN,
                            "include_lower": True, "include_upper": True,
                        },
                    },
                },
                _build_method_group(),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
        },
    }

    print(f"[Search] Querying RCSB post-cutoff (limit={limit}) …", flush=True)
    return _execute_search(payload, label="post-cutoff")


def rcsb_search_dev(limit: int) -> list[tuple[str, str]]:
    """
    Query RCSB Search API for PRE-cutoff candidates (dev / val sets).
    Returns list of (entry_id, entity_id) pairs.

    Hard filters:
      - initial_release_date <= cutoff  (pre-cutoff structures)
      - polymer type = Protein
      - sequence length [80, 400]
      - experimental method = X-RAY DIFFRACTION or SOLUTION NMR
    Resolution and resolved-fraction checks are deferred to screen_dev(),
    which uses the relaxed DEV_MAX_XRAY_RES / DEV_MIN_RESOLVED thresholds.
    """
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less_or_equal",
                        "value": f"{CUTOFF_DATE}T23:59:59Z",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {
                            "from": MIN_LEN, "to": MAX_LEN,
                            "include_lower": True, "include_upper": True,
                        },
                    },
                },
                _build_method_group(),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
        },
    }

    print(f"[Search-Dev] Querying RCSB pre-cutoff (limit={limit}) …", flush=True)
    return _execute_search(payload, label="pre-cutoff")


def _execute_search(payload: dict, label: str) -> list[tuple[str, str]]:
    """Submit a prepared RCSB Search payload and parse (entry_id, entity_id) pairs."""
    resp = SESSION.post(RCSB_SEARCH, json=payload, timeout=60)
    if not resp.ok:
        print(f"[Search-{label}] HTTP {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
    resp.raise_for_status()
    data = resp.json()

    out = []
    for r in data.get("result_set", []):
        ident = r.get("identifier", "")   # e.g. "7PNR_1"
        parts = ident.split("_")
        if len(parts) >= 2:
            out.append((parts[0].upper(), parts[1]))
        elif parts:
            out.append((parts[0].upper(), "1"))

    total = data.get("total_count", "?")
    print(f"[Search-{label}] {len(out)} candidates returned (total matching RCSB: {total})",
          flush=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. RCSB GRAPHQL  — batch metadata fetch
# ══════════════════════════════════════════════════════════════════════════════

# rcsb_polymer_instance_feature has unstable field names across RCSB GraphQL
# schema versions and is omitted here.  Resolved fraction is instead computed
# from deposited_polymer_monomer_count (entry-level ATOM records) vs
# rcsb_sample_sequence_length (full canonical sequence length), which is
# reliable for monomeric proteins and requires no extra API calls.
_GQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    exptl { method }
    refine { ls_d_res_high }
    rcsb_entry_info {
      nonpolymer_entity_count
      deposited_polymer_monomer_count
    }
    assemblies {
      rcsb_assembly_info {
        assembly_id
        polymer_entity_count_protein
        polymer_entity_instance_count_protein
        modeled_polymer_monomer_count
        polymer_monomer_count
      }
    }
    polymer_entities {
      rcsb_id
      entity_poly {
        pdbx_seq_one_letter_code_can
        rcsb_sample_sequence_length
      }
      rcsb_polymer_entity { pdbx_description }
      rcsb_entity_source_organism { scientific_name }
      uniprots { rcsb_id }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers { auth_asym_id }
        rcsb_polymer_instance_feature_summary {
          type
          count
          coverage
          maximum_length
        }
      }
    }
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp { id name formula formula_weight }
      }
    }
  }
}
"""


def _gql_batch(entry_ids: list[str]) -> list[dict]:
    payload = {"query": _GQL_QUERY, "variables": {"ids": entry_ids}}
    resp = SESSION.post(RCSB_GRAPHQL, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    # GraphQL returns HTTP 200 even on schema errors; surface them clearly.
    if "errors" in body:
        for err in body["errors"][:3]:
            print(f"  [GraphQL] schema error: {err.get('message', err)}", file=sys.stderr)
    entries = (body.get("data") or {}).get("entries") or []
    return entries


def fetch_all_entries(entry_ids: list[str]) -> dict[str, dict]:
    """Return {entry_id: entry_dict} for every requested ID."""
    unique = list(dict.fromkeys(entry_ids))
    result: dict[str, dict] = {}
    n = len(unique)
    for i in range(0, n, GRAPHQL_BATCH):
        batch = unique[i : i + GRAPHQL_BATCH]
        print(f"  [GraphQL] {i+1}–{i+len(batch)} / {n} entries …", flush=True)
        try:
            entries = _gql_batch(batch)
            if not entries and i == 0:
                print("  [GraphQL] WARNING: first batch returned 0 entries. "
                      "Check stderr for schema errors.", file=sys.stderr)
            for entry in entries:
                result[entry["rcsb_id"]] = entry
        except Exception as exc:
            print(f"  [GraphQL] batch error: {exc}", file=sys.stderr)
        time.sleep(API_SLEEP)
    print(f"  [GraphQL] fetched metadata for {len(result)} entries", flush=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIPROT
# ══════════════════════════════════════════════════════════════════════════════

def fetch_uniprot(acc: str) -> Optional[dict]:
    try:
        resp = SESSION.get(UNIPROT_REST.format(acc=acc), timeout=15)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def parse_uniprot(data: Optional[dict]) -> dict:
    """
    Return a dict of bool flags derived from UniProt annotations.
    Checks transmembrane topology, lipidation, GPI anchor,
    cell-surface subcellular location, and annotated disordered regions.
    """
    flags = {
        "has_transmembrane": False,
        "has_lipidation":    False,
        "has_gpi_anchor":    False,
        "is_membrane_assoc": False,
        "has_disorder_annot": False,
        "subloc_values":     [],
        "available":         False,
    }
    if not data:
        return flags

    flags["available"] = True

    for feat in data.get("features", []):
        ftype = feat.get("type", "").upper()
        fdesc = feat.get("description", "").lower()

        if ftype == "TRANSMEMBRANE":
            flags["has_transmembrane"] = True
        if ftype in LIPID_FEATURE_TYPES or ftype == "LIPID":
            flags["has_lipidation"] = True
        if ftype == "REGION" and GPI_PATTERNS.search(fdesc):
            flags["has_gpi_anchor"] = True
        if ftype == "REGION" and "disordered" in fdesc:
            flags["has_disorder_annot"] = True

    for kw in data.get("keywords", []):
        name = kw.get("name", "")
        if name in LIPID_KEYWORDS:
            flags["has_lipidation"] = True
        if "GPI" in name.upper() or "gpi" in name.lower():
            flags["has_gpi_anchor"] = True

    for loc_block in data.get("subcellularLocations", []):
        for loc in loc_block.get("locations", []):
            val = loc.get("value", "").lower()
            flags["subloc_values"].append(val)
            if any(tok in val for tok in MEMBRANE_SUBLOC_TOKENS):
                flags["is_membrane_assoc"] = True

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPM CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_opm(pdb_id: str) -> bool:
    """Return True if this entry exists in the OPM membrane protein database."""
    try:
        resp = SESSION.get(OPM_API.format(pdb=pdb_id.lower()), timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPERS — resolved fraction, ligand classification
# ══════════════════════════════════════════════════════════════════════════════

def count_heavy_atoms(formula: str) -> int:
    """Count non-hydrogen atoms in a CCD formula string, e.g. 'C21 H27 N7 O14 P2'."""
    total = 0
    for element, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if element == "H":
            continue
        total += int(count_str) if count_str else 1
    return total


def resolved_fraction(
    entity: dict,
    entry: dict,
    chain_id: Optional[str] = None,
) -> float:
    """
    Estimate resolved fraction for a monomeric polymer entity.

    Prefer the selected chain's RCSB UNOBSERVED_RESIDUE_XYZ coverage.  This is
    the correct chain-level measure for both ordinary monomers and priority
    cases embedded in multi-chain structures.  Assembly-level modeled/total
    residue counts are used only when chain annotations are unavailable.
    """
    instances = entity.get("polymer_entity_instances") or []
    selected_instance = None
    if chain_id:
        selected_instance = next(
            (
                instance for instance in instances
                if (
                    instance.get(
                        "rcsb_polymer_entity_instance_container_identifiers"
                    ) or {}
                ).get("auth_asym_id") == chain_id
            ),
            None,
        )
    if selected_instance is None and instances:
        selected_instance = instances[0]

    if selected_instance is not None:
        summaries = selected_instance.get("rcsb_polymer_instance_feature_summary")
        if summaries is not None:
            unobserved_coverage = sum(
                float(summary.get("coverage") or 0.0)
                for summary in summaries
                if str(summary.get("type", "")).startswith("UNOBSERVED_RESIDUE")
            )
            return max(0.0, min(1.0, 1.0 - unobserved_coverage))

    assemblies = entry.get("assemblies") or []
    asm1 = next(
        (
            a for a in assemblies
            if (a.get("rcsb_assembly_info") or {}).get("assembly_id") == "1"
        ),
        assemblies[0] if assemblies else None,
    )
    info = (asm1.get("rcsb_assembly_info") or {}) if asm1 else {}
    modeled = info.get("modeled_polymer_monomer_count")
    total = info.get("polymer_monomer_count")
    if modeled is not None and total:
        return max(0.0, min(1.0, float(modeled) / float(total)))

    # Fallback for older/incomplete API records.  This is reliable only after
    # the biological assembly has been confirmed to contain one protein chain.
    seq_len = (entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length") or 0
    deposited = (entry.get("rcsb_entry_info") or {}).get("deposited_polymer_monomer_count")
    if not seq_len or deposited is None:
        return 0.0
    return max(0.0, min(1.0, deposited / seq_len))


def classify_ligands(nonpolymer_entities: list) -> tuple[str, list[str]]:
    """
    Classify non-polymer entities as PASS / FLAG / FAIL based on molecular weight
    and heavy-atom count, ignoring trivial buffer/additive molecules.

    Returns (verdict, [note, ...]).
    """
    if not nonpolymer_entities:
        return "PASS", ["no non-polymer ligands"]

    worst   = "PASS"
    notes   = []

    for ent in nonpolymer_entities:
        comp  = ((ent.get("nonpolymer_comp") or {}).get("chem_comp") or {})
        ccd   = comp.get("id", "")
        name  = comp.get("name", "")
        formula = comp.get("formula", "")

        try:
            mw = float(comp.get("formula_weight") or 0)
        except (TypeError, ValueError):
            mw = 0.0

        if ccd in TRIVIAL_LIGANDS:
            notes.append(f"{ccd} (trivial additive/ion)")
            continue

        hatoms = count_heavy_atoms(formula)
        tag = f"{ccd} ({name}; MW={mw:.0f} Da; {hatoms} heavy atoms)"

        if mw >= LARGE_LIGAND_MW or hatoms > 25:
            notes.append(f"LARGE_LIGAND: {tag}")
            worst = "FAIL"
        elif mw >= MEDIUM_LIGAND_MW or hatoms > 12:
            notes.append(f"MEDIUM_LIGAND: {tag} — confirm not essential for folding")
            if worst != "FAIL":
                worst = "FLAG"
        else:
            notes.append(f"small ligand: {tag}")

    if not notes:
        return "PASS", ["all non-polymer entities are trivial (buffers/ions)"]
    return worst, notes


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCREENING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def screen(
    entry:         dict,
    entity_id:     str,
    up_flags:      Optional[dict],
    in_opm:        bool,
    chain_id:      Optional[str] = None,
) -> dict:
    """
    Apply all eight biological criteria. Returns a result dict:
      - per-criterion verdict (PASS / FAIL / FLAG)
      - overall (INCLUDE / EXCLUDE / FLAG_REVIEW)
      - notes, flag_reasons
    """
    res = {
        "monomeric":             "UNKNOWN",
        "soluble":               "UNKNOWN",
        "length_ok":             "UNKNOWN",
        "resolved_ok":           "UNKNOWN",
        "not_disordered":        "UNKNOWN",
        "not_membrane":          "UNKNOWN",
        "no_large_ligand":       "UNKNOWN",
        "not_obligate_complex":  "UNKNOWN",
        "overall":               "UNKNOWN",
        "notes":                 [],
        "flag_reasons":          [],
    }
    notes   = res["notes"]
    flagged = res["flag_reasons"]

    # ── Find entity ────────────────────────────────────────────────────────────
    target = None
    for ent in entry.get("polymer_entities") or []:
        if ent["rcsb_id"].split("_")[-1] == entity_id:
            target = ent
            break
    if target is None:
        res["overall"] = "ERROR"
        notes.append("entity not found in GraphQL response")
        return res

    # ── (1) Monomeric / not obligate complex ──────────────────────────────────
    assemblies = entry.get("assemblies") or []
    asm1 = None
    for a in assemblies:
        if (a.get("rcsb_assembly_info") or {}).get("assembly_id") == "1":
            asm1 = a
            break
    if asm1 is None and assemblies:
        asm1 = assemblies[0]

    n_chains = (asm1.get("rcsb_assembly_info") or {}).get(
        "polymer_entity_instance_count_protein"
    ) if asm1 else None
    oligo    = ""  # oligomeric_details not in current RCSB schema

    if n_chains is None:
        res["monomeric"] = "FLAG"
        flagged.append("Assembly chain count unavailable — confirm monomeric")
    elif n_chains == 1:
        res["monomeric"] = "PASS"
        notes.append(f"Assembly 1: {n_chains} protein chain (monomeric; {oligo})")
    else:
        res["monomeric"] = "FAIL"
        notes.append(f"Assembly 1 has {n_chains} protein chains — not monomeric ({oligo})")

    res["not_obligate_complex"] = "PASS" if res["monomeric"] == "PASS" else "FAIL"

    # ── (2) Sequence length ────────────────────────────────────────────────────
    seq_len = (target.get("entity_poly") or {}).get("rcsb_sample_sequence_length") or 0
    if MIN_LEN <= seq_len <= MAX_LEN:
        res["length_ok"] = "PASS"
        notes.append(f"Length: {seq_len} residues")
    else:
        res["length_ok"] = "FAIL"
        notes.append(f"Length {seq_len} outside [{MIN_LEN}, {MAX_LEN}]")

    # ── (3) Resolved fraction + resolution ────────────────────────────────────
    res_frac = resolved_fraction(target, entry, chain_id)
    methods  = [e.get("method", "") for e in (entry.get("exptl") or [])]
    method   = methods[0] if methods else ""
    is_nmr   = "NMR" in method.upper()

    xray_ok = True
    resolution_val = None
    if not is_nmr:
        for ref in (entry.get("refine") or []):
            v = ref.get("ls_d_res_high")
            if v is not None:
                resolution_val = float(v)
                break
        if resolution_val is not None and resolution_val > MAX_XRAY_RES:
            xray_ok = False
            notes.append(f"X-ray resolution {resolution_val:.2f} Å > {MAX_XRAY_RES} Å limit")

    if res_frac >= MIN_RESOLVED and xray_ok:
        res["resolved_ok"] = "PASS"
        notes.append(f"Resolved fraction: {res_frac:.4f}" + (f"; resolution {resolution_val:.2f} Å" if resolution_val else " (NMR)"))
    elif not xray_ok:
        res["resolved_ok"] = "FAIL"
    else:
        res["resolved_ok"] = "FAIL"
        notes.append(f"Resolved fraction {res_frac:.4f} < {MIN_RESOLVED}")

    # ── (4) Not heavily disordered ────────────────────────────────────────────
    if up_flags and up_flags.get("has_disorder_annot"):
        if res_frac < MIN_RESOLVED:
            res["not_disordered"] = "FAIL"
            notes.append("UniProt: disordered region annotated + low resolved fraction")
        else:
            res["not_disordered"] = "FLAG"
            flagged.append("UniProt annotates a disordered region — verify it is not extensive")
    elif res_frac < MIN_RESOLVED:
        res["not_disordered"] = "FAIL"
        notes.append("Low resolved fraction consistent with disorder")
    else:
        res["not_disordered"] = "PASS"

    # ── (5 & 6) Membrane / solubility ─────────────────────────────────────────
    mem_fail = []
    mem_flag = []

    if in_opm:
        mem_fail.append("Entry found in OPM membrane protein database")

    if up_flags and up_flags.get("available"):
        if up_flags["has_transmembrane"]:
            mem_fail.append("UniProt: transmembrane region annotated")
        if up_flags["has_lipidation"]:
            mem_fail.append("UniProt: lipidation / lipid-anchor modification annotated")
        if up_flags["has_gpi_anchor"]:
            mem_fail.append("UniProt: GPI-anchor annotated")
        if up_flags["is_membrane_assoc"] and not up_flags["has_transmembrane"]:
            sublocs = "; ".join(up_flags["subloc_values"][:3])
            mem_flag.append(f"UniProt subcellular location includes membrane/cell-surface ({sublocs})")
    elif up_flags and not up_flags["available"]:
        mem_flag.append("UniProt data unavailable — manual membrane/lipidation check required")

    if mem_fail:
        res["not_membrane"] = "FAIL"
        res["soluble"]      = "FAIL"
        notes.extend(mem_fail)
    elif mem_flag:
        res["not_membrane"] = "FLAG"
        res["soluble"]      = "FLAG"
        flagged.extend(mem_flag)
    else:
        res["not_membrane"] = "PASS"
        res["soluble"]      = "PASS"

    # ── (7) No large ligand ───────────────────────────────────────────────────
    lig_verdict, lig_notes = classify_ligands(entry.get("nonpolymer_entities") or [])
    res["no_large_ligand"] = lig_verdict
    notes.extend(lig_notes)
    if lig_verdict == "FLAG":
        flagged.append("Medium-weight non-trivial ligand present — confirm not essential for folding")

    # ── Overall ───────────────────────────────────────────────────────────────
    verdicts = [
        res["monomeric"], res["soluble"], res["length_ok"],
        res["resolved_ok"], res["not_disordered"],
        res["not_membrane"], res["no_large_ligand"],
        res["not_obligate_complex"],
    ]
    if "FAIL" in verdicts:
        res["overall"] = "EXCLUDE"
    elif "FLAG" in verdicts or flagged:
        res["overall"] = "FLAG_REVIEW"
    else:
        res["overall"] = "INCLUDE"

    return res


def screen_dev(
    entry:     dict,
    entity_id: str,
    up_flags:  Optional[dict],
    in_opm:    bool,
    chain_id:  Optional[str] = None,
) -> dict:
    """
    Relaxed screening for pre-cutoff development / validation set candidates.

    Differences from screen():
      - Resolution threshold: DEV_MAX_XRAY_RES (3.5 Å) instead of 2.5 Å.
      - Resolved fraction threshold: DEV_MIN_RESOLVED (0.80) instead of 0.90.
      - Medium-weight ligands are PASS (not FLAG) — common in older structures.
      - Disorder annotation alone is a FLAG, not a FAIL, unless resolved < 0.80.

    Core hard-fail criteria (unchanged):
      - Monomeric (1 protein chain instance in assembly 1)
      - Not membrane / not lipidated (OPM or UniProt transmembrane)
      - Not GPI-anchored
      - No large ligand (>500 Da)
      - Not obligate complex
      - Length 80–400 residues
    """
    res = {
        "monomeric":             "UNKNOWN",
        "soluble":               "UNKNOWN",
        "length_ok":             "UNKNOWN",
        "resolved_ok":           "UNKNOWN",
        "not_disordered":        "UNKNOWN",
        "not_membrane":          "UNKNOWN",
        "no_large_ligand":       "UNKNOWN",
        "not_obligate_complex":  "UNKNOWN",
        "overall":               "UNKNOWN",
        "notes":                 [],
        "flag_reasons":          [],
    }
    notes   = res["notes"]
    flagged = res["flag_reasons"]

    # ── Find entity ───────────────────────────────────────────────────────────
    target = None
    for ent in entry.get("polymer_entities") or []:
        if ent["rcsb_id"].split("_")[-1] == entity_id:
            target = ent
            break
    if target is None:
        res["overall"] = "ERROR"
        notes.append("entity not found in GraphQL response")
        return res

    # ── (1) Monomeric ─────────────────────────────────────────────────────────
    assemblies = entry.get("assemblies") or []
    asm1 = next(
        (a for a in assemblies
         if (a.get("rcsb_assembly_info") or {}).get("assembly_id") == "1"),
        assemblies[0] if assemblies else None,
    )
    n_chains = (asm1.get("rcsb_assembly_info") or {}).get(
        "polymer_entity_instance_count_protein"
    ) if asm1 else None

    if n_chains is None:
        res["monomeric"] = "FLAG"
        flagged.append("Assembly chain count unavailable — confirm monomeric")
    elif n_chains == 1:
        res["monomeric"] = "PASS"
        notes.append(f"Assembly 1: {n_chains} protein chain (monomeric)")
    else:
        res["monomeric"] = "FAIL"
        notes.append(f"Assembly 1 has {n_chains} protein chains — not monomeric")

    res["not_obligate_complex"] = "PASS" if res["monomeric"] == "PASS" else "FAIL"

    # ── (2) Sequence length ───────────────────────────────────────────────────
    seq_len = (target.get("entity_poly") or {}).get("rcsb_sample_sequence_length") or 0
    if MIN_LEN <= seq_len <= MAX_LEN:
        res["length_ok"] = "PASS"
        notes.append(f"Length: {seq_len} residues")
    else:
        res["length_ok"] = "FAIL"
        notes.append(f"Length {seq_len} outside [{MIN_LEN}, {MAX_LEN}]")

    # ── (3) Resolved fraction + resolution (relaxed) ──────────────────────────
    res_frac = resolved_fraction(target, entry, chain_id)
    methods  = [e.get("method", "") for e in (entry.get("exptl") or [])]
    method   = methods[0] if methods else ""
    is_nmr   = "NMR" in method.upper()

    xray_ok = True
    resolution_val = None
    if not is_nmr:
        for ref in (entry.get("refine") or []):
            v = ref.get("ls_d_res_high")
            if v is not None:
                resolution_val = float(v)
                break
        if resolution_val is not None and resolution_val > DEV_MAX_XRAY_RES:
            xray_ok = False
            notes.append(f"X-ray resolution {resolution_val:.2f} Å > {DEV_MAX_XRAY_RES} Å limit")

    if res_frac >= DEV_MIN_RESOLVED and xray_ok:
        res["resolved_ok"] = "PASS"
        notes.append(
            f"Resolved fraction: {res_frac:.4f}"
            + (f"; resolution {resolution_val:.2f} Å" if resolution_val else " (NMR)")
        )
    elif not xray_ok:
        res["resolved_ok"] = "FAIL"
    else:
        res["resolved_ok"] = "FAIL"
        notes.append(f"Resolved fraction {res_frac:.4f} < {DEV_MIN_RESOLVED}")

    # ── (4) Disorder: flag only (not fail) for dev set ────────────────────────
    if up_flags and up_flags.get("has_disorder_annot"):
        if res_frac < DEV_MIN_RESOLVED:
            res["not_disordered"] = "FAIL"
            notes.append("UniProt: disordered region annotated + low resolved fraction")
        else:
            res["not_disordered"] = "FLAG"
            flagged.append("UniProt annotates a disordered region — acceptable for dev set; verify not extensive")
    elif res_frac < DEV_MIN_RESOLVED:
        res["not_disordered"] = "FAIL"
        notes.append("Low resolved fraction consistent with disorder")
    else:
        res["not_disordered"] = "PASS"

    # ── (5 & 6) Membrane / solubility (same hard criteria as test sets) ───────
    mem_fail = []
    mem_flag = []

    if in_opm:
        mem_fail.append("Entry found in OPM membrane protein database")

    if up_flags and up_flags.get("available"):
        if up_flags["has_transmembrane"]:
            mem_fail.append("UniProt: transmembrane region annotated")
        if up_flags["has_lipidation"]:
            mem_fail.append("UniProt: lipidation / lipid-anchor modification annotated")
        if up_flags["has_gpi_anchor"]:
            mem_fail.append("UniProt: GPI-anchor annotated")
        if up_flags["is_membrane_assoc"] and not up_flags["has_transmembrane"]:
            sublocs = "; ".join(up_flags["subloc_values"][:3])
            mem_flag.append(f"UniProt subcellular location includes membrane/cell-surface ({sublocs})")
    elif up_flags and not up_flags["available"]:
        mem_flag.append("UniProt data unavailable — manual membrane/lipidation check required")

    if mem_fail:
        res["not_membrane"] = "FAIL"
        res["soluble"]      = "FAIL"
        notes.extend(mem_fail)
    elif mem_flag:
        res["not_membrane"] = "FLAG"
        res["soluble"]      = "FLAG"
        flagged.extend(mem_flag)
    else:
        res["not_membrane"] = "PASS"
        res["soluble"]      = "PASS"

    # ── (7) Ligands: same thresholds as test sets ─────────────────────────────
    lig_verdict, lig_notes = classify_ligands(entry.get("nonpolymer_entities") or [])
    res["no_large_ligand"] = lig_verdict
    notes.extend(lig_notes)
    if lig_verdict == "FLAG":
        flagged.append("Medium-weight non-trivial ligand present — confirm not essential for folding")

    # ── Overall ───────────────────────────────────────────────────────────────
    verdicts = [
        res["monomeric"], res["soluble"], res["length_ok"],
        res["resolved_ok"], res["not_disordered"],
        res["not_membrane"], res["no_large_ligand"],
        res["not_obligate_complex"],
    ]
    if "FAIL" in verdicts:
        res["overall"] = "EXCLUDE"
    elif "FLAG" in verdicts or flagged:
        res["overall"] = "FLAG_REVIEW"
    else:
        res["overall"] = "INCLUDE"

    return res


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Stage 1 protein dataset builder"
    )
    parser.add_argument("--limit",     type=int, default=4000,
                        help="Max polymer_entity results from RCSB Search for post-cutoff (default 1000)")
    parser.add_argument("--dev-limit", type=int, default=1500,
                        help="Max polymer_entity results from RCSB Search for dev/val pre-cutoff (default 1500)")
    parser.add_argument("--out-dir",   default="./output",
                        help="Output directory for CSV files (default ./output)")
    parser.add_argument("--skip-dev",  action="store_true",
                        help="Skip dev/val set generation (useful for quick test runs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Search ─────────────────────────────────────────────────────────
    print("\n=== Step 1a: RCSB Search — post-cutoff (test set candidates) ===")
    candidates = rcsb_search(args.limit)
    priority_keys = set(PRIORITY_ENTITIES)
    candidates = PRIORITY_ENTITIES + [c for c in candidates if c not in priority_keys]

    dev_candidates: list[tuple[str, str]] = []
    if not args.skip_dev:
        print("\n=== Step 1b: RCSB Search — pre-cutoff (dev/val set candidates) ===")
        dev_candidates = rcsb_search_dev(args.dev_limit)

    all_pairs  = list(dict.fromkeys(candidates + dev_candidates))
    entry_ids  = list(dict.fromkeys(e for e, _ in all_pairs))

    # ── Step 2: GraphQL metadata ───────────────────────────────────────────────
    print(f"\n=== Step 2: GraphQL metadata ({len(entry_ids)} entries) ===")
    entry_data = fetch_all_entries(entry_ids)

    # ── Steps 3–5: Screen ──────────────────────────────────────────────────────
    print("\n=== Steps 3–5: UniProt + OPM + Screening ===")
    uniprot_cache: dict[str, Optional[dict]] = {}

    rows_master  = []
    rows_screen  = []
    rows_review  = []
    rows_dev     = []   # pre-cutoff INCLUDE/FLAG_REVIEW rows for dev set
    rows_val     = []   # held-out 20% of dev rows for validation set
    pass_count   = 0
    dev_pass_count = 0

    # Screen post-cutoff candidates first, then pre-cutoff dev/val
    all_candidates_ordered = candidates + [
        c for c in dev_candidates if c not in set(candidates)
    ]

    for idx, (entry_id, entity_id) in enumerate(all_candidates_ordered):
        entry = entry_data.get(entry_id)
        if not entry:
            continue

        # Locate entity
        target = None
        for ent in entry.get("polymer_entities") or []:
            if ent["rcsb_id"].split("_")[-1] == entity_id:
                target = ent
                break
        if target is None:
            continue

        # Chain ID
        instances = target.get("polymer_entity_instances") or []
        chain_id  = "A"
        if instances:
            chain_id = (
                instances[0]
                .get("rcsb_polymer_entity_instance_container_identifiers", {})
                .get("auth_asym_id") or "A"
            )

        # UniProt
        up_accs = [u["rcsb_id"] for u in (target.get("uniprots") or [])]
        up_acc  = up_accs[0] if up_accs else None
        if up_acc and up_acc not in uniprot_cache:
            print(f"  [UniProt] {up_acc} …", flush=True)
            raw = fetch_uniprot(up_acc)
            uniprot_cache[up_acc] = parse_uniprot(raw)
            time.sleep(API_SLEEP)
        up_flags = uniprot_cache.get(up_acc) if up_acc else None

        # OPM
        in_opm = check_opm(entry_id)
        time.sleep(API_SLEEP)

        # ── Collect metadata for CSVs ──────────────────────────────────────────
        seq = re.sub(r"\s+", "", (target.get("entity_poly") or {}).get(
            "pdbx_seq_one_letter_code_can") or "")
        seq_len = (target.get("entity_poly") or {}).get("rcsb_sample_sequence_length") or len(seq)
        release = ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
        method  = ([(e.get("method") or "") for e in (entry.get("exptl") or [])] + [""])[0]

        resolution_str = ""
        for ref in (entry.get("refine") or []):
            if ref.get("ls_d_res_high") is not None:
                resolution_str = f"{float(ref['ls_d_res_high']):.2f}"
                break

        organism = ""
        orgs = target.get("rcsb_entity_source_organism") or []
        if orgs:
            organism = orgs[0].get("scientific_name") or ""

        res_frac = resolved_fraction(target, entry, chain_id)

        assemblies = entry.get("assemblies") or []
        asm1 = None
        for a in assemblies:
            if (a.get("rcsb_assembly_info") or {}).get("assembly_id") == "1":
                asm1 = a
                break
        if asm1 is None and assemblies:
            asm1 = assemblies[0]
        asm_info     = (asm1.get("rcsb_assembly_info") or {}) if asm1 else {}
        n_chains     = asm_info.get("polymer_entity_instance_count_protein") or ""
        oligo        = ""
        asm_desc     = f"assembly 1: {oligo} ({n_chains} protein chain{'s' if n_chains != 1 else ''})" if n_chains else ""

        prot_name = (target.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""

        # Ligands (non-trivial only for display)
        lig_display = []
        for ne in (entry.get("nonpolymer_entities") or []):
            cid = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id") or ""
            if cid and cid not in TRIVIAL_LIGANDS:
                lig_display.append(cid)
        ligands_str = "; ".join(lig_display) if lig_display else "none"

        # Determine whether this is a pre-cutoff (dev/val) or post-cutoff entry
        is_dev_candidate = (entry_id, entity_id) in set(dev_candidates) and \
                           (entry_id, entity_id) not in priority_keys

        # Screen: relaxed criteria for pre-cutoff dev/val; strict for post-cutoff
        if is_dev_candidate:
            result = screen_dev(entry, entity_id, up_flags, in_opm, chain_id)
        else:
            result = screen(entry, entity_id, up_flags, in_opm, chain_id)

        overall = result["overall"]

        # Train/test split label
        if release > CUTOFF_DATE:
            split = "post_cutoff_candidate"
        else:
            split = "development_candidate"   # refined to dev/val below

        candidate_key = (entry_id, entity_id)
        is_priority = candidate_key in priority_keys
        notes_str  = "; ".join(result["notes"])
        flags_str  = "; ".join(result["flag_reasons"])

        # Screening status labels matching the existing spreadsheet convention.
        # Priority cases remain in the dataset even if an automated rule flags
        # them; any exception is explicit in the review notes.
        if is_priority and overall != "INCLUDE":
            status = "manual_screen_pass"
            notes_str += (
                "; PRIORITY_CASE_OVERRIDE: retained for Stage 1/2 despite "
                f"automated decision {overall}"
            )
            flags_str += (
                ("; " if flags_str else "")
                + "Priority biological case: manually review automated exception"
            )
        elif overall == "INCLUDE":
            if is_dev_candidate:
                status = "dev_screen_pass"
                dev_pass_count += 1
            else:
                status = "auto_screen_pass"
                pass_count += 1
        elif overall == "EXCLUDE":
            status = "auto_screen_fail"
        else:
            status = "auto_screen_flag"

        master_row = {
            "PDB_ID":               entry_id,
            "chain_ID":             chain_id,
            "entity_ID":            entity_id,
            "protein_name":         prot_name,
            "release_date":         release,
            "sequence":             seq,
            "length":               seq_len,
            "experimental_method":  method,
            "resolution":           resolution_str,
            "resolved_fraction":    f"{res_frac:.5f}",
            "organism":             organism,
            "uniprot_accession":    up_acc or "",
            "number_of_chains":     n_chains,
            "ligands":              ligands_str,
            "biological_assembly":  asm_desc,
            "training_or_test_split": split,
            "sequence_cluster":     "PENDING_CLUSTERING",
            "screening_status":     status,
            "priority_case":        "YES" if is_priority else "NO",
            "review_notes":         notes_str,
            "flag_for_review":      flags_str,
            "rcsb_source":          f"https://www.rcsb.org/structure/{entry_id}",
            "uniprot_source":       (f"https://www.uniprot.org/uniprotkb/{up_acc}/entry"
                                     if up_acc else ""),
        }

        # Resolved-fraction column header reflects actual threshold used
        resolved_col = "resolved_ge_0_80" if is_dev_candidate else "resolved_ge_0_90"

        screen_row = {
            "PDB_ID":                   entry_id,
            "chain_ID":                 chain_id,
            "protein_name":             prot_name,
            "set_type":                 "dev_val" if is_dev_candidate else "test",
            "monomeric":                result["monomeric"],
            "soluble":                  result["soluble"],
            "length_80_400":            result["length_ok"],
            resolved_col:               result["resolved_ok"],
            "not_heavily_disordered":   result["not_disordered"],
            "not_membrane":             result["not_membrane"],
            "no_large_ligand":          result["no_large_ligand"],
            "not_obligate_complex":     result["not_obligate_complex"],
            "overall_decision":         ("INCLUDE_PRIORITY_EXCEPTION"
                                             if is_priority and overall != "INCLUDE"
                                         else "INCLUDE" if overall == "INCLUDE"
                                         else "FLAG" if overall == "FLAG_REVIEW"
                                         else "EXCLUDE"),
            "evidence":                 notes_str,
            "flag_for_review":          flags_str,
        }

        rows_master.append(master_row)
        rows_screen.append(screen_row)

        if result["flag_reasons"]:
            rows_review.append({**master_row, "review_reasons": flags_str})

        # Route pre-cutoff accepted entries to dev pool (val split applied later)
        if is_dev_candidate and overall in ("INCLUDE", "FLAG_REVIEW"):
            rows_dev.append(master_row)

        if (idx + 1) % 100 == 0:
            total_iterated = len(all_candidates_ordered)
            print(
                f"  [{idx+1}/{total_iterated}] "
                f"test-pass: {pass_count} | dev-pass: {dev_pass_count} | "
                f"review queue: {len(rows_review)}",
                flush=True,
            )

        # Early stop for post-cutoff once we hit the target
        # (pre-cutoff dev/val search runs to its own --dev-limit)
        if (not is_dev_candidate
                and pass_count >= int(TARGET_AUTO_PASSES * 1.4)
                and idx > 300):
            print(
                f"\n  Reached {pass_count} post-cutoff auto-passes "
                f"(target: {TARGET_AUTO_PASSES}). Continuing dev/val …",
                flush=True,
            )

    # ── Dev / Val split ────────────────────────────────────────────────────────
    # Deterministic 80/20 split by sorted PDB ID — reproducible without a seed.
    # After compute_similarity.py runs clustering on the full pre-cutoff pool,
    # re-assign by cluster so that no cluster spans both sets.
    rows_dev_sorted = sorted(rows_dev, key=lambda r: (r["PDB_ID"], r["entity_ID"]))
    val_count = max(1, round(len(rows_dev_sorted) * VAL_FRACTION))
    # Take every 5th entry as val to spread them across the alphabet
    val_indices = set(range(0, len(rows_dev_sorted), 5)[:val_count])
    for i, row in enumerate(rows_dev_sorted):
        if i in val_indices:
            row["training_or_test_split"] = "validation_set"
            rows_val.append(row)
        else:
            row["training_or_test_split"] = "development_set"

    # Update master rows with refined dev/val labels
    split_map = {(r["PDB_ID"], r["entity_ID"]): r["training_or_test_split"]
                 for r in rows_dev_sorted}
    for row in rows_master:
        key = (row["PDB_ID"], row["entity_ID"])
        if key in split_map:
            row["training_or_test_split"] = split_map[key]

    # ── Write CSVs ─────────────────────────────────────────────────────────────
    print(f"\n=== Writing outputs to {out_dir} ===")

    MASTER_FIELDS = [
        "PDB_ID", "chain_ID", "entity_ID", "protein_name",
        "release_date", "sequence", "length",
        "experimental_method", "resolution", "resolved_fraction",
        "organism", "uniprot_accession",
        "number_of_chains", "ligands", "biological_assembly",
        "training_or_test_split", "sequence_cluster",
        "screening_status", "priority_case", "review_notes", "flag_for_review",
        "rcsb_source", "uniprot_source",
    ]
    SCREEN_FIELDS = [
        "PDB_ID", "chain_ID", "protein_name",
        "set_type",
        "monomeric", "soluble", "length_80_400",
        "resolved_ge_0_90", "resolved_ge_0_80",   # one populated per row
        "not_heavily_disordered", "not_membrane", "no_large_ligand",
        "not_obligate_complex", "overall_decision",
        "evidence", "flag_for_review",
    ]
    REVIEW_FIELDS = MASTER_FIELDS + ["review_reasons"]

    def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {path.name}: {len(rows)} rows")

    write_csv(out_dir / "master_candidates.csv",  rows_master,  MASTER_FIELDS)
    write_csv(out_dir / "screening_detail.csv",   rows_screen,  SCREEN_FIELDS)
    write_csv(out_dir / "review_queue.csv",        rows_review,  REVIEW_FIELDS)
    if rows_dev_sorted:
        dev_only  = [r for r in rows_dev_sorted
                     if r["training_or_test_split"] == "development_set"]
        write_csv(out_dir / "dev_set.csv",         dev_only,     MASTER_FIELDS)
        write_csv(out_dir / "val_set.csv",         rows_val,     MASTER_FIELDS)

    # ── Summary ────────────────────────────────────────────────────────────────
    test_screen = [r for r in rows_screen if r.get("set_type") != "dev_val"]
    dev_screen  = [r for r in rows_screen if r.get("set_type") == "dev_val"]

    n_test_pass = sum(1 for r in test_screen if r["overall_decision"] == "INCLUDE")
    n_test_fail = sum(1 for r in test_screen if r["overall_decision"] == "EXCLUDE")
    n_test_flag = sum(1 for r in test_screen if r["overall_decision"] == "FLAG")
    n_dev_pass  = sum(1 for r in dev_screen  if r["overall_decision"] == "INCLUDE")
    n_val       = len(rows_val)
    n_dev       = len([r for r in rows_dev_sorted
                       if r["training_or_test_split"] == "development_set"])

    print(f"""
=== Run summary ===
  POST-CUTOFF (test set candidates)
    Processed  : {len(test_screen)}
    INCLUDE    : {n_test_pass}
    EXCLUDE    : {n_test_fail}
    FLAG       : {n_test_flag}
    {'✓ Target of ' + str(TARGET_AUTO_PASSES) + ' test passes met.'
     if n_test_pass >= TARGET_AUTO_PASSES
     else '✗ Target not yet met — re-run with a higher --limit.'}

  PRE-CUTOFF (dev / val sets)
    Processed  : {len(dev_screen)}
    Accepted   : {n_dev_pass}  (criteria: monomeric, soluble, 80-400 aa, ≥80% resolved, ≤3.5 Å)
    → development_set : {n_dev}  (80%)
    → validation_set  : {n_val}  (20%, every 5th by PDB ID)
    {'(skipped — run without --skip-dev to generate)' if args.skip_dev else ''}

Outputs
  master_candidates.csv  — all entries, all sets, training_or_test_split labelled
  screening_detail.csv   — per-criterion verdicts for every candidate
  review_queue.csv       — FLAG entries requiring manual review
  dev_set.csv            — development set (pre-cutoff, 80%)
  val_set.csv            — validation set  (pre-cutoff, 20%)

Next steps
  1. Review review_queue.csv for FLAG entries.
  2. Run compute_similarity.py on master_candidates.csv to assign Test A / Test B
     labels and cluster pre-cutoff sequences.
  3. After clustering, optionally reassign dev/val split by cluster boundary
     (edit training_or_test_split in master_candidates.csv) to prevent
     near-identical sequences spanning both sets.
""")


if __name__ == "__main__":
    main()
