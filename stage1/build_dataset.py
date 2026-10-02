"""
Stage 1: build the clean benchmark dataset.

Pipeline
--------
  1. RCSB Search API   -> pre- and post-cutoff single-chain candidate entries
  2. RCSB GraphQL API  -> batched metadata (sequence, assemblies, resolution,
                          per-chain unobserved residues, ligand chemistry,
                          UniProt mapping, membrane-database annotations)
  3. UniProt REST API  -> transmembrane / lipidation / GPI anchor, subcellular
                          location, disordered regions, subunit composition
  4. OPM               -> membrane-protein cross-reference
  5. Biological screen -> PASS / FLAG / FAIL for each Stage 1 restriction and
                          an overall INCLUDE / FLAG / EXCLUDE decision
  6. RCSB 30% sequence clusters, used to keep the splits non-redundant
  7. Direct sequence search of every post-cutoff candidate against
     pre-cutoff PDB: any hit with >=30% identity over >=80% of the
     candidate -> Test A, otherwise -> Test B
  8. Cluster-disjoint development / validation / Test A / Test B splits
  9. CSV outputs, including a review queue and a random manual-inspection
     sample

OpenFold3 structural training cutoff: 2021-09-30
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PROTEIN FILTER SETTINGS
# ============================================================

CUTOFF_DATE = "2021-09-30"

# Number of residues.
MIN_LENGTH = 80
MAX_LENGTH = 400

# Resolved fraction >= 90% and resolution <= 3.5 Å.
MIN_RESOLVED_FRACTION = 0.90
MAX_RESOLUTION = 3.5

ALLOWED_METHODS = {
    "X-RAY DIFFRACTION",
    "ELECTRON MICROSCOPY",
}

# Not heavily disordered.
# A missing (unmodelled) stretch longer than this is flagged for review even
# when the overall resolved fraction is acceptable.
MAX_UNOBSERVED_SEGMENT = 20
# Flag when UniProt-annotated disordered regions cover more than this
# fraction of the crystallised construct.
MAX_DISORDER_ANNOTATION_FRACTION = 0.20

# Not dependent on large ligands.
# Non-trivial ligands at/above the large limits fail; between the medium and
# large limits they are flagged for manual review.
LARGE_LIGAND_MW = 500.0
LARGE_LIGAND_HEAVY_ATOMS = 25
MEDIUM_LIGAND_MW = 200.0
MEDIUM_LIGAND_HEAVY_ATOMS = 12

# FLAG candidates need manual review and are not included in the standard
# development/validation or locked test sets.
ALLOW_FLAGGED_IN_DEV_VAL = False
ALLOW_FLAGGED_IN_TEST = False

# Number of raw candidates to inspect from each side of the cutoff.
# The biological screen is stricter than a simple metadata filter, so more
# raw candidates are needed to fill every split.
N_PRE_CANDIDATES = 500
N_POST_CANDIDATES = 2000

# Final dataset sizes.
N_DEVELOPMENT = 30
N_VALIDATION = 15
N_TEST_A = 20
N_TEST_B = 20

# Random subset of accepted candidates written out for manual inspection.
N_MANUAL_INSPECTION = 20

RANDOM_SEED = 42

# Number of search hits from which the raw candidate sample is drawn.
# None retrieves the complete result set, which is what makes the candidate
# sample a uniform random draw. RCSB returns hits in a fixed order that
# front-loads the oldest entries, so truncating the search silently biases
# the sample by release date (a 5000-hit cap gave pre-cutoff candidates
# exclusively from 1988-2004). Set an integer only for quick test runs.
MAX_SEARCH_HITS = None

# Test A / Test B rule. A post-cutoff protein goes to Test A when any
# pre-cutoff PDB protein aligns to it with at least TESTB_MAX_PIDENT identity
# over at least TESTB_MIN_COVERAGE of the post-cutoff sequence; otherwise it
# is eligible for Test B.
TESTB_MAX_PIDENT = 0.30
TESTB_MIN_COVERAGE = 0.80
# Hits inspected per search. Hits are ranked by alignment score, so a
# qualifying hit is almost always among the first ones returned.
MAX_SEQUENCE_SEARCH_RESULTS = 1000
SEQUENCE_SEARCH_RETRIES = 3

# Do not hammer the RCSB / UniProt servers with too many simultaneous requests.
MAX_WORKERS = 6

# Entries per RCSB GraphQL metadata request.
GRAPHQL_BATCH_SIZE = 50

# Entries per UniProt request, and how persistently to retry when UniProt
# is overloaded (waits of 5, 10, 20, 40, 80 s).
UNIPROT_BATCH_SIZE = 100
UNIPROT_RETRIES = 6

# Special proteins selected for investigation. They are always included,
# even when they fail the standard automated filters.
#
# 8EXF: BCCIPalpha is chain B in a FAM46A-BCCIPalpha complex.
# 8URV: pro-IL-18 is chain A and was determined by solution NMR.
#
# forced_split pins a target to a split. BCCIPalpha is pinned to Test B
# because its fold differs substantially from the sequence-similar
# pre-cutoff BCCIPbeta, so sequence similarity alone would misjudge it.
# With forced_split None, the target goes through the same Test A / Test B
# rule as every other post-cutoff protein.
PRIORITY_TARGETS = {
    "8EXF": {
        "chain_id": "B",
        "protein_name": "BCCIPalpha",
        "forced_split": "test_b",
    },
    "8URV": {
        "chain_id": "A",
        "protein_name": "pro-IL-18",
        "forced_split": None,
    },
}


# ============================================================
# BIOLOGICAL ANNOTATION VOCABULARY
# ============================================================

# Buffer salts, ions, cryoprotectants, and crystallisation additives.
# These are never considered folding-relevant ligands.
TRIVIAL_LIGANDS = {
    # Water
    "HOH", "DOD",
    # Common ions
    "NA", "K", "MG", "CA", "MN", "FE", "FE2", "ZN", "CU", "CU1", "CO", "NI",
    "CD", "CL", "BR", "F", "IOD", "NH4", "NO3", "SCN", "AZI", "CS", "RB",
    "LI", "SR", "BA", "UNX",
    # Crystallisation additives / cryoprotectants / buffers
    "SO4", "PO4", "PI", "ACT", "ACY", "FMT", "GOL", "EDO", "MPD", "PEG",
    "PGE", "PG4", "1PE", "P6G", "2PE", "BME", "DTT", "DMS", "TRS", "MES",
    "IMD", "EPE", "CIT", "FLC", "TAR", "MLI", "MRD", "BTB", "B3P", "IPA",
    "EOH", "MOH", "PGO", "PGR", "SIN", "CO3", "BCT", "NHE", "CXS",
    # Modified amino-acid residues (covalently part of the chain)
    "KCX", "MSE", "SEP", "TPO", "PTR", "CME", "CSO", "OCS", "HIC", "MLY",
    "M3L", "LYZ",
}

# Membrane-protein databases whose annotations appear on RCSB polymer
# entities. GO "membrane" terms are deliberately excluded here because they
# also annotate many peripheral/soluble proteins; UniProt subcellular
# location handles those cases as FLAGs instead.
MEMBRANE_DATABASES = {"PDBTM", "OPM", "MEMPROTMD", "MPSTRUC"}

UNIPROT_LIPID_KEYWORDS = {
    "Lipoprotein", "GPI-anchor", "Lipid anchor",
    "Myristate", "Palmitate", "Prenylation", "Farnesylation",
    "Geranylgeranylation", "Myristoylation", "Palmitoylation",
}

MEMBRANE_SUBLOC_TOKENS = ("membrane", "cell surface")

# UniProt SUBUNIT text that suggests an obligate oligomer or complex subunit.
OBLIGATE_COMPLEX_PATTERN = re.compile(
    r"\b(?:homo|hetero)(?:di|tri|tetra|penta|hexa|hepta|octa|dodeca|oligo)mer"
    r"|\bcomponent of the\b"
    r"|\bpart of the\b.{0,60}\bcomplex\b",
    re.I,
)

# Common expression/purification tags, searched for within TAG_WINDOW
# residues of either terminus.
EXPRESSION_TAGS = {
    "poly-His": re.compile(r"H{5,}"),
    "Strep-tag II": re.compile("WSHPQFEK"),
    "FLAG": re.compile("DYKDDDDK"),
    "HA": re.compile("YPYDVPDYA"),
    "c-Myc": re.compile("EQKLISEEDL"),
    "V5": re.compile("GKPIPNPLLGLDST"),
    "AviTag": re.compile("GLNDIFEAQKIEWHE"),
    "S-tag": re.compile("KETAAAKFERQHMDS"),
    "T7": re.compile("MASMTGGQQMG"),
}
TAG_WINDOW = 30


# ============================================================
# URLS
# ============================================================

SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_API = "https://data.rcsb.org/graphql"
# Batch endpoint: many entries per request instead of one request per
# protein, which UniProt throttles with 503 errors.
UNIPROT_BATCH_API = "https://rest.uniprot.org/uniprotkb/accessions"

# OPM publishes oriented coordinates only for entries it contains, so a
# successful HEAD request identifies an OPM membrane protein.
OPM_PDB_URL = "https://opm-assets.storage.googleapis.com/pdb/{pdb_id}.pdb"

CLUSTER_30_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/"
    "clusters-by-entity-30.txt"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HTTP AND THREADING HELPERS
# ============================================================

def post_json(url: str, payload: dict, retries: int = 3):
    """POST JSON with simple retry handling."""
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise

            wait = 2 ** attempt
            print(f"Request failed: {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)


def url_exists(url: str, retries: int = 3):
    """
    HEAD a URL. Returns True (200), False (404), or None when the answer
    could not be determined.
    """
    for attempt in range(retries):
        try:
            response = requests.head(url, timeout=20, allow_redirects=True)
            if response.status_code == 200:
                return True
            if response.status_code == 404:
                return False
            response.raise_for_status()

        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None


def run_threaded(function, items, label: str):
    """Run function(item) for every item with MAX_WORKERS threads."""

    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(function, item): item for item in items}

        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()

            if completed % 50 == 0 or completed == len(futures):
                print(f"{label}: {completed}/{len(futures)}")

    return results


# ============================================================
# RCSB SEARCH
# ============================================================

def terminal(attribute: str, operator: str, value):
    """One RCSB Search API text-attribute condition."""
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": operator,
            "value": value,
        },
    }


def search_candidate_pdb_ids(
    pre_cutoff: bool,
    max_results: int | None = MAX_SEARCH_HITS,
):
    """
    Search RCSB for entries that:
      - contain one protein entity
      - contain one deposited polymer chain
      - have sequence length 80-400
      - are before OR after the OpenFold3 cutoff

    Pages through the whole result set unless max_results caps it. The
    complete set matters: the hits are not returned in a random order, so
    a truncated search is a biased sample of the PDB rather than a smaller
    one, and no amount of later shuffling repairs that.
    """

    date_operator = "less_or_equal" if pre_cutoff else "greater"

    nodes = [
        terminal("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
        terminal(
            "rcsb_entry_info.deposited_polymer_entity_instance_count",
            "equals",
            1,
        ),
        terminal(
            "entity_poly.rcsb_sample_sequence_length",
            "greater_or_equal",
            MIN_LENGTH,
        ),
        terminal(
            "entity_poly.rcsb_sample_sequence_length",
            "less_or_equal",
            MAX_LENGTH,
        ),
        terminal(
            "rcsb_accession_info.initial_release_date",
            date_operator,
            CUTOFF_DATE,
        ),
    ]

    all_ids = []
    page_size = 1000
    start = 0
    total_count = None

    while True:

        if max_results is None:
            rows = page_size
        else:
            if len(all_ids) >= max_results:
                break
            rows = min(page_size, max_results - len(all_ids))

        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],
            },
        }

        data = post_json(SEARCH_API, payload)
        hits = data.get("result_set", [])

        if total_count is None:
            total_count = int(data.get("total_count") or 0)
            print(f"Result set contains {total_count:,} entries.")

        if not hits:
            break

        ids = [hit["identifier"].upper() for hit in hits]
        all_ids.extend(ids)
        start += len(ids)

        if len(ids) < rows:
            break

        if total_count and start >= total_count:
            break

    return list(dict.fromkeys(all_ids))


# ============================================================
# RCSB METADATA (GRAPHQL)
# ============================================================

METADATA_QUERY = """
query GetEntries($ids: [String!]!) {
    entries(entry_ids: $ids) {
        rcsb_id
        rcsb_accession_info { initial_release_date }
        exptl { method }
        rcsb_entry_info { resolution_combined }
        assemblies {
            rcsb_assembly_info {
                assembly_id
                polymer_entity_instance_count
                polymer_entity_instance_count_protein
                polymer_entity_instance_count_DNA
                polymer_entity_instance_count_RNA
                polymer_entity_instance_count_nucleic_acid_hybrid
                modeled_polymer_monomer_count
                polymer_monomer_count
            }
            pdbx_struct_assembly { details oligomeric_details }
        }
        polymer_entities {
            rcsb_id
            entity_poly {
                pdbx_seq_one_letter_code_can
                rcsb_sample_sequence_length
            }
            rcsb_polymer_entity { pdbx_description }
            rcsb_entity_source_organism { ncbi_scientific_name }
            rcsb_polymer_entity_annotation { type name provenance_source }
            rcsb_polymer_entity_align {
                reference_database_name
                reference_database_accession
                aligned_regions { entity_beg_seq_id ref_beg_seq_id length }
            }
            uniprots { rcsb_id }
            polymer_entity_instances {
                rcsb_polymer_entity_instance_container_identifiers {
                    auth_asym_id
                    asym_id
                }
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
        branched_entities {
            rcsb_branched_entity { pdbx_description formula_weight }
            rcsb_branched_entity_container_identifiers { chem_comp_monomers }
        }
    }
}
"""


def fetch_entry_metadata(pdb_ids):
    """
    Retrieve metadata for many PDB entries in batches through the
    RCSB GraphQL Data API.

    Returns {PDB_ID: entry}.
    """

    pdb_ids = list(dict.fromkeys(x.upper() for x in pdb_ids))

    result = {}

    for start in range(0, len(pdb_ids), GRAPHQL_BATCH_SIZE):

        batch = pdb_ids[start:start + GRAPHQL_BATCH_SIZE]

        data = post_json(
            GRAPHQL_API,
            {"query": METADATA_QUERY, "variables": {"ids": batch}},
        )

        entries = (data.get("data") or {}).get("entries") or []

        # GraphQL reports schema problems with HTTP 200, so surface them.
        if data.get("errors"):
            messages = "; ".join(
                str(error.get("message", error))
                for error in data["errors"][:3]
            )

            if not entries:
                raise RuntimeError(
                    f"RCSB GraphQL metadata query failed: {messages}"
                )

            print(f"GraphQL warning: {messages}")

        for entry in entries:
            if entry and entry.get("rcsb_id"):
                result[entry["rcsb_id"].upper()] = entry

        print(
            f"Fetched metadata "
            f"{min(start + len(batch), len(pdb_ids))}/{len(pdb_ids)}"
        )

    return result


# ============================================================
# METADATA EXTRACTION
# ============================================================

def clean_sequence(sequence: str) -> str:
    """Remove whitespace/newlines from an RCSB sequence."""
    if not sequence:
        return ""

    return re.sub(r"\s+", "", sequence).upper()


def is_standard_protein_sequence(sequence: str) -> bool:
    """Require only the 20 standard amino-acid letters."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    return bool(sequence) and set(sequence).issubset(standard)


def instance_chain_ids(instance) -> tuple[str, str]:
    """Return (author chain ID, label chain ID) for a polymer instance."""
    ids = (
        instance.get("rcsb_polymer_entity_instance_container_identifiers")
        or {}
    )

    return (
        str(ids.get("auth_asym_id") or ""),
        str(ids.get("asym_id") or ""),
    )


def select_entity(entry: dict, target_chain_id: str | None = None):
    """
    Select the protein polymer entity and chain instance to benchmark.

    Standard candidates must contain exactly one polymer entity.

    Priority targets specify a chain ID, and the matching polymer entity is
    selected even when the entry contains several protein entities
    (for example, 8EXF).

    Returns (entity, instance), or (None, None) when the entry is not a
    valid standard candidate.
    """

    entities = entry.get("polymer_entities") or []

    if target_chain_id is None:

        if len(entities) != 1:
            return None, None

        entity = entities[0]
        instances = entity.get("polymer_entity_instances") or []

        return entity, (instances[0] if instances else None)

    target_chain_id = str(target_chain_id)

    for entity in entities:
        for instance in entity.get("polymer_entity_instances") or []:
            if target_chain_id in instance_chain_ids(instance):
                return entity, instance

    raise RuntimeError(
        f"Could not find target chain {target_chain_id} "
        f"in {entry.get('rcsb_id')}."
    )


def assembly_info(assembly: dict | None) -> dict:
    if not assembly:
        return {}

    return assembly.get("rcsb_assembly_info") or {}


def preferred_assembly(entry: dict):
    """
    Return the first biological assembly (assembly 1), which is the
    preferred/author-defined assembly in RCSB, or the first listed one.
    """

    assemblies = entry.get("assemblies") or []

    for assembly in assemblies:
        if str(assembly_info(assembly).get("assembly_id")) == "1":
            return assembly

    return assemblies[0] if assemblies else None


def nucleic_acid_count(info: dict) -> int:
    return sum(
        info.get(key) or 0
        for key in (
            "polymer_entity_instance_count_DNA",
            "polymer_entity_instance_count_RNA",
            "polymer_entity_instance_count_nucleic_acid_hybrid",
        )
    )


def chain_resolution_stats(entry: dict, instance: dict | None):
    """
    Return (resolved_fraction, longest_unobserved_segment) for the chain.

    Prefer the selected chain's RCSB UNOBSERVED_RESIDUE_XYZ annotation.
    This is correct at the chain level for both ordinary monomers and
    priority targets embedded in multi-chain structures. Assembly-level
    modelled/total residue counts are used only when chain annotations are
    unavailable.
    """

    if instance is not None:

        summaries = instance.get("rcsb_polymer_instance_feature_summary")

        if summaries is not None:

            unobserved = [
                summary
                for summary in summaries
                if summary
                and str(summary.get("type", "")) == "UNOBSERVED_RESIDUE_XYZ"
            ]

            coverage = sum(
                float(summary.get("coverage") or 0.0)
                for summary in unobserved
            )

            longest = max(
                (
                    int(summary.get("maximum_length") or 0)
                    for summary in unobserved
                ),
                default=0,
            )

            return max(0.0, min(1.0, 1.0 - coverage)), longest

    info = assembly_info(preferred_assembly(entry))

    modeled = info.get("modeled_polymer_monomer_count")
    total = info.get("polymer_monomer_count")

    if modeled is not None and total:
        return max(0.0, min(1.0, float(modeled) / float(total))), None

    return None, None


def extract_metadata(
    pdb_id: str,
    entry: dict,
    entity: dict,
    instance: dict | None,
    priority_target: bool = False,
    priority_name: str = "",
    forced_split: str | None = None,
):
    """Build the metadata row for one candidate protein chain."""

    entity_poly = entity.get("entity_poly") or {}

    sequence = clean_sequence(
        entity_poly.get("pdbx_seq_one_letter_code_can", "")
    )

    length = entity_poly.get("rcsb_sample_sequence_length")

    if length is None:
        length = len(sequence)

    chain_id, label_chain_id = (
        instance_chain_ids(instance) if instance is not None else ("", "")
    )

    entity_number = str(entity.get("rcsb_id", "")).split("_")[-1]

    release_date = (
        (entry.get("rcsb_accession_info") or {}).get("initial_release_date")
        or ""
    )[:10]

    methods = [
        experiment["method"].upper()
        for experiment in entry.get("exptl") or []
        if experiment.get("method")
    ]

    resolutions = [
        float(x)
        for x in (
            (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
            or []
        )
        if x is not None
    ]

    organisms = sorted(
        {
            organism["ncbi_scientific_name"]
            for organism in entity.get("rcsb_entity_source_organism") or []
            if organism.get("ncbi_scientific_name")
        }
    )

    uniprot_accessions = [
        uniprot["rcsb_id"]
        for uniprot in entity.get("uniprots") or []
        if uniprot.get("rcsb_id")
    ]

    resolved_fraction, longest_gap = chain_resolution_stats(entry, instance)

    assembly = preferred_assembly(entry)
    info = assembly_info(assembly)
    oligomeric_details = (
        (assembly or {}).get("pdbx_struct_assembly") or {}
    ).get("oligomeric_details") or ""

    biological_assembly = None

    if info.get("assembly_id") is not None:
        biological_assembly = str(info["assembly_id"])

        if oligomeric_details:
            biological_assembly += f" ({oligomeric_details.lower()})"

    ligand_ids = [ligand["id"] for ligand in entry_ligands(entry)]

    return {
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "label_chain_id": label_chain_id,
        "polymer_entity_id": f"{pdb_id}_{entity_number}",
        "protein_name": (
            (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description")
            or ""
        ),
        "release_date": release_date,
        "sequence": sequence,
        "length": int(length),
        "experimental_method": "; ".join(methods),
        "resolution": min(resolutions) if resolutions else None,
        "resolved_fraction": resolved_fraction,
        "longest_unobserved_segment": longest_gap,
        "organism": "; ".join(organisms),
        "uniprot_accession": (
            uniprot_accessions[0] if uniprot_accessions else ""
        ),
        "number_of_chains": info.get("polymer_entity_instance_count"),
        "ligands": "; ".join(sorted(set(ligand_ids))),
        "ligand_count": len(ligand_ids),
        "biological_assembly": biological_assembly,
        "standard_sequence": is_standard_protein_sequence(sequence),
        "sequence_cluster": None,
        "training_or_test_split": "unassigned",
        "priority_target": bool(priority_target),
        "priority_name": priority_name,
        "forced_split": forced_split,
    }


# ============================================================
# UNIPROT AND OPM ANNOTATIONS
# ============================================================

def feature_range(feature: dict):
    """Return (start, end) UniProt positions of a feature, or None."""
    location = feature.get("location") or {}
    start = (location.get("start") or {}).get("value")
    end = (location.get("end") or {}).get("value")

    if start is None or end is None:
        return None

    return int(start), int(end)


def parse_uniprot(data: dict | None) -> dict:
    """
    Extract the UniProt annotations relevant to the Stage 1 biological
    restrictions: membrane topology, lipid anchors, subcellular location,
    disordered regions, and subunit composition.
    """

    flags = {
        "available": False,
        "transmembrane_regions": [],
        "has_lipidation": False,
        "has_gpi_anchor": False,
        # Positions of the lipidation/GPI features, so the screen can ask
        # whether they fall inside the crystallised construct. Keyword-only
        # evidence contributes no position and leaves this list short.
        "lipidation_regions": [],
        "membrane_locations": [],
        "disordered_regions": [],
        "subunit_text": "",
    }

    if not data:
        return flags

    flags["available"] = True

    for feature in data.get("features") or []:

        feature_type = str(feature.get("type", "")).upper()
        description = str(feature.get("description", ""))
        region = feature_range(feature)

        if feature_type in {"TRANSMEMBRANE", "INTRAMEMBRANE"}:
            flags["transmembrane_regions"].append(region)

        elif feature_type == "LIPIDATION":
            flags["has_lipidation"] = True
            flags["lipidation_regions"].append(region)

            if "GPI" in description.upper():
                flags["has_gpi_anchor"] = True

        elif (
            feature_type == "REGION"
            and "disordered" in description.lower()
            and region is not None
        ):
            flags["disordered_regions"].append(region)

    for keyword in data.get("keywords") or []:

        name = keyword.get("name", "")

        if name in UNIPROT_LIPID_KEYWORDS:
            flags["has_lipidation"] = True

        if "GPI" in name.upper():
            flags["has_gpi_anchor"] = True

    subunit_texts = []

    for comment in data.get("comments") or []:

        comment_type = comment.get("commentType", "")

        if comment_type == "SUBCELLULAR LOCATION":
            for location in comment.get("subcellularLocations") or []:
                value = (location.get("location") or {}).get("value", "")

                if any(
                    token in value.lower()
                    for token in MEMBRANE_SUBLOC_TOKENS
                ):
                    flags["membrane_locations"].append(value)

        elif comment_type == "SUBUNIT":
            subunit_texts.extend(
                text.get("value", "")
                for text in comment.get("texts") or []
            )

    flags["subunit_text"] = " ".join(subunit_texts)

    return flags


def fetch_uniprot_batch(accessions: list[str]) -> list[dict] | None:
    """
    Download up to UNIPROT_BATCH_SIZE UniProt entries in one request,
    following pagination links. Returns None if UniProt stays unavailable.
    """

    url = UNIPROT_BATCH_API
    params = {
        "accessions": ",".join(accessions),
        "format": "json",
        "size": len(accessions),
    }
    entries = []

    while url:

        for attempt in range(UNIPROT_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=60)

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"{response.status_code} from UniProt"
                    )

                response.raise_for_status()
                break

            except requests.RequestException as exc:
                if attempt == UNIPROT_RETRIES - 1:
                    print(f"UniProt batch failed: {exc}")
                    return None

                wait = 5 * 2 ** attempt
                print(f"UniProt request failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)

        entries.extend(response.json().get("results") or [])

        # The next-page URL already carries the query parameters.
        url = response.links.get("next", {}).get("url")
        params = None

    return entries


def fetch_uniprot_annotations(accessions: list[str]) -> dict:
    """
    Download and parse UniProt entries for many accessions, in batches.

    Returns {accession: flags}. An accession whose entry could not be
    retrieved gets available=False, so the screen asks for a manual check
    rather than silently passing the protein.
    """

    annotations = {}

    for start in range(0, len(accessions), UNIPROT_BATCH_SIZE):

        batch = accessions[start:start + UNIPROT_BATCH_SIZE]
        entries = fetch_uniprot_batch(batch) or []

        # Requested accessions may be secondary accessions of an entry.
        by_accession = {}
        for entry in entries:
            for accession in (
                [entry.get("primaryAccession")]
                + (entry.get("secondaryAccessions") or [])
            ):
                if accession:
                    by_accession[accession] = entry

        for accession in batch:
            annotations[accession] = parse_uniprot(
                by_accession.get(accession)
            )

        print(
            f"UniProt: {min(start + len(batch), len(accessions))}"
            f"/{len(accessions)}"
        )

    missing = [
        accession
        for accession, flags in annotations.items()
        if not flags["available"]
    ]

    if missing:
        print(
            f"WARNING: no UniProt entry retrieved for {len(missing)} "
            "accession(s); these proteins will be FLAGged for manual review."
        )

    return annotations


def check_opm(pdb_id: str):
    """
    True if the entry is in the OPM membrane-protein database, False if
    not, None if OPM could not be reached.
    """
    return url_exists(OPM_PDB_URL.format(pdb_id=pdb_id.lower()))


def uniprot_construct_positions(entity: dict, accession: str) -> set[int]:
    """
    UniProt residue numbers covered by the crystallised construct,
    from RCSB's entity-to-UniProt sequence alignment.
    """

    positions = set()

    for alignment in entity.get("rcsb_polymer_entity_align") or []:

        database = str(alignment.get("reference_database_name") or "")

        if database.upper() != "UNIPROT":
            continue

        if (
            accession
            and alignment.get("reference_database_accession") != accession
        ):
            continue

        for region in alignment.get("aligned_regions") or []:

            start = region.get("ref_beg_seq_id")
            length = region.get("length")

            if start is None or not length:
                continue

            positions.update(range(int(start), int(start) + int(length)))

    return positions


def count_overlap(regions, positions: set[int]) -> int:
    """Number of construct positions that fall inside any region."""
    return sum(
        1
        for position in positions
        if any(start <= position <= end for start, end in regions)
    )


def membrane_database_annotations(entity: dict) -> list[str]:
    """Membrane-protein databases (PDBTM, OPM, ...) annotating an entity."""

    hits = set()

    for annotation in entity.get("rcsb_polymer_entity_annotation") or []:
        for key in ("type", "provenance_source"):
            value = str(annotation.get(key) or "").upper()

            if value in MEMBRANE_DATABASES:
                hits.add(value)

    return sorted(hits)


# ============================================================
# LIGAND CLASSIFICATION
# ============================================================

def count_heavy_atoms(formula: str) -> int:
    """Count non-hydrogen atoms in a CCD formula, e.g. 'C21 H27 N7 O14 P2'."""
    total = 0

    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if element in {"H", "D"}:
            continue

        total += int(count) if count else 1

    return total


def to_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def entry_ligands(entry: dict) -> list[dict]:
    """
    Non-trivial ligands of an entry as {id, description, weight (Da),
    heavy_atoms}: small molecules (excluding buffers, ions, and
    crystallisation additives) plus oligosaccharides.
    """

    ligands = []

    for ligand in entry.get("nonpolymer_entities") or []:

        comp = (ligand.get("nonpolymer_comp") or {}).get("chem_comp") or {}
        ccd = comp.get("id") or ""

        if not ccd or ccd in TRIVIAL_LIGANDS:
            continue

        weight = to_float(comp.get("formula_weight"))
        heavy_atoms = count_heavy_atoms(comp.get("formula") or "")

        ligands.append(
            {
                "id": ccd,
                "description": (
                    f"{ccd} ({comp.get('name', '')}; "
                    f"{weight:.0f} Da; {heavy_atoms} heavy atoms)"
                ),
                "weight": weight,
                "heavy_atoms": heavy_atoms,
            }
        )

    # Bound sugars and glycans are separate "branched" entities in RCSB,
    # not chemical components, so they never appear among the non-polymer
    # entities. Their formula_weight is in kDa and no formula is given, so
    # they are judged on weight alone.
    for branched in entry.get("branched_entities") or []:

        info = branched.get("rcsb_branched_entity") or {}
        monomers = (
            branched.get("rcsb_branched_entity_container_identifiers") or {}
        ).get("chem_comp_monomers") or []

        ligand_id = f"oligosaccharide({'/'.join(sorted(monomers))})"
        weight = to_float(info.get("formula_weight")) * 1000

        ligands.append(
            {
                "id": ligand_id,
                "description": f"{ligand_id} ({weight:.0f} Da)",
                "weight": weight,
                "heavy_atoms": 0,
            }
        )

    return ligands


def classify_ligands(entry: dict):
    """
    Classify an entry's non-trivial ligands by molecular weight and
    heavy-atom count.

    Returns (verdict, notes) where verdict is PASS / FLAG / FAIL.
    """

    verdict = "PASS"
    notes = []

    for ligand in entry_ligands(entry):

        weight = ligand["weight"]
        heavy_atoms = ligand["heavy_atoms"]
        description = ligand["description"]

        if weight >= LARGE_LIGAND_MW or heavy_atoms > LARGE_LIGAND_HEAVY_ATOMS:
            notes.append(f"large ligand {description}")
            verdict = "FAIL"

        elif (
            weight >= MEDIUM_LIGAND_MW
            or heavy_atoms > MEDIUM_LIGAND_HEAVY_ATOMS
        ):
            notes.append(f"medium ligand {description}")

            if verdict != "FAIL":
                verdict = "FLAG"

    return verdict, notes


# ============================================================
# BIOLOGICAL SCREENING
# ============================================================

CRITERIA = [
    "monomeric",
    "not_obligate_complex",
    "soluble",
    "not_membrane",
    "length_80_400",
    "sequence_length_consistent",
    "no_terminal_tag",
    "resolved_ge_0_90",
    "structure_quality",
    "not_heavily_disordered",
    "no_large_ligand",
]


def screen_candidate(
    row: dict,
    entry: dict,
    entity: dict,
    uniprot: dict | None,
    in_opm,
) -> dict:
    """
    Apply the Stage 1 initial biological restrictions.

    Every criterion receives PASS, FLAG (needs manual review), or FAIL.
    The overall screening_decision is EXCLUDE if any criterion fails,
    FLAG if any criterion is flagged, and INCLUDE otherwise.
    """

    verdicts = {}
    notes = []
    flags = []

    def flag(criterion: str, reason: str):
        if verdicts.get(criterion) != "FAIL":
            verdicts[criterion] = "FLAG"
        flags.append(reason)

    def fail(criterion: str, reason: str):
        verdicts[criterion] = "FAIL"
        notes.append(reason)

    accession = row["uniprot_accession"]

    # --------------------------------------------------------
    # Monomeric: the preferred biological assembly must contain exactly
    # one protein chain and no nucleic acid. Other assemblies that are
    # oligomeric (e.g. software-predicted interfaces) are flagged.
    # --------------------------------------------------------

    info = assembly_info(preferred_assembly(entry))
    protein_chains = info.get("polymer_entity_instance_count_protein")

    verdicts["monomeric"] = "PASS"

    if protein_chains is None:
        flag("monomeric", "biological assembly composition unavailable")

    elif protein_chains != 1:
        fail(
            "monomeric",
            f"assembly {info.get('assembly_id')} has "
            f"{protein_chains} protein chains",
        )

    else:
        other_oligomers = [
            str(assembly_info(assembly).get("assembly_id"))
            for assembly in entry.get("assemblies") or []
            if (
                assembly_info(assembly)
                .get("polymer_entity_instance_count_protein")
                or 0
            ) > 1
        ]

        if other_oligomers:
            flag(
                "monomeric",
                "alternative oligomeric assembly "
                + ", ".join(other_oligomers),
            )

    # --------------------------------------------------------
    # Not an obligate complex
    # --------------------------------------------------------

    verdicts["not_obligate_complex"] = "PASS"

    if nucleic_acid_count(info):
        fail("not_obligate_complex", "assembly contains nucleic acid")

    if verdicts["monomeric"] == "FAIL":
        verdicts["not_obligate_complex"] = "FAIL"

    subunit_text = (uniprot or {}).get("subunit_text", "")
    subunit_match = OBLIGATE_COMPLEX_PATTERN.search(subunit_text)

    if subunit_match:
        flag(
            "not_obligate_complex",
            f"UniProt subunit: '{subunit_match.group(0)}'",
        )

    # --------------------------------------------------------
    # Membrane protein / soluble
    # --------------------------------------------------------

    verdicts["not_membrane"] = "PASS"
    verdicts["soluble"] = "PASS"

    if in_opm:
        fail("not_membrane", "entry is in the OPM membrane database")
    elif in_opm is None:
        flag("not_membrane", "OPM membrane database lookup failed")

    membrane_dbs = membrane_database_annotations(entity)

    if membrane_dbs:
        fail(
            "not_membrane",
            "RCSB membrane annotation: " + ", ".join(membrane_dbs),
        )

    construct_positions = (
        uniprot_construct_positions(entity, accession)
        if accession
        else set()
    )

    if uniprot and uniprot["available"]:

        tm_regions = uniprot["transmembrane_regions"]

        if tm_regions:
            known_regions = [r for r in tm_regions if r is not None]

            # Fail unless every TM segment is known to lie outside the
            # construct; then it is only a soluble domain of a TM protein.
            if (
                not construct_positions
                or len(known_regions) < len(tm_regions)
                or count_overlap(known_regions, construct_positions)
            ):
                fail(
                    "not_membrane",
                    "UniProt transmembrane segment in construct",
                )
            else:
                flag(
                    "not_membrane",
                    "soluble domain of a transmembrane protein "
                    "(TM segment outside construct)",
                )

        # Lipidation is judged against the construct, like the
        # transmembrane check above: an N-terminal myristoylation site or a
        # C-terminal GPI signal that was not crystallised does not make the
        # modelled fragment a membrane-associated protein. Keyword-only
        # evidence carries no position, so it can only be flagged for
        # manual review.
        if uniprot["has_gpi_anchor"] or uniprot["has_lipidation"]:

            label = (
                "GPI anchor"
                if uniprot["has_gpi_anchor"]
                else "lipid anchor / lipidation"
            )

            lipid_regions = uniprot["lipidation_regions"]
            known_lipid = [r for r in lipid_regions if r is not None]

            if (
                not construct_positions
                or not known_lipid
                or len(known_lipid) < len(lipid_regions)
            ):
                flag(
                    "soluble",
                    f"UniProt {label}; position could not be checked "
                    "against the construct",
                )

            elif count_overlap(known_lipid, construct_positions):
                fail("soluble", f"UniProt {label} site inside construct")

            else:
                flag(
                    "soluble",
                    f"UniProt {label} site outside construct",
                )

        if uniprot["membrane_locations"]:
            flag(
                "soluble",
                "UniProt membrane/cell-surface location: "
                + "; ".join(uniprot["membrane_locations"][:3]),
            )

    elif accession:
        flag(
            "soluble",
            f"UniProt {accession} unavailable; check membrane "
            "association manually",
        )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if MIN_LENGTH <= row["length"] <= MAX_LENGTH:
        verdicts["length_80_400"] = "PASS"
    else:
        fail(
            "length_80_400",
            f"length {row['length']} outside [{MIN_LENGTH}, {MAX_LENGTH}]",
        )

    # --------------------------------------------------------
    # Construct sequence sanity checks
    # --------------------------------------------------------

    sequence = row["sequence"]

    verdicts["sequence_length_consistent"] = "PASS"

    if len(sequence) != row["length"]:
        flag(
            "sequence_length_consistent",
            f"sequence string length {len(sequence)} "
            f"!= metadata length {row['length']}",
        )

    verdicts["no_terminal_tag"] = "PASS"

    termini = (sequence[:TAG_WINDOW], sequence[-TAG_WINDOW:])
    tags = [
        name
        for name, pattern in EXPRESSION_TAGS.items()
        if any(pattern.search(terminus) for terminus in termini)
    ]

    if tags:
        flag(
            "no_terminal_tag",
            "possible terminal expression tag: " + ", ".join(tags),
        )

    # --------------------------------------------------------
    # Largely structurally resolved
    # --------------------------------------------------------

    resolved = row["resolved_fraction"]

    if resolved is not None and resolved >= MIN_RESOLVED_FRACTION:
        verdicts["resolved_ge_0_90"] = "PASS"
    else:
        fail(
            "resolved_ge_0_90",
            "resolved fraction unavailable"
            if resolved is None
            else f"resolved fraction {resolved:.3f} "
                 f"< {MIN_RESOLVED_FRACTION}",
        )

    # --------------------------------------------------------
    # Experimental quality
    # --------------------------------------------------------

    verdicts["structure_quality"] = "PASS"

    # experimental_method is a "; "-joined string, so a multi-method entry
    # ("X-RAY DIFFRACTION; NEUTRON DIFFRACTION") must be compared method by
    # method rather than as one value.
    methods = {
        method.strip()
        for method in str(row["experimental_method"] or "").split(";")
        if method.strip()
    }

    if not methods or not methods <= ALLOWED_METHODS:
        fail(
            "structure_quality",
            "method "
            + ("; ".join(sorted(methods - ALLOWED_METHODS)) or "unknown")
            + " not allowed",
        )

    if row["resolution"] is None:
        fail("structure_quality", "resolution unavailable")
    elif row["resolution"] > MAX_RESOLUTION:
        fail(
            "structure_quality",
            f"resolution {row['resolution']:.2f} Å > {MAX_RESOLUTION} Å",
        )

    if not row["standard_sequence"]:
        fail("structure_quality", "non-standard residues in sequence")

    # --------------------------------------------------------
    # Not heavily disordered
    # --------------------------------------------------------

    verdicts["not_heavily_disordered"] = "PASS"

    if verdicts["resolved_ge_0_90"] == "FAIL":
        fail(
            "not_heavily_disordered",
            "low resolved fraction consistent with disorder",
        )

    longest_gap = row["longest_unobserved_segment"]

    if longest_gap is not None and longest_gap > MAX_UNOBSERVED_SEGMENT:
        flag(
            "not_heavily_disordered",
            f"{longest_gap}-residue unmodelled segment",
        )

    if uniprot and uniprot["available"] and construct_positions:

        disordered = count_overlap(
            uniprot["disordered_regions"],
            construct_positions,
        )

        disordered_fraction = disordered / max(row["length"], 1)

        if disordered_fraction > MAX_DISORDER_ANNOTATION_FRACTION:
            flag(
                "not_heavily_disordered",
                f"UniProt disordered regions cover "
                f"{disordered_fraction:.0%} of construct",
            )

    # --------------------------------------------------------
    # Not dependent on large ligands
    # --------------------------------------------------------

    ligand_verdict, ligand_notes = classify_ligands(entry)

    verdicts["no_large_ligand"] = ligand_verdict

    if ligand_verdict == "FAIL":
        notes.extend(ligand_notes)
    elif ligand_verdict == "FLAG":
        flags.extend(
            f"{note}; confirm not required for folding"
            for note in ligand_notes
        )

    # --------------------------------------------------------
    # Overall decision
    # --------------------------------------------------------

    values = [verdicts[criterion] for criterion in CRITERIA]

    if "FAIL" in values:
        decision = "EXCLUDE"
    elif "FLAG" in values:
        decision = "FLAG"
    else:
        decision = "INCLUDE"

    return {
        **{criterion: verdicts[criterion] for criterion in CRITERIA},
        "screening_decision": decision,
        "screening_notes": "; ".join(notes),
        "flag_reasons": "; ".join(flags),
    }


# ============================================================
# SEQUENCE CLUSTERS
# ============================================================

def download_sequence_clusters() -> tuple[dict, str]:
    """
    Download RCSB's current 30%-sequence-identity clusters.

    Returns ({polymer entity ID: cluster ID}, snapshot date), where the
    snapshot date is the file's Last-Modified header. RCSB rebuilds these
    clusters as the PDB grows, so the split hashes only mean something
    alongside a record of which cluster snapshot produced them.
    """

    print("\nDownloading RCSB 30% sequence clusters...")

    response = requests.get(CLUSTER_30_URL, timeout=120)
    response.raise_for_status()

    snapshot = response.headers.get("Last-Modified", "unknown")

    entity_to_cluster = {}
    n_clusters = 0

    # One cluster per line, listing its member entities (e.g. "1ABC_1").
    for cluster_number, line in enumerate(response.text.splitlines(), start=1):

        entities = [x.upper() for x in line.split() if x.strip()]

        if not entities:
            continue

        cluster_id = f"seq30_{cluster_number:06d}"
        n_clusters += 1

        for entity in entities:
            entity_to_cluster[entity] = cluster_id

    print(
        f"Loaded {n_clusters:,} 30% sequence clusters "
        f"(snapshot: {snapshot})."
    )

    return entity_to_cluster, snapshot


# ============================================================
# PRE-CUTOFF SEQUENCE SIMILARITY (TEST A / TEST B RULE)
# ============================================================

def alignment_metrics(result: dict) -> list[dict]:
    """Extract identity, target coverage, and e-value from one RCSB hit."""

    alignments = []

    for service in result.get("services", []):
        if service.get("service_type") != "sequence":
            continue

        for node in service.get("nodes", []):
            for match in node.get("match_context", []):

                query_length = int(match.get("query_length") or 0)

                if not query_length:
                    continue

                query_span = max(
                    0,
                    int(match.get("query_end") or 0)
                    - int(match.get("query_beg") or 0)
                    + 1,
                )

                alignments.append(
                    {
                        "hit": result.get("identifier", ""),
                        "identity": float(
                            match.get("sequence_identity") or 0.0
                        ),
                        # The query is the post-cutoff candidate, so this is
                        # the fraction of that target covered by the alignment.
                        "coverage": query_span / query_length,
                        "evalue": float(match.get("evalue") or 0.0),
                    }
                )

    return alignments


def query_precutoff_similarity(
    sequence: str,
    retries: int = SEQUENCE_SEARCH_RETRIES,
):
    """
    Search pre-cutoff PDB proteins for a >=30%-identity / >=80%-target-
    coverage match to one post-cutoff candidate.

    Returns (best_hit, identity, coverage, evalue, status), where status is:
      - similar_hit: at least one pre-cutoff alignment reaches both thresholds
      - no_similar_hit: the complete search has no alignment reaching both
      - incomplete_search: more hits existed than were inspected
      - api_error: the search could not be completed
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
                terminal(
                    "rcsb_accession_info.initial_release_date",
                    "less_or_equal",
                    f"{CUTOFF_DATE}T00:00:00Z",
                ),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": MAX_SEQUENCE_SEARCH_RESULTS},
            "scoring_strategy": "sequence",
            # Verbose results include the alignment details (match_context).
            "results_verbosity": "verbose",
        },
    }

    for attempt in range(retries):
        try:
            response = requests.post(SEARCH_API, json=payload, timeout=60)
            response.raise_for_status()

            # RCSB answers 204 No Content (empty body) when nothing
            # matches, i.e. no pre-cutoff protein reaches 30% identity.
            if response.status_code == 204:
                return "", 0.0, 0.0, 0.0, "no_similar_hit"

            data = response.json()
            results = data.get("result_set", [])
            total_count = int(data.get("total_count") or len(results))

            alignments = [
                alignment
                for result in results
                for alignment in alignment_metrics(result)
            ]

            if not alignments:
                return "", 0.0, 0.0, 0.0, "no_similar_hit"

            qualifying_hits = [
                alignment
                for alignment in alignments
                if alignment["identity"] >= TESTB_MAX_PIDENT
                and alignment["coverage"] >= TESTB_MIN_COVERAGE
            ]

            if qualifying_hits:
                best = max(
                    qualifying_hits,
                    key=lambda a: (a["identity"], a["coverage"]),
                )
                status = "similar_hit"

            else:
                # Report the best-covering near miss for reference.
                best = max(
                    alignments,
                    key=lambda a: (a["coverage"], a["identity"]),
                )
                status = (
                    "incomplete_search"
                    if total_count > len(results)
                    else "no_similar_hit"
                )

            return (
                best["hit"],
                best["identity"],
                best["coverage"],
                best["evalue"],
                status,
            )

        except Exception as exc:
            if attempt == retries - 1:
                print(
                    "Pre-cutoff sequence search failed after "
                    f"{retries} attempts: {exc}"
                )
                break

            wait = 2 ** attempt
            print(
                f"Pre-cutoff sequence search failed: {exc}; "
                f"retrying in {wait}s..."
            )
            time.sleep(wait)

    return "", 0.0, 0.0, 0.0, "api_error"


def annotate_precutoff_similarity(df):
    """
    Apply the 30%-identity / 80%-target-coverage rule to every post-cutoff
    candidate and record its best pre-cutoff alignment.
    """

    df = df.copy()

    df["best_precutoff_hit"] = ""
    df["best_precutoff_pident"] = None
    df["best_precutoff_coverage"] = None
    df["best_precutoff_evalue"] = None
    df["precutoff_search_status"] = ""

    post_indices = df.index[df["release_date"] > CUTOFF_DATE].tolist()

    print(
        "\nChecking direct pre-cutoff sequence similarity for "
        f"{len(post_indices)} post-cutoff candidates "
        f"(>={TESTB_MAX_PIDENT:.0%} identity, "
        f">={TESTB_MIN_COVERAGE:.0%} target coverage)..."
    )

    sequences = {
        index: str(df.at[index, "sequence"] or "").strip()
        for index in post_indices
    }

    # Search each distinct sequence once.
    search_results = run_threaded(
        query_precutoff_similarity,
        sorted({sequence for sequence in sequences.values() if sequence}),
        "Pre-cutoff sequence search",
    )

    for index, sequence in sequences.items():

        hit, identity, coverage, evalue, status = search_results.get(
            sequence,
            ("", 0.0, 0.0, 0.0, "api_error"),
        )

        df.at[index, "best_precutoff_hit"] = hit
        df.at[index, "best_precutoff_pident"] = identity
        df.at[index, "best_precutoff_coverage"] = coverage
        df.at[index, "best_precutoff_evalue"] = evalue
        df.at[index, "precutoff_search_status"] = status

    post_status = df.loc[post_indices, "precutoff_search_status"]

    print(
        "Pre-cutoff sequence search statuses: "
        f"{post_status.value_counts().to_dict()}"
    )

    unclassified = df.loc[post_indices][
        ~post_status.isin(["similar_hit", "no_similar_hit"])
    ]

    if not unclassified.empty:
        print(
            f"WARNING: {len(unclassified)} post-cutoff candidate(s) could "
            "not be classified and are excluded from Test A and Test B: "
            + ", ".join(
                f"{row.pdb_id} ({row.precutoff_search_status})"
                for row in unclassified.itertuples()
            )
        )

    df.loc[
        df["release_date"] <= CUTOFF_DATE,
        "precutoff_search_status",
    ] = "not_applicable_pre_cutoff"

    return df


# ============================================================
# FILTERING
# ============================================================

def passes_filters(row):
    """
    Decide whether a screened candidate may be sampled into a split.

    INCLUDE candidates are always eligible. FLAG candidates (which need
    manual review) are eligible only where the configuration allows it.
    By default they are excluded from every split.
    """

    if pd.isna(row["sequence_cluster"]):
        return False

    decision = row["screening_decision"]

    if decision == "INCLUDE":
        return True

    if decision != "FLAG":
        return False

    if row["release_date"] <= CUTOFF_DATE:
        return ALLOW_FLAGGED_IN_DEV_VAL

    return ALLOW_FLAGGED_IN_TEST


# ============================================================
# SPLIT CREATION
# ============================================================

def nonempty_uniprots(df) -> set[str]:
    """Set of the non-empty UniProt accessions in df."""
    accessions = df["uniprot_accession"].fillna("").astype(str).str.strip()
    return set(accessions[accessions != ""].tolist())


def take_cluster_representatives(
    df,
    n,
    rng,
    excluded_clusters=None,
    excluded_uniprots=None,
):
    """
    Select at most n structures, at most one from each 30% sequence
    cluster and each non-empty UniProt accession, skipping any cluster or
    accession already used.
    """

    excluded_clusters = excluded_clusters or set()
    excluded_uniprots = excluded_uniprots or set()

    uniprot_accessions = (
        df["uniprot_accession"].fillna("").astype(str).str.strip()
    )

    working = df[
        (~df["sequence_cluster"].isin(excluded_clusters))
        & (
            (uniprot_accessions == "")
            | (~uniprot_accessions.isin(excluded_uniprots))
        )
    ].copy()

    # Shuffle first so we do not always keep the same PDB
    # when a cluster or UniProt accession contains multiple candidates.
    working = working.sample(frac=1, random_state=rng.randint(0, 2**32 - 1))

    working = working.drop_duplicates(
        subset=["sequence_cluster"],
        keep="first",
    )

    working_uniprots = (
        working["uniprot_accession"].fillna("").astype(str).str.strip()
    )

    duplicate_uniprot = (
        working_uniprots.ne("") & working_uniprots.duplicated(keep="first")
    )

    working = working[~duplicate_uniprot]

    # Reshuffle so the order no longer reflects the de-duplication.
    working = working.sample(frac=1, random_state=rng.randint(0, 2**32 - 1))

    return working.head(n).copy()


def create_splits(df):
    """
    Create the development, validation, Test A, and Test B splits, with no
    30%-cluster or non-empty UniProt overlap between them.

      - development / validation: pre-cutoff proteins
      - Test A: post-cutoff proteins with a pre-cutoff PDB match at >=30%
        identity over >=80% of the target (precutoff_search_status
        "similar_hit")
      - Test B: post-cutoff proteins with no such match ("no_similar_hit")

    Priority targets (special proteins selected for investigation) are
    always included. One with a forced_split goes to that split; any other
    post-cutoff priority target follows the Test A / Test B rule above.
    Their clusters and UniProt accessions are reserved first, so no other
    split can contain a protein similar to them.
    """

    rng = random.Random(RANDOM_SEED)

    pre = df[df["release_date"] <= CUTOFF_DATE].copy()
    post = df[df["release_date"] > CUTOFF_DATE].copy()
    priority = df[df["priority_target"] == True].copy()

    priority_clusters = set(priority["sequence_cluster"].dropna().tolist())
    priority_uniprots = nonempty_uniprots(priority)

    # --------------------------------------------------------
    # DEVELOPMENT AND VALIDATION
    # --------------------------------------------------------

    pre_pool = pre[pre["priority_target"] != True]

    development = take_cluster_representatives(
        pre_pool,
        N_DEVELOPMENT,
        rng,
        excluded_clusters=priority_clusters,
        excluded_uniprots=priority_uniprots,
    )

    used_clusters = set(development["sequence_cluster"]) | priority_clusters
    used_uniprots = nonempty_uniprots(development) | priority_uniprots

    validation = take_cluster_representatives(
        pre_pool,
        N_VALIDATION,
        rng,
        excluded_clusters=used_clusters,
        excluded_uniprots=used_uniprots,
    )

    used_clusters.update(validation["sequence_cluster"])
    used_uniprots.update(nonempty_uniprots(validation))

    # --------------------------------------------------------
    # POST-CUTOFF PRIORITY TARGETS
    # --------------------------------------------------------

    priority_post = post[post["priority_target"] == True].copy()
    unforced = priority_post["forced_split"].isna()
    status = priority_post["precutoff_search_status"]

    unresolved_priority = priority_post[
        unforced & ~status.isin(["similar_hit", "no_similar_hit"])
    ]

    if not unresolved_priority.empty:
        raise RuntimeError(
            "Could not classify priority target(s) with the direct "
            "30/80 pre-cutoff sequence search: "
            + ", ".join(unresolved_priority["pdb_id"].tolist())
        )

    priority_test_b = priority_post[
        (priority_post["forced_split"] == "test_b")
        | (unforced & (status == "no_similar_hit"))
    ].copy()

    priority_test_a = priority_post[
        (priority_post["forced_split"] == "test_a")
        | (unforced & (status == "similar_hit"))
    ].copy()

    # --------------------------------------------------------
    # TEST B: post-cutoff, with no pre-cutoff PDB sequence reaching
    # >=30% identity over >=80% of the target.
    # --------------------------------------------------------

    difficult_pool = post[
        (post["precutoff_search_status"] == "no_similar_hit")
        & (post["priority_target"] != True)
    ]

    sampled_test_b = take_cluster_representatives(
        difficult_pool,
        max(N_TEST_B - len(priority_test_b), 0),
        rng,
        excluded_clusters=used_clusters,
        excluded_uniprots=used_uniprots,
    )

    test_b = pd.concat([priority_test_b, sampled_test_b], axis=0)

    used_clusters.update(test_b["sequence_cluster"].dropna().tolist())
    used_uniprots.update(nonempty_uniprots(test_b))

    # --------------------------------------------------------
    # TEST A: post-cutoff, with a pre-cutoff PDB sequence reaching
    # >=30% identity over >=80% of the target.
    # --------------------------------------------------------

    temporal_pool = post[
        (post["precutoff_search_status"] == "similar_hit")
        & (post["priority_target"] != True)
    ]

    sampled_test_a = take_cluster_representatives(
        temporal_pool,
        max(N_TEST_A - len(priority_test_a), 0),
        rng,
        excluded_clusters=used_clusters,
        excluded_uniprots=used_uniprots,
    )

    test_a = pd.concat([priority_test_a, sampled_test_a], axis=0)

    # --------------------------------------------------------
    # Assign split names
    # --------------------------------------------------------

    df = df.copy()
    df["training_or_test_split"] = "unassigned"

    splits = {
        "development": development,
        "validation": validation,
        "test_a": test_a,
        "test_b": test_b,
    }

    for name, split in splits.items():
        df.loc[split.index, "training_or_test_split"] = name

    # Return the splits as rows of df so they carry the split name.
    return (df, *(df.loc[split.index].copy() for split in splits.values()))


# ============================================================
# MAIN
# ============================================================

def main():

    rng = random.Random(RANDOM_SEED)

    print("=" * 60)
    print("STAGE 1: BUILD CLEAN BENCHMARK DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Search PDB
    # --------------------------------------------------------

    print("\nSearching pre-cutoff PDB entries...")
    pre_ids = search_candidate_pdb_ids(pre_cutoff=True)
    print(f"Found {len(pre_ids):,} searchable pre-cutoff entries.")

    print("\nSearching post-cutoff PDB entries...")
    post_ids = search_candidate_pdb_ids(pre_cutoff=False)
    print(f"Found {len(post_ids):,} searchable post-cutoff entries.")

    # Sort IDs before seeded sampling so the same search results
    # produce the same sampled candidates across runs.
    pre_ids = sorted(pre_ids)
    post_ids = sorted(post_ids)

    if len(pre_ids) > N_PRE_CANDIDATES:
        pre_ids = rng.sample(pre_ids, N_PRE_CANDIDATES)

    if len(post_ids) > N_POST_CANDIDATES:
        post_ids = rng.sample(post_ids, N_POST_CANDIDATES)

    # Remove priority targets from the random pool if they happened
    # to be sampled already. They will be added explicitly below.
    metadata_jobs = [
        {
            "pdb_id": pdb_id,
            "target_chain_id": None,
            "priority_target": False,
            "priority_name": "",
            "forced_split": None,
        }
        for pdb_id in pre_ids + post_ids
        if pdb_id not in PRIORITY_TARGETS
    ]

    for pdb_id, spec in PRIORITY_TARGETS.items():
        metadata_jobs.append(
            {
                "pdb_id": pdb_id,
                "target_chain_id": spec["chain_id"],
                "priority_target": True,
                "priority_name": spec["protein_name"],
                "forced_split": spec.get("forced_split"),
            }
        )

    print(
        f"\nRetrieving metadata for {len(metadata_jobs)} candidates "
        f"(including {len(PRIORITY_TARGETS)} required priority targets)..."
    )

    # --------------------------------------------------------
    # 2. Download RCSB metadata
    # --------------------------------------------------------

    entries = fetch_entry_metadata(job["pdb_id"] for job in metadata_jobs)

    candidates = []

    for job in metadata_jobs:

        pdb_id = job["pdb_id"]
        entry = entries.get(pdb_id)

        if entry is None:
            if job["priority_target"]:
                raise RuntimeError(
                    f"Failed to retrieve required priority target: {pdb_id}"
                )
            continue

        entity, instance = select_entity(entry, job["target_chain_id"])

        if entity is None:
            continue

        row = extract_metadata(
            pdb_id,
            entry,
            entity,
            instance,
            job["priority_target"],
            job["priority_name"],
            job["forced_split"],
        )

        candidates.append((row, entry, entity))

    if not candidates:
        raise RuntimeError("No candidate structures were retrieved.")

    # --------------------------------------------------------
    # 3. UniProt and OPM annotations
    # --------------------------------------------------------

    accessions = sorted(
        {
            row["uniprot_accession"]
            for row, _, _ in candidates
            if row["uniprot_accession"]
        }
    )

    print(
        f"\nRetrieving UniProt annotations for "
        f"{len(accessions)} accessions..."
    )

    uniprot_annotations = fetch_uniprot_annotations(accessions)

    print(
        f"\nChecking {len(candidates)} entries against "
        "the OPM membrane-protein database..."
    )

    opm_hits = run_threaded(
        check_opm,
        sorted({row["pdb_id"] for row, _, _ in candidates}),
        "OPM",
    )

    # --------------------------------------------------------
    # 4. Biological screening
    # --------------------------------------------------------

    rows = []

    for row, entry, entity in candidates:

        row.update(
            screen_candidate(
                row,
                entry,
                entity,
                uniprot_annotations.get(row["uniprot_accession"]),
                opm_hits.get(row["pdb_id"]),
            )
        )

        # Priority targets are kept whatever the automated screen says;
        # record that they were kept despite the screen.
        if row["priority_target"] and row["screening_decision"] != "INCLUDE":
            row["flag_reasons"] = "; ".join(
                x
                for x in (
                    row["flag_reasons"],
                    "PRIORITY_CASE_OVERRIDE: retained despite "
                    f"automated decision {row['screening_decision']}",
                )
                if x
            )

        rows.append(row)

    # Sort by PDB ID so later seeded sampling is reproducible.
    all_candidates = (
        pd.DataFrame(rows)
        .sort_values("pdb_id")
        .reset_index(drop=True)
    )

    decision_counts = all_candidates["screening_decision"].value_counts()

    print(
        "\nBiological screen: "
        + ", ".join(
            f"{decision} {decision_counts.get(decision, 0)}"
            for decision in ("INCLUDE", "FLAG", "EXCLUDE")
        )
    )

    # --------------------------------------------------------
    # 5. Add 30% sequence clusters
    # --------------------------------------------------------

    entity_to_cluster, cluster_snapshot = download_sequence_clusters()

    all_candidates["sequence_cluster"] = (
        all_candidates["polymer_entity_id"].str.upper().map(entity_to_cluster)
    )

    # --------------------------------------------------------
    # 6. Apply filters
    # --------------------------------------------------------

    all_candidates["passes_standard_filters"] = all_candidates.apply(
        passes_filters,
        axis=1,
    )

    # --------------------------------------------------------
    # 7. Save all candidates and the manual review queue
    # --------------------------------------------------------

    all_path = DATASET_DIR / "all_candidates.csv"
    all_candidates.to_csv(all_path, index=False)
    print(f"\nSaved screened candidates:\n{all_path}")

    review_queue = all_candidates[
        (all_candidates["screening_decision"] == "FLAG")
        | all_candidates["priority_target"]
    ]

    review_queue.to_csv(DATASET_DIR / "review_queue.csv", index=False)

    # Standard benchmark candidates must pass all filters. Priority targets
    # are always included, keeping passes_standard_filters=False when they
    # fall outside the general benchmark criteria.
    filtered = (
        all_candidates[
            all_candidates["passes_standard_filters"]
            | all_candidates["priority_target"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\nPassed standard filters: "
        f"{int(all_candidates['passes_standard_filters'].sum())}/"
        f"{len(all_candidates)}"
    )

    print(
        "Force-included priority targets: "
        + ", ".join(sorted(PRIORITY_TARGETS))
    )

    # --------------------------------------------------------
    # 8. Classify post-cutoff candidates as Test A or Test B
    #    by direct sequence search against pre-cutoff PDB
    # --------------------------------------------------------

    filtered = annotate_precutoff_similarity(filtered)

    # --------------------------------------------------------
    # 9. Create dataset splits
    # --------------------------------------------------------

    (
        filtered,
        development,
        validation,
        test_a,
        test_b,
    ) = create_splits(filtered)

    # Check that priority targets with a forced_split landed there.
    for pdb_id, spec in PRIORITY_TARGETS.items():
        split = spec.get("forced_split")
        placed = filtered.loc[
            filtered["pdb_id"] == pdb_id,
            "training_or_test_split",
        ].tolist()

        if split and placed != [split]:
            raise RuntimeError(
                f"{spec['protein_name']} ({pdb_id}) was placed in "
                f"{placed} instead of the required {split}."
            )

    # --------------------------------------------------------
    # 10. Save CSV files
    # --------------------------------------------------------

    filtered.to_csv(DATASET_DIR / "filtered_candidates.csv", index=False)

    splits = {
        "development": development,
        "validation": validation,
        "test_a": test_a,
        "test_b": test_b,
    }

    for name, split in splits.items():
        split.to_csv(DATASET_DIR / f"{name}.csv", index=False)

    # Random subset of automatically accepted, split-assigned proteins to
    # check by hand that the filters are behaving sensibly.
    selected = filtered[
        (filtered["training_or_test_split"] != "unassigned")
        & (filtered["priority_target"] != True)
    ]

    manual_sample = selected.sample(
        n=min(N_MANUAL_INSPECTION, len(selected)),
        random_state=RANDOM_SEED,
    ).copy()

    manual_sample["manual_verdict"] = ""
    manual_sample["manual_notes"] = ""

    manual_sample.to_csv(
        DATASET_DIR / "manual_inspection_sample.csv",
        index=False,
    )

    # Record the benchmark configuration and SHA-256 hashes of the split
    # files, so any later change to a locked test set is detectable.
    manifest = {
        # Provenance. Both the PDB and the RCSB cluster file change over
        # time, so the hashes below only certify a rebuild against the same
        # inputs if these are recorded too.
        "build_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sequence_cluster_url": CLUSTER_30_URL,
        "sequence_cluster_snapshot": cluster_snapshot,
        "cutoff_date": CUTOFF_DATE,
        "random_seed": RANDOM_SEED,
        "length_range": [MIN_LENGTH, MAX_LENGTH],
        "min_resolved_fraction": MIN_RESOLVED_FRACTION,
        "max_resolution_angstrom": MAX_RESOLUTION,
        "allowed_methods": sorted(ALLOWED_METHODS),
        "max_unobserved_segment": MAX_UNOBSERVED_SEGMENT,
        "max_disorder_annotation_fraction": MAX_DISORDER_ANNOTATION_FRACTION,
        "large_ligand": [LARGE_LIGAND_MW, LARGE_LIGAND_HEAVY_ATOMS],
        "medium_ligand": [MEDIUM_LIGAND_MW, MEDIUM_LIGAND_HEAVY_ATOMS],
        "allow_flagged_in_dev_val": ALLOW_FLAGGED_IN_DEV_VAL,
        "allow_flagged_in_test": ALLOW_FLAGGED_IN_TEST,
        "n_pre_candidates": N_PRE_CANDIDATES,
        "n_post_candidates": N_POST_CANDIDATES,
        "sequence_cluster_identity": 0.30,
        "test_b_max_sequence_identity": TESTB_MAX_PIDENT,
        "test_b_min_target_coverage": TESTB_MIN_COVERAGE,
        "priority_targets": PRIORITY_TARGETS,
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "sha256": {
            name: hashlib.sha256(
                (DATASET_DIR / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in splits
        },
    }

    with open(
        DATASET_DIR / "dataset_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)

    # --------------------------------------------------------
    # 11. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    n_flagged = int((all_candidates["screening_decision"] == "FLAG").sum())

    print(f"All candidates:       {len(all_candidates)}")
    print(f"Review queue (FLAG):  {n_flagged}")
    print(f"Filtered candidates:  {len(filtered)}")
    print(f"Development:          {len(development)}")
    print(f"Validation:           {len(validation)}")
    print(f"Test A:               {len(test_a)}")
    print(f"Test B:               {len(test_b)}")

    priority_summary = filtered[filtered["priority_target"] == True][
        [
            "pdb_id",
            "priority_name",
            "chain_id",
            "experimental_method",
            "forced_split",
            "screening_decision",
            "precutoff_search_status",
            "best_precutoff_pident",
            "best_precutoff_coverage",
            "training_or_test_split",
        ]
    ]

    print("\nPriority targets:")
    print(priority_summary.to_string(index=False))

    print(f"\nFiles saved in:\n{DATASET_DIR}")

    targets = {
        "Development set": (development, N_DEVELOPMENT),
        "Validation set": (validation, N_VALIDATION),
        "Test A": (test_a, N_TEST_A),
        "Test B": (test_b, N_TEST_B),
    }

    for label, (split, requested) in targets.items():
        if len(split) < requested:
            print(f"WARNING: {label} is smaller than requested.")


if __name__ == "__main__":
    main()
