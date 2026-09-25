from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PROTEIN FILTER SETTINGS
# ============================================================

CUTOFF_DATE = "2021-09-30"

# No. of residues
MIN_LENGTH = 80
MAX_LENGTH = 400

# Resolved fraction >= 90% and Resolution <= 3.5 Å
MIN_RESOLVED_FRACTION = 0.90
MAX_RESOLUTION = 3.5

ALLOWED_METHODS = {
    "X-RAY DIFFRACTION",
    "ELECTRON MICROSCOPY",
}

# Number of raw candidates to inspect from each side of the cutoff.
N_PRE_CANDIDATES = 300
N_POST_CANDIDATES = 700

# Final dataset sizes
N_DEVELOPMENT = 30
N_VALIDATION = 15
N_TEST_A = 20
N_TEST_B = 20

RANDOM_SEED = 42

# Number of search hits from which the raw candidate sample is drawn.
MAX_SEARCH_HITS = 5000

# Do not hammer the RCSB servers with too many simultaneous requests.
MAX_WORKERS = 6

# These are force-included even when they do not satisfy the standard
# automated benchmark filters.
#
# 8EXF: BCCIPalpha is chain B in a FAM46A-BCCIPalpha complex.
# 8URV: pro-IL-18 is chain A and was determined by solution NMR.
PRIORITY_TARGETS = {
    "8EXF": {
        "chain_id": "B",
        "protein_name": "BCCIPalpha",
    },
    "8URV": {
        "chain_id": "A",
        "protein_name": "pro-IL-18",
    },
}


# ============================================================
# URLS
# ============================================================

SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API = "https://data.rcsb.org/rest/v1/core"
GRAPHQL_API = "https://data.rcsb.org/graphql"

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
# BASIC HTTP FUNCTIONS
# ============================================================

def get_json(url: str, retries: int = 3):
    """GET JSON with simple retry handling."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise

            wait = 2 ** attempt
            print(f"Request failed: {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)


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


# ============================================================
# RCSB SEARCH
# ============================================================

def terminal(attribute: str, operator: str, value):
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
    max_results: int = MAX_SEARCH_HITS,
):
    """
    Search RCSB for entries that:
      - contain one protein entity
      - contain one deposited polymer chain
      - have sequence length 80-400
      - are before OR after the OpenFold3 cutoff
    """

    date_operator = "less_or_equal" if pre_cutoff else "greater"

    nodes = [
        terminal(
            "rcsb_entry_info.polymer_entity_count_protein",
            "equals",
            1,
        ),
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

    while len(all_ids) < max_results:

        rows = min(page_size, max_results - len(all_ids))

        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": start,
                    "rows": rows,
                },
                "results_content_type": ["experimental"],
            },
        }

        result = post_json(SEARCH_API, payload)

        hits = result.get("result_set", [])

        if not hits:
            break

        ids = [hit["identifier"].upper() for hit in hits]

        all_ids.extend(ids)
        start += len(ids)

        if len(ids) < rows:
            break

    return list(dict.fromkeys(all_ids))


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


def is_membrane_entity(polymer_json: dict) -> bool:
    """
    Look for membrane-protein annotations supplied by RCSB.

    This is deliberately conservative and should still be checked manually.
    """

    annotations = polymer_json.get(
        "rcsb_polymer_entity_annotation",
        [],
    ) or []

    membrane_terms = (
        "PDBTM",
        "OPM",
        "MEMPROTMD",
        "MPSTRUC",
        "MEMBRANE",
    )

    for annotation in annotations:
        text = " ".join(
            str(value)
            for value in annotation.values()
            if value is not None
        ).upper()

        if any(term in text for term in membrane_terms):
            return True

    return False


def find_monomeric_assembly(
    pdb_id: str,
    assembly_ids: list[str],
):
    """
    Find a biological assembly containing exactly one protein chain
    and no DNA/RNA chains.
    """

    for assembly_id in assembly_ids:

        try:
            data = get_json(
                f"{DATA_API}/assembly/{pdb_id}/{assembly_id}"
            )
        except Exception:
            continue

        info = data.get("rcsb_assembly_info", {}) or {}

        protein_count = info.get(
            "polymer_entity_instance_count_protein",
            0,
        ) or 0

        dna_count = info.get(
            "polymer_entity_instance_count_DNA",
            0,
        ) or 0

        rna_count = info.get(
            "polymer_entity_instance_count_RNA",
            0,
        ) or 0

        hybrid_count = info.get(
            "polymer_entity_instance_count_nucleic_acid_hybrid",
            0,
        ) or 0

        if (
            protein_count == 1
            and dna_count == 0
            and rna_count == 0
            and hybrid_count == 0
        ):
            return assembly_id, data

    return None, None


def extract_metadata(
    pdb_id: str,
    target_chain_id: str | None = None,
    priority_target: bool = False,
    priority_name: str = "",
):
    """
    Download metadata for one PDB entry.

    Standard candidates are required to contain exactly one protein
    polymer entity, preserving the original benchmark logic.

    Priority targets may specify a chain ID. In that case, the matching
    polymer entity is selected even when the PDB entry contains multiple
    protein entities (for example, 8EXF).

    Returns one dictionary representing one candidate protein.
    """

    try:
        entry = get_json(f"{DATA_API}/entry/{pdb_id}")

        entry_ids = entry.get(
            "rcsb_entry_container_identifiers",
            {},
        ) or {}

        polymer_entity_ids = (
            entry_ids.get("polymer_entity_ids", []) or []
        )

        # ----------------------------------------------------
        # Select the protein entity
        # ----------------------------------------------------

        polymer = None
        entity_number = None

        if target_chain_id is None:

            # Preserve the original clean-benchmark requirement.
            if len(polymer_entity_ids) != 1:
                return None

            entity_number = str(polymer_entity_ids[0])

            polymer = get_json(
                f"{DATA_API}/polymer_entity/"
                f"{pdb_id}/{entity_number}"
            )

        else:

            # Priority entries can contain multiple protein entities.
            # Find the entity containing the requested author or label chain.
            target_chain_id = str(target_chain_id)

            for candidate_entity_id in polymer_entity_ids:

                candidate_entity_id = str(candidate_entity_id)

                candidate_polymer = get_json(
                    f"{DATA_API}/polymer_entity/"
                    f"{pdb_id}/{candidate_entity_id}"
                )

                candidate_container = candidate_polymer.get(
                    "rcsb_polymer_entity_container_identifiers",
                    {},
                ) or {}

                candidate_auth_chains = [
                    str(x)
                    for x in (
                        candidate_container.get("auth_asym_ids", [])
                        or []
                    )
                ]

                candidate_label_chains = [
                    str(x)
                    for x in (
                        candidate_container.get("asym_ids", [])
                        or []
                    )
                ]

                if (
                    target_chain_id in candidate_auth_chains
                    or target_chain_id in candidate_label_chains
                ):
                    entity_number = candidate_entity_id
                    polymer = candidate_polymer
                    break

            if polymer is None:
                raise RuntimeError(
                    f"Could not find target chain {target_chain_id} "
                    f"in {pdb_id}."
                )

        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        entity_poly = polymer.get("entity_poly", {}) or {}

        sequence = clean_sequence(
            entity_poly.get(
                "pdbx_seq_one_letter_code_can",
                "",
            )
        )

        length = entity_poly.get(
            "rcsb_sample_sequence_length"
        )

        if length is None:
            length = len(sequence)

        # ----------------------------------------------------
        # Chain IDs
        # ----------------------------------------------------

        container = polymer.get(
            "rcsb_polymer_entity_container_identifiers",
            {},
        ) or {}

        author_chains = [
            str(x)
            for x in (
                container.get("auth_asym_ids", [])
                or []
            )
        ]

        label_chains = [
            str(x)
            for x in (
                container.get("asym_ids", [])
                or []
            )
        ]

        if (
            target_chain_id is not None
            and target_chain_id in author_chains
        ):
            chain_id = target_chain_id
        else:
            chain_id = (
                author_chains[0]
                if author_chains
                else ""
            )

        if (
            target_chain_id is not None
            and target_chain_id in label_chains
        ):
            label_chain_id = target_chain_id
        else:
            label_chain_id = (
                label_chains[0]
                if label_chains
                else ""
            )

        # ----------------------------------------------------
        # Release date
        # ----------------------------------------------------

        accession = entry.get(
            "rcsb_accession_info",
            {},
        ) or {}

        release_date = accession.get(
            "initial_release_date",
            "",
        )

        if release_date:
            release_date = release_date[:10]

        # ----------------------------------------------------
        # Experimental method
        # ----------------------------------------------------

        methods = []

        for experiment in entry.get("exptl", []) or []:
            method = experiment.get("method")

            if method:
                methods.append(method.upper())

        experimental_method = "; ".join(methods)

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        entry_info = entry.get(
            "rcsb_entry_info",
            {},
        ) or {}

        resolutions = (
            entry_info.get("resolution_combined", [])
            or []
        )

        resolution = None

        if resolutions:
            numeric_resolutions = [
                float(x)
                for x in resolutions
                if x is not None
            ]

            if numeric_resolutions:
                resolution = min(numeric_resolutions)

        # ----------------------------------------------------
        # Organism
        # ----------------------------------------------------

        organisms = []

        for organism in (
            polymer.get(
                "rcsb_entity_source_organism",
                [],
            )
            or []
        ):
            name = organism.get(
                "ncbi_scientific_name"
            )

            if name:
                organisms.append(name)

        organism = "; ".join(
            sorted(set(organisms))
        )

        # ----------------------------------------------------
        # Biological assembly
        # ----------------------------------------------------

        assembly_ids = (
            entry_ids.get("assembly_ids", [])
            or []
        )

        monomer_assembly_id, monomer_assembly = (
            find_monomeric_assembly(
                pdb_id,
                assembly_ids,
            )
        )

        is_monomer = monomer_assembly is not None

        assembly_id = monomer_assembly_id
        assembly = monomer_assembly

        # A priority target such as 8EXF may intentionally be part of
        # a complex. Keep it and record the first biological assembly
        # rather than dropping it for not being monomeric.
        if (
            assembly is None
            and priority_target
            and assembly_ids
        ):
            assembly_id = str(assembly_ids[0])

            try:
                assembly = get_json(
                    f"{DATA_API}/assembly/"
                    f"{pdb_id}/{assembly_id}"
                )
            except Exception:
                assembly = None

        resolved_fraction = None
        number_of_chains = None
        assembly_ligand_count = None

        if assembly:

            assembly_info = assembly.get(
                "rcsb_assembly_info",
                {},
            ) or {}

            total_residues = assembly_info.get(
                "polymer_monomer_count",
                0,
            ) or 0

            modeled_residues = assembly_info.get(
                "modeled_polymer_monomer_count",
                0,
            ) or 0

            if total_residues > 0:
                resolved_fraction = (
                    modeled_residues / total_residues
                )

            number_of_chains = assembly_info.get(
                "polymer_entity_instance_count"
            )

            assembly_ligand_count = (
                assembly_info.get(
                    "nonpolymer_entity_instance_count",
                    0,
                )
                or 0
            )

        # ----------------------------------------------------
        # Ligands
        # ----------------------------------------------------

        nonpolymer_ids = (
            entry_ids.get(
                "non_polymer_entity_ids",
                [],
            )
            or []
        )

        ligand_count = len(nonpolymer_ids)

        # ----------------------------------------------------
        # Membrane annotation
        # ----------------------------------------------------

        membrane = is_membrane_entity(polymer)

        # ----------------------------------------------------
        # Polymer entity ID used by RCSB clustering
        # ----------------------------------------------------

        polymer_entity_id = (
            f"{pdb_id}_{entity_number}"
        )

        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "label_chain_id": label_chain_id,
            "polymer_entity_id": polymer_entity_id,
            "release_date": release_date,
            "sequence": sequence,
            "length": int(length),
            "experimental_method": experimental_method,
            "resolution": resolution,
            "resolved_fraction": resolved_fraction,
            "organism": organism,
            "number_of_chains": number_of_chains,
            "ligands": ligand_count,
            "assembly_ligand_count": assembly_ligand_count,
            "biological_assembly": assembly_id,
            "is_monomer": is_monomer,
            "is_membrane": membrane,
            "standard_sequence": is_standard_protein_sequence(
                sequence
            ),
            "sequence_cluster": None,
            "has_pre_cutoff_cluster_member": None,
            "training_or_test_split": "unassigned",
            "priority_target": bool(priority_target),
            "priority_name": priority_name,
        }

    except Exception as exc:
        print(f"\nFailed to process {pdb_id}: {exc}")
        return None


# ============================================================
# SEQUENCE CLUSTERS
# ============================================================

def download_sequence_clusters():
    """
    Download RCSB's current 30%-sequence-identity clusters.

    Returns:
        entity_to_cluster
        cluster_to_entities
    """

    print("\nDownloading RCSB 30% sequence clusters...")

    response = requests.get(
        CLUSTER_30_URL,
        timeout=120,
    )
    response.raise_for_status()

    entity_to_cluster = {}
    cluster_to_entities = {}

    for cluster_number, line in enumerate(
        response.text.splitlines(),
        start=1,
    ):
        entities = [
            x.upper()
            for x in line.split()
            if x.strip()
        ]

        if not entities:
            continue

        cluster_id = f"seq30_{cluster_number:06d}"

        cluster_to_entities[cluster_id] = entities

        for entity in entities:
            entity_to_cluster[entity] = cluster_id

    print(
        f"Loaded {len(cluster_to_entities):,} "
        f"30% sequence clusters."
    )

    return entity_to_cluster, cluster_to_entities


# ============================================================
# RELEASE DATES FOR CLUSTER MEMBERS
# ============================================================

def fetch_release_dates_batch(pdb_ids):
    """
    Retrieve initial PDB release dates in batches through
    the RCSB GraphQL Data API.
    """

    pdb_ids = sorted(set(x.upper() for x in pdb_ids))

    query = """
    query GetEntries($ids: [String!]!) {
        entries(entry_ids: $ids) {
            rcsb_id
            rcsb_accession_info {
                initial_release_date
            }
        }
    }
    """

    result = {}

    batch_size = 300

    for start in range(0, len(pdb_ids), batch_size):

        batch = pdb_ids[
            start:start + batch_size
        ]

        payload = {
            "query": query,
            "variables": {
                "ids": batch,
            },
        }

        data = post_json(
            GRAPHQL_API,
            payload,
        )

        entries = (
            data.get("data", {})
            .get("entries", [])
            or []
        )

        for entry in entries:

            if not entry:
                continue

            pdb_id = entry.get("rcsb_id")

            accession = (
                entry.get(
                    "rcsb_accession_info",
                    {}
                )
                or {}
            )

            date = accession.get(
                "initial_release_date"
            )

            if pdb_id and date:
                result[pdb_id.upper()] = date[:10]

    return result


def annotate_pre_cutoff_cluster_members(
    df,
    cluster_to_entities,
):
    """
    Determine whether each candidate's 30% cluster contains
    ANY PDB structure released on/before 30 Sep 2021.

    This is used to identify difficult Test B candidates.
    """

    post_df = df[
        df["release_date"] > CUTOFF_DATE
    ]

    post_clusters = (
        post_df["sequence_cluster"]
        .dropna()
        .unique()
        .tolist()
    )

    relevant_pdb_ids = set()

    for cluster_id in post_clusters:

        members = cluster_to_entities.get(
            cluster_id,
            [],
        )

        for member in members:
            pdb_id = member.split("_")[0]
            relevant_pdb_ids.add(pdb_id)

    print(
        "\nChecking pre-cutoff homologs for "
        f"{len(post_clusters)} post-cutoff clusters..."
    )

    print(
        f"Need release dates for "
        f"{len(relevant_pdb_ids):,} PDB entries."
    )

    release_dates = fetch_release_dates_batch(
        relevant_pdb_ids
    )

    cluster_has_pre_cutoff = {}

    for cluster_id in post_clusters:

        members = cluster_to_entities.get(
            cluster_id,
            [],
        )

        has_old_member = False

        for member in members:

            pdb_id = member.split("_")[0]

            release_date = release_dates.get(
                pdb_id
            )

            if (
                release_date
                and release_date <= CUTOFF_DATE
            ):
                has_old_member = True
                break

        cluster_has_pre_cutoff[
            cluster_id
        ] = has_old_member

    df["has_pre_cutoff_cluster_member"] = (
        df["sequence_cluster"]
        .map(cluster_has_pre_cutoff)
    )

    # Pre-cutoff proteins obviously belong to the
    # pre-cutoff period themselves.
    pre_mask = (
        df["release_date"] <= CUTOFF_DATE
    )

    df.loc[
        pre_mask,
        "has_pre_cutoff_cluster_member",
    ] = True

    return df


# ============================================================
# FILTERING
# ============================================================

def passes_filters(row):
    """
    Apply the clean initial biological restrictions.
    """

    if not (
        MIN_LENGTH
        <= row["length"]
        <= MAX_LENGTH
    ):
        return False

    if not row["is_monomer"]:
        return False

    if row["is_membrane"]:
        return False

    if not row["standard_sequence"]:
        return False

    if pd.isna(row["resolved_fraction"]):
        return False

    if (
        row["resolved_fraction"]
        < MIN_RESOLVED_FRACTION
    ):
        return False

    if row["experimental_method"] not in ALLOWED_METHODS:
        return False

    if pd.isna(row["resolution"]):
        return False

    if row["resolution"] > MAX_RESOLUTION:
        return False

    if pd.isna(row["sequence_cluster"]):
        return False

    return True


# ============================================================
# SPLIT CREATION
# ============================================================

def take_cluster_representatives(
    df,
    n,
    rng,
    excluded_clusters=None,
):
    """
    Select at most one structure from each 30% sequence cluster.
    """

    if excluded_clusters is None:
        excluded_clusters = set()

    working = df[
        ~df["sequence_cluster"].isin(
            excluded_clusters
        )
    ].copy()

    # Shuffle first so we do not always keep the same PDB
    # when a cluster contains multiple candidates.
    random_state = rng.randint(
        0,
        2**32 - 1,
    )

    working = working.sample(
        frac=1,
        random_state=random_state,
    )

    working = working.drop_duplicates(
        subset=["sequence_cluster"],
        keep="first",
    )

    random_state = rng.randint(
        0,
        2**32 - 1,
    )

    working = working.sample(
        frac=1,
        random_state=random_state,
    )

    return working.head(n).copy()


def create_splits(df):
    """
    Create:
        development
        validation
        test_a
        test_b

    with no 30%-cluster overlap between the selected sets.

    Supervisor-selected priority targets are guaranteed inclusion.
    Post-cutoff priority targets are assigned as:
      - Test B if their 30% cluster has no pre-cutoff PDB member
      - Test A otherwise

    Their sequence clusters are reserved before development/validation
    sampling so the selected splits remain cluster-disjoint.
    """

    rng = random.Random(RANDOM_SEED)

    pre = df[
        df["release_date"] <= CUTOFF_DATE
    ].copy()

    post = df[
        df["release_date"] > CUTOFF_DATE
    ].copy()

    priority = df[
        df["priority_target"] == True
    ].copy()

    priority_clusters = set(
        priority["sequence_cluster"]
        .dropna()
        .tolist()
    )

    # --------------------------------------------------------
    # DEVELOPMENT
    # --------------------------------------------------------

    # Reserve all priority clusters so no pre-cutoff development
    # protein overlaps at 30% identity with a required test target.
    development = take_cluster_representatives(
        pre[
            pre["priority_target"] != True
        ],
        N_DEVELOPMENT,
        rng,
        excluded_clusters=priority_clusters,
    )

    used_clusters = (
        set(development["sequence_cluster"])
        | priority_clusters
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation = take_cluster_representatives(
        pre[
            pre["priority_target"] != True
        ],
        N_VALIDATION,
        rng,
        excluded_clusters=used_clusters,
    )

    used_clusters.update(
        validation["sequence_cluster"]
    )

    # --------------------------------------------------------
    # REQUIRED POST-CUTOFF PRIORITY TARGETS
    # --------------------------------------------------------

    priority_post = post[
        post["priority_target"] == True
    ].copy()

    priority_test_b = priority_post[
        priority_post[
            "has_pre_cutoff_cluster_member"
        ]
        == False
    ].copy()

    priority_test_a = priority_post[
        ~priority_post.index.isin(
            priority_test_b.index
        )
    ].copy()

    # --------------------------------------------------------
    # TEST B
    #
    # Post-cutoff AND no member of its 30% sequence cluster
    # existed before the cutoff.
    # --------------------------------------------------------

    difficult_pool = post[
        (
            post[
                "has_pre_cutoff_cluster_member"
            ]
            == False
        )
        & (
            post["priority_target"] != True
        )
    ].copy()

    n_test_b_remaining = max(
        N_TEST_B - len(priority_test_b),
        0,
    )

    sampled_test_b = take_cluster_representatives(
        difficult_pool,
        n_test_b_remaining,
        rng,
        excluded_clusters=used_clusters,
    )

    test_b = pd.concat(
        [
            priority_test_b,
            sampled_test_b,
        ],
        axis=0,
    )

    used_clusters.update(
        test_b["sequence_cluster"]
        .dropna()
        .tolist()
    )

    # --------------------------------------------------------
    # TEST A
    #
    # Post-cutoff temporal proteins.
    # Keep Test A separate from Test B.
    # --------------------------------------------------------

    temporal_pool = post[
        (
            ~post.index.isin(test_b.index)
        )
        & (
            post["priority_target"] != True
        )
    ].copy()

    n_test_a_remaining = max(
        N_TEST_A - len(priority_test_a),
        0,
    )

    sampled_test_a = take_cluster_representatives(
        temporal_pool,
        n_test_a_remaining,
        rng,
        excluded_clusters=used_clusters,
    )

    test_a = pd.concat(
        [
            priority_test_a,
            sampled_test_a,
        ],
        axis=0,
    )

    # --------------------------------------------------------
    # Assign split names
    # --------------------------------------------------------

    df = df.copy()

    df["training_or_test_split"] = (
        "unassigned"
    )

    df.loc[
        development.index,
        "training_or_test_split",
    ] = "development"

    df.loc[
        validation.index,
        "training_or_test_split",
    ] = "validation"

    df.loc[
        test_a.index,
        "training_or_test_split",
    ] = "test_a"

    df.loc[
        test_b.index,
        "training_or_test_split",
    ] = "test_b"

    development = df.loc[
        development.index
    ].copy()

    validation = df.loc[
        validation.index
    ].copy()

    test_a = df.loc[
        test_a.index
    ].copy()

    test_b = df.loc[
        test_b.index
    ].copy()

    return (
        df,
        development,
        validation,
        test_a,
        test_b,
    )


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

    pre_ids = search_candidate_pdb_ids(
        pre_cutoff=True
    )

    print(
        f"Found {len(pre_ids):,} searchable "
        f"pre-cutoff entries."
    )

    print("\nSearching post-cutoff PDB entries...")

    post_ids = search_candidate_pdb_ids(
        pre_cutoff=False
    )

    print(
        f"Found {len(post_ids):,} searchable "
        f"post-cutoff entries."
    )

    # Sort IDs before seeded sampling so the same search results
    # produce the same sampled candidates across runs.
    pre_ids = sorted(pre_ids)
    post_ids = sorted(post_ids)

    if len(pre_ids) > N_PRE_CANDIDATES:
        pre_ids = rng.sample(
            pre_ids,
            N_PRE_CANDIDATES,
        )

    if len(post_ids) > N_POST_CANDIDATES:
        post_ids = rng.sample(
            post_ids,
            N_POST_CANDIDATES,
        )

    # Remove priority targets from the random pool if they happened
    # to be sampled already. They will be added explicitly below.
    priority_ids = set(PRIORITY_TARGETS)

    pdb_ids = [
        pdb_id
        for pdb_id in (pre_ids + post_ids)
        if pdb_id not in priority_ids
    ]

    metadata_jobs = [
        {
            "pdb_id": pdb_id,
            "target_chain_id": None,
            "priority_target": False,
            "priority_name": "",
        }
        for pdb_id in pdb_ids
    ]

    for pdb_id, spec in PRIORITY_TARGETS.items():
        metadata_jobs.append(
            {
                "pdb_id": pdb_id,
                "target_chain_id": spec["chain_id"],
                "priority_target": True,
                "priority_name": spec["protein_name"],
            }
        )

    print(
        f"\nRetrieving metadata for "
        f"{len(metadata_jobs)} candidates "
        f"(including {len(PRIORITY_TARGETS)} "
        f"required priority targets)..."
    )

    # --------------------------------------------------------
    # 2. Download metadata
    # --------------------------------------------------------

    rows = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                extract_metadata,
                job["pdb_id"],
                job["target_chain_id"],
                job["priority_target"],
                job["priority_name"],
            ): job["pdb_id"]
            for job in metadata_jobs
        }

        completed = 0

        for future in as_completed(futures):

            result = future.result()

            if result is not None:
                rows.append(result)

            completed += 1

            if (
                completed % 25 == 0
                or completed == len(futures)
            ):
                print(
                    f"Processed "
                    f"{completed}/"
                    f"{len(futures)}"
                )

    # Threaded downloads finish in a nondeterministic order.
    # Sort by PDB ID so later seeded sampling is reproducible.
    all_candidates = (
        pd.DataFrame(rows)
        .sort_values("pdb_id")
        .reset_index(drop=True)
    )

    retrieved_priority_ids = set(
        all_candidates.loc[
            all_candidates["priority_target"] == True,
            "pdb_id",
        ].tolist()
    )

    missing_priority_ids = (
        set(PRIORITY_TARGETS)
        - retrieved_priority_ids
    )

    if missing_priority_ids:
        raise RuntimeError(
            "Failed to retrieve required priority target(s): "
            + ", ".join(sorted(missing_priority_ids))
        )

    if all_candidates.empty:
        raise RuntimeError(
            "No candidate structures were retrieved."
        )

    # --------------------------------------------------------
    # 3. Add 30% sequence clusters
    # --------------------------------------------------------

    (
        entity_to_cluster,
        cluster_to_entities,
    ) = download_sequence_clusters()

    all_candidates[
        "sequence_cluster"
    ] = (
        all_candidates[
            "polymer_entity_id"
        ]
        .str.upper()
        .map(entity_to_cluster)
    )

    # --------------------------------------------------------
    # 4. Save all candidates
    # --------------------------------------------------------

    all_path = (
        DATASET_DIR
        / "all_candidates.csv"
    )

    all_candidates.to_csv(
        all_path,
        index=False,
    )

    print(
        f"\nSaved raw candidates:\n{all_path}"
    )

    # --------------------------------------------------------
    # 5. Apply filters
    # --------------------------------------------------------

    all_candidates[
        "passes_standard_filters"
    ] = all_candidates.apply(
        passes_filters,
        axis=1,
    )

    # Standard benchmark candidates must pass all filters.
    # Supervisor-selected targets are deliberate exceptions and are
    # force-included while retaining passes_standard_filters=False
    # when they fall outside the general benchmark criteria.
    filter_mask = (
        all_candidates[
            "passes_standard_filters"
        ]
        | all_candidates[
            "priority_target"
        ]
    )

    filtered = (
        all_candidates[
            filter_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    standard_pass_count = int(
        all_candidates[
            "passes_standard_filters"
        ].sum()
    )

    print(
        f"\nPassed standard filters: "
        f"{standard_pass_count}/"
        f"{len(all_candidates)}"
    )

    print(
        "Force-included priority targets: "
        + ", ".join(
            sorted(PRIORITY_TARGETS)
        )
    )

    # --------------------------------------------------------
    # 6. Determine whether post-cutoff clusters existed
    #    before OpenFold3's cutoff
    # --------------------------------------------------------

    filtered = (
        annotate_pre_cutoff_cluster_members(
            filtered,
            cluster_to_entities,
        )
    )

    # --------------------------------------------------------
    # 7. Create dataset splits
    # --------------------------------------------------------

    (
        filtered,
        development,
        validation,
        test_a,
        test_b,
    ) = create_splits(filtered)

    # --------------------------------------------------------
    # 8. Save CSV files
    # --------------------------------------------------------

    filtered.to_csv(
        DATASET_DIR
        / "filtered_candidates.csv",
        index=False,
    )

    development.to_csv(
        DATASET_DIR
        / "development.csv",
        index=False,
    )

    validation.to_csv(
        DATASET_DIR
        / "validation.csv",
        index=False,
    )

    test_a.to_csv(
        DATASET_DIR
        / "test_a.csv",
        index=False,
    )

    test_b.to_csv(
        DATASET_DIR
        / "test_b.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    print(
        f"All candidates:       "
        f"{len(all_candidates)}"
    )

    print(
        f"Filtered candidates:  "
        f"{len(filtered)}"
    )

    print(
        f"Development:          "
        f"{len(development)}"
    )

    print(
        f"Validation:           "
        f"{len(validation)}"
    )

    print(
        f"Test A:               "
        f"{len(test_a)}"
    )

    print(
        f"Test B:               "
        f"{len(test_b)}"
    )

    priority_summary = filtered[
        filtered["priority_target"] == True
    ][
        [
            "pdb_id",
            "priority_name",
            "chain_id",
            "experimental_method",
            "passes_standard_filters",
            "has_pre_cutoff_cluster_member",
            "training_or_test_split",
        ]
    ]

    print("\nPriority targets:")
    print(
        priority_summary.to_string(
            index=False
        )
    )

    print(
        "\nFiles saved in:"
        f"\n{DATASET_DIR}"
    )

    if len(development) < N_DEVELOPMENT:
        print(
            "\nWARNING: Development set is "
            "smaller than requested."
        )

    if len(validation) < N_VALIDATION:
        print(
            "WARNING: Validation set is "
            "smaller than requested."
        )

    if len(test_a) < N_TEST_A:
        print(
            "WARNING: Test A is smaller "
            "than requested."
        )

    if len(test_b) < N_TEST_B:
        print(
            "WARNING: Test B is smaller "
            "than requested."
        )


if __name__ == "__main__":
    main()