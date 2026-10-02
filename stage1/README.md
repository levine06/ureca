# Stage 1 — Benchmark Dataset Construction

This stage constructs the benchmark datasets used to develop and evaluate OpenFold3-based methods for experiment-guided protein structure prediction.

OpenFold3's structural training cutoff is:

**30 September 2021**

The goal is to create reproducible development, validation, temporal-test, and difficult-generalization datasets while reducing leakage from closely related structures.

---

## Files

```text
stage1/
├── README.md
├── build_dataset.py
└── datasets/
    ├── all_candidates.csv
    ├── filtered_candidates.csv
    ├── review_queue.csv
    ├── manual_inspection_sample.csv
    ├── dataset_manifest.json
    ├── development.csv
    ├── validation.csv
    ├── test_a.csv
    └── test_b.csv
```

## `build_dataset.py`

The main script is used to:

1. Search RCSB PDB for all pre- and post-cutoff single-chain structures, then draw a seeded random sample from each side.
2. Retrieve structural and sequence metadata in batches from the RCSB GraphQL API.
3. Retrieve UniProt annotations and check the OPM membrane-protein database.
4. Screen every candidate against the Stage 1 biological restrictions.
5. Assign RCSB 30% sequence-identity clusters, used to keep the splits non-redundant.
6. Search every post-cutoff candidate's sequence directly against pre-cutoff PDB structures to decide between Test A and Test B.
7. Construct cluster-disjoint development, validation, Test A, and Test B splits.
8. Force-include two priority targets.
9. Save the generated datasets, a manual review queue, and a random manual-inspection sample as CSV files.

---

## Initial Benchmark Criteria

Every candidate is screened against the Stage 1 biological restrictions. Each criterion is recorded as `PASS`, `FLAG` (needs manual review), or `FAIL` in its own column:

| Column | FAIL when | FLAG when |
|---|---|---|
| `monomeric` | the preferred biological assembly (assembly 1) has more than one protein chain | assembly composition is unavailable, or another assembly is oligomeric |
| `not_obligate_complex` | the assembly is not monomeric or contains DNA/RNA | the UniProt subunit annotation describes a homo-/hetero-oligomer or complex component |
| `soluble` | a UniProt lipidation or GPI-anchor site lies inside the construct | a lipidation/GPI site lies outside the construct or its position cannot be checked (e.g. keyword-only evidence), UniProt subcellular location includes a membrane or the cell surface, or the UniProt entry could not be retrieved |
| `not_membrane` | the entry is in OPM, is annotated by PDBTM/OPM/MemProtMD/mpstruc in RCSB, or a UniProt transmembrane segment lies inside the construct (or its position cannot be checked against the construct) | the construct is a soluble domain whose transmembrane segment lies outside it, or OPM could not be reached |
| `length_80_400` | the sequence is outside 80–400 residues | — |
| `sequence_length_consistent` | — | the sequence string length differs from the RCSB sequence length |
| `no_terminal_tag` | — | a common expression tag (poly-His, Strep-tag II, FLAG, HA, c-Myc, V5, AviTag, S-tag, T7) lies within 30 residues of either terminus |
| `resolved_ge_0_90` | less than 90% of the chain is modelled (from the chain's RCSB unobserved-residue annotation), or the resolved fraction cannot be determined | — |
| `structure_quality` | any listed experimental method is not X-ray/EM (multi-method entries are checked method by method), resolution is worse than 3.5 Å or unavailable, or the sequence contains non-standard residues | — |
| `not_heavily_disordered` | the chain is less than 90% resolved | a single unmodelled segment is longer than 20 residues, or UniProt disordered regions cover more than 20% of the construct |
| `no_large_ligand` | a non-trivial ligand is ≥ 500 Da or has > 25 heavy atoms (e.g. heme, FAD, NAD) | a non-trivial ligand is ≥ 200 Da or has > 12 heavy atoms |

Water, ions, buffers, cryoprotectants, common crystallisation additives, and common modified amino acids (e.g. selenomethionine) are ignored by the ligand check. Because metal ions are ignored, a protein whose fold depends on a bound metal is not caught by the screen.

The ligand check also covers oligosaccharides: bound sugars and covalently attached glycans. RCSB stores these separately from small-molecule ligands. They are judged on weight alone (no formula is available) and appear in the `ligands` column as, for example, `oligosaccharide(NAG)`. A glycosylated protein whose glycan is ≥ 500 Da therefore fails like any other large-ligand case.

The overall `screening_decision` is `EXCLUDE` if any criterion fails, `FLAG` if any criterion is flagged, and `INCLUDE` otherwise. Notes explaining each failure and flag are stored in `screening_notes` and `flag_reasons`.

Candidates eligible for the splits (`passes_standard_filters`) must have an `INCLUDE` decision and an RCSB 30% sequence cluster. `FLAG` candidates need manual review, so by default they are kept out of every split. This is controlled separately for the pre-cutoff development/validation pool and the post-cutoff test pool by `ALLOW_FLAGGED_IN_DEV_VAL` and `ALLOW_FLAGGED_IN_TEST` (both currently `False`).

The two priority proteins are special proteins selected for investigation. They are force-included even when they do not satisfy all of the standard automated filters. Their automated verdicts are kept and the override is recorded in `flag_reasons`.

---

## Dataset Splits

### Development Set

File:

```text
datasets/development.csv
```

Current size: **30 proteins**

The development set contains structures released on or before 30 September 2021.

It is intended for method development, debugging, and implementation work.

---

### Validation Set

File:

```text
datasets/validation.csv
```

Current size: **15 proteins**

The validation set contains structures released on or before 30 September 2021 and is kept separate from the development set.

It can later be used for selecting parameters such as:

- thresholds;
- accessibility scoring parameters;
- guidance strengths; and
- other hyperparameters.

---

### Test A — Temporal Test

File:

```text
datasets/test_a.csv
```

Current size: **20 proteins**

Test A contains proteins whose experimental structures were released after 30 September 2021 and that have a close pre-cutoff relative: at least one PDB structure released on or before the cutoff aligns to them with ≥ 30% sequence identity over ≥ 80% of their sequence (see [Test A / Test B Rule](#test-a--test-b-rule)).

It tests performance on structures that were not directly present within the OpenFold3 structural-training period.

Test A and Test B are mutually exclusive.

#### Priority Target

- **pro-IL-18**
- PDB: `8URV`
- Chain: `A`
- Split: Test A

pro-IL-18 is not pinned to a split; it goes through the same Test A / Test B rule as every other post-cutoff protein. It is placed in Test A because the pre-cutoff mature IL-18 structure `1J0S` aligns to it with 100% identity over 81% of its sequence.

---

### Test B — Difficult-Generalization Test

File:

```text
datasets/test_b.csv
```

Current size: **20 proteins**

For normal candidates, Test B contains post-cutoff proteins for which no PDB structure released on or before 30 September 2021 reaches ≥ 30% sequence identity over ≥ 80% of the protein's sequence (see [Test A / Test B Rule](#test-a--test-b-rule)).

This is intended to provide a more difficult generalization test where the model cannot simply rely on a closely related pre-cutoff structural example.

Test A and Test B are mutually exclusive.

#### Priority Target

- **BCCIPα**
- PDB: `8EXF`
- Chain: `B`
- Split: Test B

BCCIPα is treated as a special edge case.

Although BCCIPα has strong sequence similarity to the pre-cutoff BCCIPβ structure, its experimentally determined 3D fold is substantially different.

It is also one of the more interesting cases where the AlphaFold3 prediction differs substantially from the experimentally determined structure.

For this reason, BCCIPα is pinned to Test B through `forced_split` in `PRIORITY_TARGETS`, regardless of how the sequence-similarity rule classifies it. The script raises an error if it is not placed there.

(Under the current rule BCCIPα would also fall into Test B on its own: its closest pre-cutoff match, BCCIPβ `7KYQ`, has 92% identity but covers only 75% of the BCCIPα sequence.)

---

## Sequence Clustering

RCSB 30% sequence-identity clusters are used to reduce leakage between dataset splits.

At most one selected protein is taken from each 30% sequence cluster.

The selected:

- Development set;
- Validation set;
- Test A; and
- Test B

are constructed so that they do not share 30% sequence clusters or non-empty UniProt accessions.

---

## Test A / Test B Rule

Post-cutoff candidates are separated into Test A and Test B by searching their sequences directly against pre-cutoff PDB structures, rather than by cluster membership:

```text
post-cutoff protein
        ↓
RCSB sequence search against PDB polymer entities
released on or before 30 September 2021
        ↓
for every returned alignment:
    identity = alignment sequence identity
    coverage = aligned residues of the post-cutoff protein
               / length of the post-cutoff protein
        ↓
does ANY pre-cutoff alignment have
identity ≥ 30% AND coverage ≥ 80%?
        ↓
YES → Test A        NO → Test B
```

The result is stored in `precutoff_search_status`:

| Status | Meaning | Eligible for |
|---|---|---|
| `similar_hit` | at least one pre-cutoff alignment reaches both thresholds | Test A |
| `no_similar_hit` | no pre-cutoff alignment reaches both thresholds (including no hits at all) | Test B |
| `incomplete_search` | more than 1000 hits were returned and none of the inspected ones qualified | neither |
| `api_error` | the search could not be completed | neither |
| `not_applicable_pre_cutoff` | the candidate itself is pre-cutoff | — |

Candidates with `incomplete_search` or `api_error` are left out of both test sets and listed in a warning when the script runs. If a priority target without a `forced_split` cannot be classified, the script stops with an error.

The best pre-cutoff alignment for each post-cutoff candidate is recorded in `best_precutoff_hit`, `best_precutoff_pident`, `best_precutoff_coverage`, and `best_precutoff_evalue`. This is the best qualifying hit for `similar_hit`, or the best-covering near miss otherwise.

The thresholds are set by `TESTB_MAX_PIDENT` and `TESTB_MIN_COVERAGE`.

---

## Priority Targets

Two proteins are always included in the benchmark:

| Protein | PDB | Chain | Split |
|---|---|---|---|
| BCCIPα | 8EXF | B | Test B |
| pro-IL-18 | 8URV | A | Test A |

These targets are retained even if they fail one or more of the standard benchmark filters.

---

## Generated Dataset Files

The CSV files in `datasets/` are the snapshot produced by the current `build_dataset.py` on 2 October 2026, using the RCSB 30% cluster file dated 27 September 2026.

### `all_candidates.csv`

Contains all successfully retrieved pre- and post-cutoff candidates with their per-criterion screening verdicts, overall `screening_decision`, and `passes_standard_filters`.

Current snapshot:

```text
2502 candidates  (500 pre-cutoff + 2000 post-cutoff + 2 priority targets)

                    INCLUDE   FLAG   EXCLUDE
pre-cutoff               68    108       324
post-cutoff             225    490      1285
priority targets          0      0         2
total                   293    598      1611
```

---

### `filtered_candidates.csv`

Contains candidates that pass the standard benchmark filters together with the force-included priority targets.

It also records information used during split construction, including:

- sequence cluster;
- the Test A / Test B sequence-search result (`precutoff_search_status` and the `best_precutoff_*` columns);
- priority-target status; and
- final dataset assignment where applicable.

Current snapshot:

```text
295 candidates  (293 passing the standard filters + 2 priority targets)

post-cutoff Test A / Test B search:
  similar_hit       189   (eligible for Test A)
  no_similar_hit     38   (eligible for Test B)
```

---

### `review_queue.csv`

Candidates with a `FLAG` decision, plus the priority targets, for manual review.

Current snapshot: 600 rows (598 flagged candidates + 2 priority targets).

---

### `manual_inspection_sample.csv`

A random subset of 20 automatically accepted, split-assigned proteins. The script writes empty `manual_verdict` and `manual_notes` columns, which are filled in by hand to confirm the filters behave sensibly.

---

### `dataset_manifest.json`

Records the filter configuration (including the Test A / Test B identity and coverage thresholds), split sizes, and SHA-256 hashes of `development.csv`, `validation.csv`, `test_a.csv`, and `test_b.csv`, so any later change to a locked test set can be detected.

It also records the build date and the URL and `Last-Modified` date of the RCSB 30% cluster file used, since both the PDB and the cluster file change over time and the hashes are only reproducible against the same inputs.

---

### Final Dataset Sizes

```text
development.csv    30 proteins
validation.csv     15 proteins
test_a.csv         20 proteins
test_b.csv         20 proteins
```

The four selected datasets are mutually exclusive and do not share RCSB 30% sequence clusters.

All selected proteins have an `INCLUDE` decision, except the two priority targets (both `EXCLUDE` by the automated screen and force-included).

---

## Reproducibility

The current script uses the following settings:

```text
Random seed:                  42
Pre-cutoff candidates:        500
Post-cutoff candidates:       2000
Maximum search hits:          None (complete result set)

Development set size:         30
Validation set size:          15
Test A size:                  20
Test B size:                  20
```

The current biological filters are:

```text
Minimum length:               80 residues
Maximum length:               400 residues
Minimum resolved fraction:    0.90
Maximum resolution:           3.5 Å
Max unmodelled segment:       20 residues (flag)
Max UniProt-disordered:       20% of construct (flag)
Large ligand:                 ≥ 500 Da or > 25 heavy atoms (fail)
Medium ligand:                ≥ 200 Da or > 12 heavy atoms (flag)
```

The Test A / Test B rule uses:

```text
Minimum identity (Test A):    30%
Minimum coverage (Test A):    80% of the post-cutoff protein
Search hits inspected:        1000 per protein
```

The candidate search retrieves the complete RCSB result set before sampling. RCSB returns hits in a fixed order that front-loads the oldest entries, so capping the search (as an earlier version did at 5000 hits) biases the sample: the pre-cutoff candidates all came from 1988–2004. `MAX_SEARCH_HITS` should only be set to an integer for quick test runs.

2000 post-cutoff candidates are sampled because the 30/80 rule leaves relatively few Test B candidates: with 1200, only 17 distinct Test B proteins remained after removing cluster and UniProt duplicates.

Because RCSB metadata, sequence-cluster files, and sequence-search results may change over time, the committed CSV files should be treated as the saved benchmark snapshot used for the project.

Once the final Test A and Test B datasets are agreed upon, they should be treated as locked test sets and should not be repeatedly regenerated during method development.

---

## Running the Script

From the `stage1/` directory:

```bash
python build_dataset.py
```

The generated CSV files are automatically written to:

```text
stage1/datasets/
```

The script queries RCSB PDB, UniProt, and OPM, so it needs network access to `search.rcsb.org`, `data.rcsb.org`, `cdn.rcsb.org`, `rest.uniprot.org`, and `opm-assets.storage.googleapis.com`.

The script requires the following Python packages:

```text
pandas
requests
```

---

## Next Step

OpenFold3 structure prediction will be performed for proteins in both Test A and Test B.

For each protein, the predicted structure will be compared against its experimentally determined structure.

A continuous metric describing the difference between the OpenFold3 prediction and the experimental structure will then be calculated.

This metric can be used to prioritize proteins for subsequent solvent-accessibility modelling.

Proteins for which the OpenFold3 prediction differs more strongly from the experimental structure will be higher-priority targets for SASA analysis.