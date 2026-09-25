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

1. Search RCSB PDB for pre- and post-cutoff single-chain structures.
2. Retrieve structural and sequence metadata in batches from the RCSB GraphQL API.
3. Retrieve UniProt annotations and check the OPM membrane-protein database.
4. Screen every candidate against the Stage 1 biological restrictions.
5. Assign RCSB 30% sequence-identity clusters.
6. Determine whether post-cutoff clusters contain pre-cutoff structures.
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
| `soluble` | UniProt annotates a lipid anchor, lipidation, or GPI anchor | UniProt subcellular location includes a membrane or the cell surface, or the UniProt entry could not be retrieved |
| `not_membrane` | the entry is in OPM, is annotated by PDBTM/OPM/MemProtMD/mpstruc in RCSB, or a UniProt transmembrane segment lies inside the construct | the construct is a soluble domain whose transmembrane segment lies outside it, or OPM could not be reached |
| `length_80_400` | the sequence is outside 80–400 residues | — |
| `resolved_ge_0_90` | less than 90% of the chain is modelled (from the chain's RCSB unobserved-residue annotation) | — |
| `structure_quality` | the method is not X-ray/EM, resolution is worse than 3.5 Å, or the sequence contains non-standard residues | — |
| `not_heavily_disordered` | the chain is less than 90% resolved | a single unmodelled segment is longer than 20 residues, or UniProt disordered regions cover more than 20% of the construct |
| `no_large_ligand` | a non-trivial ligand is ≥ 500 Da or has > 25 heavy atoms (e.g. heme, FAD, NAD) | a non-trivial ligand is ≥ 200 Da or has > 12 heavy atoms |

Water, ions, buffers, cryoprotectants, and common crystallisation additives are ignored by the ligand check.

The overall `screening_decision` is `EXCLUDE` if any criterion fails, `FLAG` if any criterion is flagged, and `INCLUDE` otherwise. Notes explaining each failure and flag are stored in `screening_notes` and `flag_reasons`.

Candidates eligible for the splits (`passes_standard_filters`) must also have an RCSB 30% sequence cluster, and:

- **Development / validation** (pre-cutoff): `INCLUDE` or `FLAG`, since these sets are used freely during method development.
- **Test A / Test B** (post-cutoff): `INCLUDE` only, so the locked test sets contain no unreviewed exceptions.

These choices are controlled by `ALLOW_FLAGGED_IN_DEV_VAL` and `ALLOW_FLAGGED_IN_TEST`.

The two priority proteins are deliberately force-included even when they do not satisfy all of the standard automated filters. Their automated verdicts are kept and the override is recorded in `flag_reasons`.

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

Test A contains proteins whose experimental structures were released after 30 September 2021.

It tests performance on structures that were not directly present within the OpenFold3 structural-training period.

Test A and Test B are mutually exclusive.

#### Priority Target

- **pro-IL-18**
- PDB: `8URV`
- Chain: `A`
- Split: Test A

---

### Test B — Difficult-Generalization Test

File:

```text
datasets/test_b.csv
```

Current size: **20 proteins**

For normal candidates, Test B contains post-cutoff proteins whose RCSB 30% sequence cluster contains no PDB structure released on or before 30 September 2021.

This is intended to provide a more difficult generalization test where the model cannot simply rely on a very closely related pre-cutoff structural example.

Test A and Test B are mutually exclusive.

#### Priority Target

- **BCCIPα**
- PDB: `8EXF`
- Chain: `B`
- Split: Test B

BCCIPα is treated as a special edge case.

Although BCCIPα has strong sequence similarity to the pre-cutoff BCCIPβ structure, its experimentally determined 3D fold is substantially different.

It is also one of the more interesting cases where the AlphaFold3 prediction differs substantially from the experimentally determined structure.

For this reason, BCCIPα is pinned to Test B through `forced_split` in `PRIORITY_TARGETS`, regardless of its current 30% sequence cluster. The script raises an error if it is not placed there.

---

## Sequence Clustering

RCSB 30% sequence-identity clusters are used to reduce leakage between dataset splits.

At most one selected protein is taken from each 30% sequence cluster.

The selected:

- Development set;
- Validation set;
- Test A; and
- Test B

are constructed so that they do not share 30% sequence clusters.

For Test B, the script additionally checks whether any member of a post-cutoff candidate's 30% sequence cluster has a PDB structure released on or before 30 September 2021.

If no pre-cutoff member exists, the candidate can be considered for Test B.

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

> **Note:** the committed CSV snapshot below was produced by the previous version of `build_dataset.py`, before the UniProt/OPM/ligand biological screen was added. Re-run the script to regenerate the datasets (including `review_queue.csv`, `manual_inspection_sample.csv`, and `dataset_manifest.json`) with the current filters.

### `all_candidates.csv`

Contains all successfully retrieved pre- and post-cutoff candidates with their per-criterion screening verdicts, overall `screening_decision`, and `passes_standard_filters`.

Current snapshot:

```text
1002 candidates
```

---

### `filtered_candidates.csv`

Contains candidates that pass the standard benchmark filters together with the force-included priority targets.

It also records information used during split construction, including:

- sequence cluster;
- whether the cluster has a pre-cutoff member;
- priority-target status; and
- final dataset assignment where applicable.

Current snapshot:

```text
247 candidates
```

---

### `review_queue.csv`

Candidates with a `FLAG` decision, plus the priority targets, for manual review.

---

### `manual_inspection_sample.csv`

A random subset of 20 automatically accepted, split-assigned proteins with empty `manual_verdict` and `manual_notes` columns. Inspect these by hand to confirm the filters behave sensibly.

---

### `dataset_manifest.json`

Records the filter configuration, split sizes, and SHA-256 hashes of `development.csv`, `validation.csv`, `test_a.csv`, and `test_b.csv`, so any later change to a locked test set can be detected.

---

### Final Dataset Sizes

```text
development.csv    30 proteins
validation.csv     15 proteins
test_a.csv         20 proteins
test_b.csv         20 proteins
```

The four selected datasets are mutually exclusive and do not share RCSB 30% sequence clusters.

---

## Reproducibility

The current script uses the following settings:

```text
Random seed:                  42
Pre-cutoff candidates:        500
Post-cutoff candidates:       1200
Maximum search hits:          5000

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

Because RCSB metadata and sequence-cluster files may change over time, the committed CSV files should be treated as the saved benchmark snapshot used for the project.

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