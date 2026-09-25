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
    ├── development.csv
    ├── validation.csv
    ├── test_a.csv
    └── test_b.csv
```

## `build_dataset.py`

The main script is used to:

1. Search RCSB PDB for pre- and post-cutoff structures.
2. Retrieve structural and sequence metadata.
3. Apply the benchmark filters.
4. Assign RCSB 30% sequence-identity clusters.
5. Determine whether post-cutoff clusters contain pre-cutoff structures.
6. Construct cluster-disjoint development, validation, Test A, and Test B splits.
7. Force-include two priority targets.
8. Save the generated datasets as CSV files.

---

## Initial Benchmark Criteria

Standard benchmark candidates are restricted to proteins with:

- sequence length between 80 and 400 residues;
- at least 90% of the structure resolved;
- resolution ≤ 3.5 Å;
- X-ray diffraction or electron microscopy structures;
- a monomeric biological assembly;
- no detected membrane-protein annotation;
- only the 20 standard amino-acid residues; and
- an available RCSB 30% sequence cluster.

The two priority proteins are deliberately force-included even when they do not satisfy all of the standard automated filters.

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

For this reason, BCCIPα is included in Test B.

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

### `all_candidates.csv`

Contains all successfully retrieved pre- and post-cutoff candidates before the standard benchmark filters are applied.

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
Pre-cutoff candidates:        300
Post-cutoff candidates:       700
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