# Stage 3, step 2: Candidate generation

This is step 2 of the [Stage 3 experiment](../README.md): generating the candidate OpenFold3 structures `X_1, ..., X_K`.

Files in this folder:

- `generate_candidates.py`: builds the OpenFold3 queries, runs inference and writes the manifest.
- `submit_shards.pbs`: PBS array job that runs the shards.

Candidates, the manifest, scratch query files and logs are written to `candidates/`, `candidates_manifest.csv`, `work/` and `logs/` in this folder. Only the manifest is tracked by Git.

`generate_candidates.py` generates a fixed ensemble of OpenFold3 candidate structures for each protein in the development and validation sets.

For each protein, OpenFold3 is run using:

- **5 seeds:** `42`, `123`, `7`, `2024`, `31337`
- **5 diffusion samples per seed**

This gives:

**5 seeds × 5 diffusion samples = 25 candidate structures per protein**

Each `(seed, diffusion sample)` pair is treated as one independent candidate structure.

Templates are deliberately turned off (`--use-templates false`). Template search runs against the PDB and would leak the true structure for these pre-cutoff proteins, so it must stay off.

The candidates are saved under:

```text
candidates/<split>/<PDB>_<CHAIN>/seed_<SEED>/
```

For example:

```text
candidates/development/1L66_A/seed_42/
```

## Sharding

Candidate generation is computationally expensive, so the proteins are divided across multiple PBS jobs using sharding.

For `N` shards:

```text
--shard I/N
```

where `I` is the shard index.

Before assigning proteins to shards, the entries are sorted by sequence length. They are then distributed across the shards in a round-robin manner.

For example, with 4 shards:

```text
Shard 0: protein 1, 5, 9, ...
Shard 1: protein 2, 6, 10, ...
Shard 2: protein 3, 7, 11, ...
Shard 3: protein 4, 8, 12, ...
```

This distributes proteins of different lengths across the jobs instead of placing all of the longest proteins into the same shard.

Sharding only divides the **proteins** between jobs. It does not divide the candidates for a protein: every protein still receives all **25 candidate structures**.

`submit_shards.pbs` launches the shards as a PBS array job, with each array element running one shard independently.

## Running

Submit all shards:

```bash
qsub stage3/candidate_generation/submit_shards.pbs
```

Shard jobs do not write the manifest. Once every array element has finished, build it once:

```bash
python stage3/candidate_generation/generate_candidates.py --manifest-only
```

Re-running is safe: finished candidates are skipped, so a failed shard can simply be resubmitted.

PBS logs are written to `stage3/candidate_generation/logs/` (git-ignored).

## Outputs

The generated OpenFold3 structures are stored in:

```text
stage3/candidate_generation/candidates/
```

A summary of the completed candidates is stored in:

```text
stage3/candidate_generation/candidates_manifest.csv
```

The manifest contains one row per candidate:

| Column | Meaning |
|---|---|
| `split` | `development` or `validation` |
| `query_name` | protein name, `<pdb_id>_<chain_id>` |
| `pdb_id`, `chain_id`, `length` | from the Stage 1 CSV |
| `seed`, `sample` | OpenFold3 seed and diffusion-sample index (1-based) |
| `candidate_id` | unique ID, `<query_name>_s<seed>_n<sample>` |
| `model_path` | candidate `.cif`, relative to the repository root |
| `avg_plddt` | mean pLDDT, 0-100 |
| `ptm` | predicted TM-score |
| `gpde` | global predicted distance error |
| `has_clash` | OpenFold3 clash flag |
| `sample_ranking_score` | OpenFold3 ranking score for the sample |

The candidate structures themselves are not committed to Git because of their size.

## Workflow

```text
Stage 1 development + validation proteins
                |
                v
      generate_candidates.py
                |
        5 seeds per protein
                |
      5 samples per seed
                |
                v
     25 candidates per protein
                |
                v
      candidates_manifest.csv
```

These fixed candidate ensembles are then used for the accessibility-versus-structural-accuracy analysis and subsequent reranking experiments.

## Current status

- 45 proteins (30 development, 15 validation), 25 candidates each: 1125 candidates in the manifest.
- All candidates have `has_clash = 0`.
