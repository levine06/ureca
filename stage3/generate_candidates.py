#!/usr/bin/env python
"""Stage 3 -- generate multiple OpenFold3 candidate structures per benchmark entry.

Reads the Stage 1 development and validation CSVs, builds an OpenFold3 query for
every entry (one protein chain each), and runs ``run_openfold predict`` with
several model seeds and several diffusion samples per seed.  Every
(seed, sample) pair is one candidate structure X_k.

Output layout (OpenFold3's own layout, one directory per entry):

    stage3/candidates/<split>/<PDB>_<CHAIN>/seed_<S>/<name>_seed_<S>_sample_<N>_model.cif
    stage3/candidates/<split>/<PDB>_<CHAIN>/seed_<S>/<name>_seed_<S>_sample_<N>_confidences_aggregated.json

After inference, ``candidates_manifest.csv`` lists every candidate with its
split, seed, sample, model path and OpenFold3 confidence scores, so the later
accessibility step only needs to read that one file.

Examples
--------
    # everything (30 dev + 15 val entries, 5 seeds x 5 samples each)
    python stage3/generate_candidates.py

    # smoke test on one entry with a few candidates
    python stage3/generate_candidates.py --only 1L66_A --seeds 42 7 --num-diffusion-samples 2

    # run 4 OpenFold3 processes side by side inside one job (threads split evenly)
    python stage3/generate_candidates.py --workers 4

    # or split the work over separate PBS jobs: see stage3/submit_shards.pbs
    python stage3/generate_candidates.py --shard 0/4

    # rebuild the manifest from whatever has finished, without running inference
    python stage3/generate_candidates.py --manifest-only

Re-running is safe: finished candidates are skipped (``skip_existing``).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

URECA_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = URECA_ROOT / "stage1" / "datasets"
STAGE3_DIR = URECA_ROOT / "stage3"
DEFAULT_RUNNER_YAML = URECA_ROOT / "cpu_inference.yml"
DEFAULT_OPENFOLD_REPO = URECA_ROOT / "openfold-3"

SPLIT_CSVS = {"development": "development.csv", "validation": "validation.csv"}
DEFAULT_SEEDS = [42, 123, 7, 2024, 31337]

MODEL_RE = re.compile(r"^(?P<name>.+)_seed_(?P<seed>\d+)_sample_(?P<sample>\d+)_model\.cif$")


def query_name(row):
    """Unique, filesystem-safe name for one benchmark entry."""
    return f"{row['pdb_id']}_{row['chain_id']}"


def load_entries(splits):
    frames = []
    for split in splits:
        df = pd.read_csv(STAGE1_DIR / SPLIT_CSVS[split])
        df["split"] = split
        frames.append(df)
    entries = pd.concat(frames, ignore_index=True)
    entries["query_name"] = entries.apply(query_name, axis=1)
    if entries["query_name"].duplicated().any():
        dup = entries.loc[entries["query_name"].duplicated(), "query_name"].tolist()
        sys.exit(f"Duplicate entries across splits: {dup}")
    if entries["sequence"].isna().any():
        sys.exit("Some entries have no sequence.")
    return entries


def select(entries, only, shard):
    if only:
        missing = set(only) - set(entries["query_name"])
        if missing:
            sys.exit(f"--only entries not found: {sorted(missing)}")
        entries = entries[entries["query_name"].isin(only)]
    if shard:
        index, total = shard
        # Round-robin on sorted length so shards get a similar mix of sizes.
        entries = entries.sort_values("length").reset_index(drop=True)
        entries = entries[entries.index % total == index]
    return entries.reset_index(drop=True)


def parse_shard(text):
    try:
        index, total = (int(x) for x in text.split("/"))
        assert 0 <= index < total
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("--shard must look like I/N with 0 <= I < N")
    return index, total


def build_query(entries):
    return {
        "queries": {
            row.query_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [row.chain_id],
                        "sequence": row.sequence,
                    }
                ]
            }
            for row in entries.itertuples()
        }
    }


def build_runner(runner_yaml, seeds, use_msa_server, use_templates):
    runner = yaml.safe_load(Path(runner_yaml).read_text())
    exp = runner.setdefault("experiment_settings", {})
    exp["seeds"] = list(seeds)
    exp["skip_existing"] = True
    exp["use_msa_server"] = use_msa_server
    exp["use_templates"] = use_templates
    return runner


def thread_env(threads):
    """Environment for OpenFold3 with the CPU thread count pinned (if given)."""
    env = os.environ.copy()
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[var] = str(threads)
    return env


def available_cpus():
    """CPUs granted to this job: PBS NCPUS if set, else what the OS lets us use."""
    if os.environ.get("NCPUS"):
        return int(os.environ["NCPUS"])
    return len(os.sched_getaffinity(0))


def strip_option(argv, name):
    """Drop `--name value` / `--name=value` from an argv list."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
        elif tok == name:
            skip = True
        elif not tok.startswith(name + "="):
            out.append(tok)
    return out


def run_workers(args):
    """Launch one subprocess per shard, splitting the CPUs between them."""
    threads = args.threads or max(1, available_cpus() // args.workers)
    base = strip_option(strip_option(sys.argv[1:], "--workers"), "--threads")
    work_dir = STAGE3_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.workers} workers x {threads} threads", flush=True)

    procs = []
    try:
        for i in range(args.workers):
            log_path = work_dir / f"log_shard{i}of{args.workers}.txt"
            log = open(log_path, "w")
            cmd = [sys.executable, str(Path(__file__).resolve()), *base,
                   "--shard", f"{i}/{args.workers}", "--threads", str(threads)]
            print(f"  worker {i}: log -> {log_path}", flush=True)
            procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        failed = []
        for i, proc, log in procs:
            if proc.wait() != 0:
                failed.append(i)
            log.close()
    except BaseException:
        for _, proc, _ in procs:
            proc.terminate()
        raise
    if failed:
        sys.exit(f"Workers {failed} failed; see stage3/work/log_shard*.txt. "
                 "Re-run the same command to resume (finished candidates are skipped).")


def out_dir_for(split):
    return STAGE3_DIR / "candidates" / split


def run_inference(args, entries):
    work_dir = STAGE3_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"shard{args.shard[0]}of{args.shard[1]}" if args.shard else "all"

    runner_path = work_dir / f"runner_{shard_tag}.yml"
    runner_path.write_text(
        yaml.safe_dump(
            build_runner(args.runner_yaml, args.seeds, args.use_msa_server, args.use_templates),
            sort_keys=False,
        )
    )

    # One OpenFold3 call per split so each split lands in its own output directory.
    for split, group in entries.groupby("split", sort=False):
        query_path = work_dir / f"query_{split}_{shard_tag}.json"
        query_path.write_text(json.dumps(build_query(group), indent=2))
        out_dir = out_dir_for(split)
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            args.openfold_bin,
            "predict",
            "--query-json", str(query_path),
            "--output-dir", str(out_dir),
            "--use-msa-server", str(args.use_msa_server),
            "--use-templates", str(args.use_templates),
            "--runner-yaml", str(runner_path),
            "--num-diffusion-samples", str(args.num_diffusion_samples),
        ]
        print(f"[{split}] {len(group)} entries -> {out_dir}", flush=True)
        print("  " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=args.openfold_repo, check=True, env=thread_env(args.threads))


def build_manifest(entries_all):
    """Scan the output tree and list every finished candidate."""
    rows = []
    for split in SPLIT_CSVS:
        split_dir = out_dir_for(split)
        if not split_dir.is_dir():
            continue
        for entry in entries_all[entries_all["split"] == split].itertuples():
            for cif in sorted((split_dir / entry.query_name).glob("seed_*/*_model.cif")):
                match = MODEL_RE.match(cif.name)
                if not match:
                    continue
                row = {
                    "split": split,
                    "query_name": entry.query_name,
                    "pdb_id": entry.pdb_id,
                    "chain_id": entry.chain_id,
                    "length": entry.length,
                    "seed": int(match["seed"]),
                    "sample": int(match["sample"]),
                    "candidate_id": f"{entry.query_name}_s{match['seed']}_n{match['sample']}",
                    "model_path": str(cif.relative_to(URECA_ROOT)),
                }
                conf = cif.with_name(cif.name.replace("_model.cif", "_confidences_aggregated.json"))
                if conf.exists():
                    scores = json.loads(conf.read_text())
                    for key in ("avg_plddt", "ptm", "gpde", "has_clash", "sample_ranking_score"):
                        row[key] = scores.get(key)
                rows.append(row)
    manifest = pd.DataFrame(rows)
    path = STAGE3_DIR / "candidates_manifest.csv"
    manifest.to_csv(path, index=False)
    return manifest, path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(SPLIT_CSVS), choices=list(SPLIT_CSVS))
    p.add_argument("--only", nargs="+", help="restrict to these entries, e.g. 1L66_A")
    p.add_argument("--shard", type=parse_shard, help="process shard I of N (I/N), for parallel jobs")
    p.add_argument("--workers", type=int, default=1,
                   help="run N shard processes in parallel in this job; CPUs are split between them")
    p.add_argument("--threads", type=int,
                   help="CPU threads for this OpenFold3 process (default: all visible; "
                        "with --workers, available CPUs / workers)")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="model seeds; each seed gives an independent trunk+diffusion run")
    p.add_argument("--num-diffusion-samples", type=int, default=5,
                   help="structures sampled per seed (candidates per entry = seeds x samples)")
    p.add_argument("--use-msa-server", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--use-templates", type=lambda s: s.lower() == "true", default=False,
                   help="default off: templates are searched against the PDB and would "
                        "leak the true structure into these pre-cutoff proteins")
    p.add_argument("--runner-yaml", default=DEFAULT_RUNNER_YAML)
    p.add_argument("--openfold-bin", default=shutil.which("run_openfold") or "run_openfold")
    p.add_argument("--openfold-repo", default=DEFAULT_OPENFOLD_REPO)
    p.add_argument("--dry-run", action="store_true", help="write queries and print commands only")
    p.add_argument("--manifest-only", action="store_true", help="skip inference, just rebuild the manifest")
    args = p.parse_args()

    if args.workers > 1:
        if args.shard or args.manifest_only or args.dry_run:
            sys.exit("--workers cannot be combined with --shard, --manifest-only or --dry-run")
        run_workers(args)
        manifest, path = build_manifest(load_entries(args.splits))
        print(f"Manifest: {len(manifest)} candidates -> {path}")
        return

    entries_all = load_entries(args.splits)
    if not args.manifest_only:
        entries = select(entries_all, args.only, args.shard)
        print(f"{len(entries)} entries x {len(args.seeds)} seeds x {args.num_diffusion_samples} samples "
              f"= {len(entries) * len(args.seeds) * args.num_diffusion_samples} candidates", flush=True)
        run_inference(args, entries)

    if args.shard:
        # Shards run concurrently; building the manifest is left to the last step
        # (--manifest-only) so they don't write the same file at once.
        print("Shard done. Run with --manifest-only after all shards finish.")
    elif not args.dry_run:
        manifest, path = build_manifest(entries_all)
        print(f"Manifest: {len(manifest)} candidates -> {path}")


if __name__ == "__main__":
    main()
