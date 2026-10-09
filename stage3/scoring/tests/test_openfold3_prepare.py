"""stage3/scoring/prepare_openfold3_outputs.py on a synthetic OpenFold3-style output tree."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
PB = pytest.importorskip("PeptideBuilder")

from accessfold import AtomicStructure
from accessfold.structures.mmcif import ONE_TO_THREE
from accessfold.synthetic import write_chain_cif

ROOT = Path(__file__).resolve().parents[1]
SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def structure(seq):
    s = PB.make_structure(seq, [-60.0] * (len(seq) - 1), [-45.0] * (len(seq) - 1))
    xyz, el, nm, ri = [], [], [], []
    for i, res in enumerate(s[0]["A"]):
        for a in res:
            if a.element != "H":
                xyz.append(a.coord)
                el.append(a.element.upper())
                nm.append(a.get_name())
                ri.append(i)
    return AtomicStructure(np.array(xyz), el, nm, ri, np.array([ONE_TO_THREE[c] for c in seq]))


def make_tree(root, query="AAAA_A", seeds=(42, 43), samples=3, drop_json=None):
    full = structure(SEQ)
    for sd in seeds:
        for i in range(1, samples + 1):
            d = root / query / f"seed_{sd}"
            d.mkdir(parents=True, exist_ok=True)
            base = f"{query}_seed_{sd}_sample_{i}"
            write_chain_cif(d / f"{base}_model.cif", full, full.coords + 0.1 * i, asym_id="A")
            if (sd, i) != drop_json:
                (d / f"{base}_confidences_aggregated.json").write_text(json.dumps(
                    {"avg_plddt": 80.0 + i, "ptm": 0.7 + 0.01 * i, "iptm": None, "gpde": 1.5, "has_clash": 0.0,
                     "sample_ranking_score": 0.5 + sd / 1000 + i / 100, "disorder": 0.1}))
    return full


@pytest.fixture()
def dataset(tmp_path):
    ds = tmp_path / "ds.csv"
    ds.write_text("pdb_id,chain_id,label_chain_id,sequence\n" f"AAAA,A,A,{SEQ}\n" f"BBBB,A,A,{SEQ}\n")
    return ds


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_ranking_table_and_report(tmp_path, dataset):
    make_tree(tmp_path / "of3", drop_json=(43, 2))
    make_tree(tmp_path / "of3", query="ZZZZ_A", seeds=(42,), samples=1)                  # not a Stage 1 entry
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", dataset, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv",
            "--allow-missing")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(tmp_path / "rank.csv")))
    assert len(rows) == 6 and {x["entry"] for x in rows} == {"AAAA_A"} and all(x["validated"] == "ok" for x in rows)
    top = max((x for x in rows if x["score"]), key=lambda x: float(x["score"]))
    assert top["candidate"] == "AAAA_A_seed_43_sample_3_model.cif" and float(top["score"]) == pytest.approx(0.5 + 0.043 + 0.03)
    missing = [x for x in rows if x["score"] == ""]
    assert len(missing) == 1 and "missing" in missing[0]["note"] and missing[0]["candidate"].endswith("seed_43_sample_2_model.cif")
    assert "ZZZZ_A" in r.stdout and "BBBB_A" in r.stdout and "no predictions yet" in r.stdout and "seed_*/*_model.cif*" in r.stdout
    r2 = run(ROOT / "prepare_openfold3_outputs.py", "--csv", dataset, "--predictions", tmp_path / "of3", "--out", tmp_path / "r2.csv")
    assert r2.returncode == 1                                                              # BBBB_A has no predictions


def test_output_is_accepted_by_score_candidates_without_options(tmp_path, dataset):
    make_tree(tmp_path / "of3")
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", dataset, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv", "--allow-missing")
    assert r.returncode == 0, r.stdout + r.stderr
    spec = importlib.util.spec_from_file_location("score_candidates", ROOT / "score_candidates.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    ranking = sc.read_ranking(tmp_path / "rank.csv")                                       # column auto-detection is unambiguous
    meta = {"entry": "AAAA_A", "pdb_id": "AAAA",
            "candidates": ["seed_42/AAAA_A_seed_42_sample_1_model.cif", "seed_43/AAAA_A_seed_43_sample_3_model.cif"]}
    scores = sc.resolve_ranking(ranking, meta)["scores"]                 # bare file names in the table match the unique ids
    assert scores[0] == pytest.approx(0.5 + 0.042 + 0.01) and scores[1] == pytest.approx(0.5 + 0.043 + 0.03)


def test_validation_catches_wrong_sequence_chain_and_numbering(tmp_path, dataset):
    make_tree(tmp_path / "of3", seeds=(42,), samples=2)
    other = "".join("A" if c != "A" else "G" for c in SEQ)                                 # same length, different residues
    bad = structure(other)
    write_chain_cif(tmp_path / "of3" / "AAAA_A" / "seed_42" / "AAAA_A_seed_42_sample_9_model.cif", bad)
    args = [ROOT / "prepare_openfold3_outputs.py", "--csv", dataset, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv", "--allow-missing"]
    r = run(*args)
    assert r.returncode == 1 and "FAILED" in r.stdout and "residue identities differ" in r.stdout
    rows = {x["candidate"]: x["validated"] for x in csv.DictReader(open(tmp_path / "rank.csv"))}
    assert rows["AAAA_A_seed_42_sample_1_model.cif"] == "ok" and rows["AAAA_A_seed_42_sample_9_model.cif"].startswith("FAILED")
    (tmp_path / "of3" / "AAAA_A" / "seed_42" / "AAAA_A_seed_42_sample_9_model.cif").unlink()
    assert run(*args).returncode == 0
    r = run(*args, "--chain", "B")                                                         # wrong chain id: reports what the file has
    assert r.returncode == 1 and "no polymer atoms" in r.stdout and "polymer chains in the file: ['A']" in r.stdout
    assert run(*args, "--chain", "auto").returncode == 0
    r = run(*args, "--no-validate")
    assert r.returncode == 0 and "skipped" in (tmp_path / "rank.csv").read_text()


def test_query_named_by_pdb_id_is_matched(tmp_path, dataset):
    make_tree(tmp_path / "of3", query="AAAA", seeds=(42,), samples=1)
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", dataset, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv", "--allow-missing")
    assert r.returncode == 0, r.stdout + r.stderr
    row = list(csv.DictReader(open(tmp_path / "rank.csv")))[0]
    assert row["entry"] == "AAAA_A" and row["query"] == "AAAA" and "{pdb_id}/seed_*" in r.stdout


def test_printed_pattern_includes_development_validation_subfolders(tmp_path):
    ds = tmp_path / "ds.csv"
    ds.write_text("pdb_id,chain_id,label_chain_id,sequence\n" f"AAAA,A,A,{SEQ}\n" f"BBBB,A,A,{SEQ}\n")
    make_tree(tmp_path / "of3" / "development", query="AAAA_A", seeds=(42,), samples=1)
    make_tree(tmp_path / "of3" / "validation", query="BBBB_A", seeds=(42, 43), samples=1)
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", ds, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'"{(tmp_path / "of3").as_posix()}/*/{{entry}}/seed_*/*_model.cif*"' in r.stdout
    # the printed pattern really finds every model, under both subfolders
    import glob
    pat = f"{(tmp_path / 'of3').as_posix()}/*/{{entry}}/seed_*/*_model.cif*"
    assert len(glob.glob(pat.format(entry="AAAA_A"))) == 1 and len(glob.glob(pat.format(entry="BBBB_A"))) == 2


def test_printed_pattern_keeps_common_subfolder_and_flags_mixed_layouts(tmp_path):
    ds = tmp_path / "ds.csv"
    ds.write_text("pdb_id,chain_id,label_chain_id,sequence\n" f"AAAA,A,A,{SEQ}\n" f"BBBB,A,A,{SEQ}\n")
    make_tree(tmp_path / "of3" / "runs" / "development", query="AAAA_A", seeds=(42,), samples=1)
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", ds, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv", "--allow-missing")
    assert f'/of3/runs/development/{{entry}}/seed_*/*_model.cif*"' in r.stdout and "NOTE" not in r.stdout
    make_tree(tmp_path / "of3", query="BBBB_A", seeds=(42,), samples=1)                    # a second, shallower layout
    r = run(ROOT / "prepare_openfold3_outputs.py", "--csv", ds, "--predictions", tmp_path / "of3", "--out", tmp_path / "rank.csv")
    assert r.stdout.count(" --candidate-chain ") == 2 and "one pattern per layout" in r.stdout
