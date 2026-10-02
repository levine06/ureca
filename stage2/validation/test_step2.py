"""Independent small-probe and reference-winner comparisons (criteria set before export)."""
import csv
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent

def measurements():
    with (ROOT / "step2_pymol.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["kind"], r["id"], int(r["residue"]), float(r["probe"])) for r in rows]
    assert len(keys) == len(set(keys)) == 100
    return {key: float(row["pymol_A2"]) for key, row in zip(keys, rows)}

def test_small_probe_agreement():
    observed = measurements()
    with (ROOT / "step2_accessfold.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    for row in rows:
        key = ("residue",row["pdb_id"],int(row["residue"]),float(row["probe"]))
        # Absolute tolerance near zero; modest sampling tolerance for larger areas.
        assert observed[key] == pytest.approx(float(row["accessfold_A2"]), abs=3.0, rel=0.03), key

def test_independent_reference_winner_areas():
    observed = measurements()
    rows = json.loads((ROOT / "reference_winners.json").read_text())
    assert len(rows) == 80
    for row in rows:
        key = ("reference",row["name"],2,float(row["probe"]))
        assert observed[key] == pytest.approx(row["stored_max_A2"], abs=3.0, rel=0.02), key
