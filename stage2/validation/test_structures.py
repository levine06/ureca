"""Real-structure checks against independent PyMOL measurements."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from biotite.structure.io.pdb import PDBFile
from accessfold import compute_accessibility
from accessfold.structures.adapters import from_biotite

ROOT = Path(__file__).resolve().parent

@pytest.mark.parametrize("pdb_id", ["1UBQ", "1CRN", "1LYZ"])
@pytest.mark.parametrize("probe", [1.4, 2.5, 4.0, 6.0])
def test_pymol_agreement(pdb_id, probe):
    record = next(r for r in json.loads((ROOT / "manifest.json").read_text())["structures"] if r["pdb_id"] == pdb_id)
    assert record["release_date"] < "2021-09-30"
    arr = PDBFile.read(str(ROOT / f"{pdb_id}_clean.pdb")).get_structure(model=1)
    structure = from_biotite(arr, "A")
    result = compute_accessibility(structure, probe_radius=probe, n_points=4000)
    with (ROOT / "pymol_sasa.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["pdb_id"] == pdb_id and float(r["probe"]) == probe]
    theirs = np.array([float(r["pymol_sasa"]) for r in rows])
    ours = result.absolute
    assert len(ours) == len(theirs)
    assert np.isfinite(result.y).all()
    # Different sampling grids permit small residue-level differences.
    assert abs(ours.sum() - theirs.sum()) / theirs.sum() < 0.02
    assert np.corrcoef(ours, theirs)[0, 1] > 0.99
    assert np.max(np.abs(ours - theirs)) < 8.0

def test_manual_labels():
    path = ROOT / "manual_labels.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    confirmed = [r for r in rows if r["label"] in ("buried", "exposed")]
    if not confirmed:
        pytest.skip("Manual inspection pending: no confirmed labels")
    for pdb_id in ("1UBQ", "1CRN", "1LYZ"):
        labels = [r for r in confirmed if r["pdb_id"] == pdb_id]
        assert sum(r["label"] == "buried" for r in labels) >= 3
        assert sum(r["label"] == "exposed" for r in labels) >= 3
        arr = PDBFile.read(str(ROOT / f"{pdb_id}_clean.pdb")).get_structure(model=1)
        result = compute_accessibility(from_biotite(arr, "A"), n_points=4000)
        buried = [result.relative[int(r["residue"]) - 1] for r in labels if r["label"] == "buried"]
        exposed = [result.relative[int(r["residue"]) - 1] for r in labels if r["label"] == "exposed"]
        assert max(buried) < min(exposed), f"Inspect classification or calculation for {pdb_id}"
