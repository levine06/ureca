import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")

from accessfold import compute_accessibility
from accessfold.structures.adapters import from_biotite


def _atom(res_id, res_name, name, element, xyz, chain="A", hetero=False):
    a = struc.Atom(np.array(xyz, dtype=np.float32), chain_id=chain, res_id=res_id, ins_code="", res_name=res_name,
                   hetero=hetero, atom_name=name, element=element)
    return a


def test_from_biotite_builds_gap_and_drops_water_hetero_and_other_chains():
    atoms = [
        _atom(1, "ALA", "N", "N", (0, 0, 0)), _atom(1, "ALA", "CA", "C", (1.4, 0, 0)), _atom(1, "ALA", "CB", "C", (1.8, 1.4, 0)),
        _atom(4, "GLY", "N", "N", (9, 0, 0)), _atom(4, "GLY", "CA", "C", (10.4, 0, 0)),          # residues 2, 3 unresolved
        _atom(100, "HOH", "O", "O", (5, 5, 5), hetero=True),
        _atom(1, "ALA", "CA", "C", (30, 0, 0), chain="B"),
    ]
    arr = struc.array(atoms)
    s = from_biotite(arr, chain_id="A")
    assert s.n_residues == 4 and s.n_atoms == 5
    assert s.resolved_mask.tolist() == [True, False, False, True]
    assert s.residue_names.tolist() == ["ALA", "UNK", "UNK", "GLY"]
    r = compute_accessibility(s, "absolute_sasa", 1.4)
    assert np.isnan(r.y[1]) and np.isfinite(r.y[0]) and np.isfinite(r.y[3])


def test_from_biotite_rejects_insertion_codes():
    a = _atom(1, "ALA", "CA", "C", (0, 0, 0))
    a.ins_code = "A"
    with pytest.raises(ValueError, match="insertion"):
        from_biotite(struc.array([a]))
