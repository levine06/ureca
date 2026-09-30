import os

import numpy as np
import pytest

from accessfold.accessibility.dunbrack import DunbrackLibrary
from accessfold.accessibility.tripeptide import TripeptideConfig, scan_residue

_HEADER = "# comment line\n# T  Phi  Psi  Count    r1 r2 r3 r4 Probabil  chi1Val chi2Val chi3Val chi4Val  chi1Sig chi2Sig chi3Sig chi4Sig\n"
_ROWS = [
    "VAL -180 -180 10 2 0 0 0 0.900000 170.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VAL -180 -180 10 1 0 0 0 0.095000 65.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VAL -180 -180 10 3 0 0 0 0.005000 -60.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VAL 180 180 10 2 0 0 0 0.900000 171.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VAL 180 180 10 1 0 0 0 0.095000 66.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VAL -60 -40 10 1 0 0 0 0.004000 60.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "SER -60 -40 10 1 0 0 0 0.600000 64.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",
    "VALX -60 -40 10 1 0 0 0 0.500000 1.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0",  # must not match VAL
]


@pytest.fixture()
def lib_path(tmp_path):
    p = tmp_path / "mini.lib"
    p.write_text(_HEADER + "\n".join(_ROWS) + "\n")
    return str(p)


def test_probability_floor_and_shapes(lib_path):
    chi = DunbrackLibrary(lib_path, min_probability=0.01).chi_sets("VAL", -180.0, -180.0)
    assert chi.shape == (2, 1) and set(chi[:, 0]) == {170.0, 65.0}
    assert DunbrackLibrary(lib_path, min_probability=0.0).chi_sets("VAL", -180.0, -180.0).shape == (3, 1)


def test_plus_and_minus_180_are_the_same_grid_point(lib_path):
    lib = DunbrackLibrary(lib_path)
    assert lib.chi_sets("VAL", 180.0, 180.0).shape == (2, 1)
    # a file that only stores the -180 end is still found from +180 (and the reverse)


def test_below_floor_keeps_most_probable_rotamer(lib_path):
    chi = DunbrackLibrary(lib_path, min_probability=0.01).chi_sets("VAL", -60.0, -40.0)
    assert chi.shape == (1, 1) and chi[0, 0] == 60.0


def test_residue_name_is_matched_exactly_and_no_chi_residues(lib_path):
    lib = DunbrackLibrary(lib_path)
    assert lib.chi_sets("SER", -60.0, -40.0).tolist() == [[64.0]]
    assert lib.chi_sets("GLY", 0.0, 0.0).shape == (1, 0)
    assert lib.chi_sets("PRO", 0.0, 0.0).shape == (1, 0)
    with pytest.raises(KeyError):
        lib.chi_sets("VAL", 0.0, 0.0)


def test_describe_records_choices(lib_path):
    d = DunbrackLibrary(lib_path, min_probability=0.02).describe()
    assert d["min_probability"] == 0.02 and d["backbone_dependent"] is True and "Dunbrack" in d["citation"]


@pytest.mark.skipif("ACCESSFOLD_DUNBRACK_LIB" not in os.environ, reason="set ACCESSFOLD_DUNBRACK_LIB to ALL.bbdep.rotamers.lib")
def test_real_library_gives_maxima_close_to_staggered_grid():
    lib = DunbrackLibrary(os.environ["ACCESSFOLD_DUNBRACK_LIB"])
    cfg = TripeptideConfig(phi_psi_step=30.0, n_points=500, n_points_refine=2000, top_m=8)
    for res in ("LEU", "SER"):
        a = scan_residue(res, (1.4,), cfg).max_total[1.4]
        b = scan_residue(res, (1.4,), cfg, source=lib).max_total[1.4]
        assert 0.85 < b / a <= 1.02, (res, a, b)
