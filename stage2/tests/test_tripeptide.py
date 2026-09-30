"""Tests for the Gly-X-Gly reference generator (needs PeptideBuilder + Biopython)."""
import numpy as np
import pytest

pytest.importorskip("PeptideBuilder")

from accessfold.accessibility import tripeptide as T
from accessfold.accessibility.reference import STANDARD_RESIDUES, TIEN2013_THEORETICAL_1P4
from accessfold.accessibility.sasa import atom_sasa

CFG = T.TripeptideConfig()
EXPECTED_K = {"ALA": 1, "GLY": 1, "PRO": 1, "SER": 3, "CYS": 3, "THR": 3, "VAL": 3, "ILE": 9, "LEU": 9,
              "ASP": 18, "PHE": 18, "TYR": 18, "ASN": 36, "HIS": 36, "TRP": 36, "MET": 27, "GLU": 54,
              "GLN": 108, "LYS": 81, "ARG": 81}


def _topo(res, phi=-120.0, psi=140.0):
    coords, names, elems, resi = T._structure_arrays(T._build_conformer_structure(res, "G", phi, psi))
    return coords, T._make_topology(res, coords, names, elems, resi, CFG)


def test_staggered_grid_sizes_are_as_documented():
    src = T.StaggeredChiGrid()
    assert set(EXPECTED_K) == set(STANDARD_RESIDUES)
    for res, k in EXPECTED_K.items():
        assert src.chi_sets(res, 0, 0).shape == (k, len(T.CHI_ATOMS.get(res, []))), res


@pytest.mark.parametrize("res", ["TRP", "ARG", "ILE", "GLU", "PHE", "LYS"])
def test_chi_rotation_hits_targets_and_preserves_bonds(res):
    coords, topo = _topo(res)
    chi = T.StaggeredChiGrid().chi_sets(res, 0, 0)
    sel = chi[np.random.default_rng(0).integers(0, len(chi), 25)]
    out = T._apply_chi(np.repeat(coords[None], len(sel), 0), topo, sel)
    meas = np.stack([T.dihedral_deg(*[out[:, i] for i in idx]) for idx in topo.chi_idx], axis=1)
    assert np.abs((meas - sel + 180) % 360 - 180).max() < 1e-6
    d0 = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    d1 = np.linalg.norm(out[:, :, None] - out[:, None], axis=-1)
    bonded = d0 < 1.9
    assert np.abs(d1[:, bonded] - d0[bonded]).max() < 1e-9
    # atoms that no chi rotation is allowed to move (backbone, flanks, CB) are untouched
    movable = np.unique(np.concatenate(topo.chi_moving))
    fixed = np.setdiff1d(np.arange(len(coords)), movable)
    assert np.abs(out[:, fixed] - coords[fixed][None]).max() < 1e-9


def test_chi_moving_sets_have_expected_sizes():
    assert [len(m) for m in _topo("TRP")[1].chi_moving] == [9, 8]
    assert [len(m) for m in _topo("ARG")[1].chi_moving] == [6, 5, 4, 3]
    assert _topo("ALA")[1].chi_moving == [] and _topo("GLY")[1].chi_moving == []


def test_backbone_and_flanks_do_not_move_under_chi_changes():
    coords, topo = _topo("LYS")
    out = T._apply_chi(coords[None], topo, np.array([[60.0, 60.0, 60.0, 60.0]]))
    side = np.flatnonzero(~np.isin(topo.atom_names, ["N", "CA", "C", "O"]) & (topo.residue_of_atom == 1))
    moved = np.flatnonzero(np.linalg.norm(out[0] - coords, axis=1) > 1e-9)
    assert set(moved) <= set(side)
    assert (topo.residue_of_atom[moved] == 1).all()


@pytest.mark.parametrize("probe", [1.4, 4.0])
def test_batched_sasa_equals_the_main_engine(probe):
    coords, topo = _topo("TRP", -60.0, -40.0)
    confs = T._apply_chi(np.repeat(coords[None], 4, 0), topo, T.StaggeredChiGrid().chi_sets("TRP", 0, 0)[:4])
    batched = T.batch_target_sasa(confs, topo.radii, topo.target, probe, 1000, np.float64)
    ref = np.stack([atom_sasa(x, topo.radii, probe, 1000)[topo.target] for x in confs])
    assert np.allclose(batched, ref, atol=1e-9)


def test_clash_filter_accepts_extended_and_rejects_eclipsed_backbone():
    ext, topo = _topo("ALA", -120.0, 140.0)
    ecl, _ = _topo("ALA", 0.0, 0.0)
    ok = T._clash_free(np.stack([ext, ecl]), topo)
    assert ok.tolist() == [True, False]


def test_scan_reproduces_paper_scale_for_small_residues_and_orders_them():
    """Coarse (30 deg) scan: fast, and only checks scale/ordering, not the final table (see bundled-table tests)."""
    cfg = T.TripeptideConfig(phi_psi_step=30.0, n_points_refine=4000, top_m=6)
    scans = {r: T.scan_residue(r, (1.4, 4.0), cfg) for r in ("GLY", "ALA", "SER", "TRP")}
    for res in ("GLY", "ALA"):
        paper = TIEN2013_THEORETICAL_1P4.max_asa[res]
        assert scans[res].max_total[1.4] == pytest.approx(paper, rel=0.06), res
    for r in (1.4, 4.0):
        vals = [scans[k].max_total[r] for k in ("GLY", "ALA", "SER", "TRP")]
        assert vals == sorted(vals)
    assert 1.4 not in scans["GLY"].max_side          # glycine has no side chain
    for k in ("ALA", "SER", "TRP"):
        assert 0 < scans[k].max_side[1.4] < scans[k].max_total[1.4]
    assert all(scans[k].max_total[4.0] > scans[k].max_total[1.4] for k in scans)  # bigger probe, bigger sphere area


def test_refinement_removes_the_upward_bias_of_a_maximum_over_noisy_estimates():
    cfg = T.TripeptideConfig(phi_psi_step=30.0, n_points_refine=8000, top_m=8)
    sc = T.scan_residue("ARG", (1.4,), cfg)
    assert sc.max_total[1.4] <= sc.coarse_max_total[1.4] * 1.002
    assert sc.max_total[1.4] == pytest.approx(sc.coarse_max_total[1.4], rel=0.03)


def test_builder_returns_one_consistent_table_per_radius():
    cfg = T.TripeptideConfig(phi_psi_step=60.0, n_points_refine=2000, top_m=4)
    tabs = T.build_tripeptide_references((1.4, 2.5), residues=["GLY", "ALA", "SER"], config=cfg)
    assert set(tabs) == {1.4, 2.5}
    for r, t in tabs.items():
        assert t.probe_radius == r and set(t.max_asa) == {"GLY", "ALA", "SER"}
        assert set(t.sidechain_max_asa) == {"ALA", "SER"}
        assert t.radii_table == "bondi" and t.extra["config"]["clash_scale"] == 0.80
        assert "differs_from_paper" in t.extra and t.extra["rotamers"]["source"] == "staggered_chi_grid_v1"
    assert all(tabs[2.5].max_asa[k] > tabs[1.4].max_asa[k] for k in tabs[1.4].max_asa)


def test_nonfinite_conformers_are_rejected_not_read_as_fully_exposed():
    coords, topo = _topo("MET")
    bad = coords.copy()
    bad[5] = np.nan
    assert T._clash_free(np.stack([coords, bad]), topo).tolist() == [True, False]
    with pytest.raises(ValueError, match="non-finite"):
        T.batch_target_sasa(bad[None], topo.radii, topo.target, 1.4, 100)


def test_the_known_degenerate_peptidebuilder_angle_does_not_corrupt_the_met_maximum():
    """PeptideBuilder yields NaN for Met at (phi, psi) = (0, 90); the scan must survive it and stay physical."""
    grid = np.array([[0.0, 90.0], [-120.0, 140.0], [-60.0, -40.0]])
    cfg = T.TripeptideConfig(n_points_refine=2000, top_m=4)
    with np.errstate(all="ignore"):
        sc = T.scan_residue("MET", (1.4,), cfg, phipsi=grid)
    assert sc.n_nonfinite >= 1
    assert 150.0 < sc.max_total[1.4] < 260.0  # Tien's Met value is 224; NaN-as-exposed would give a value far above


def test_scan_result_does_not_depend_on_grid_order_or_on_the_first_grid_point():
    grid = np.array([[-120.0, 140.0], [-60.0, -40.0], [60.0, 40.0], [-90.0, 0.0]])
    cfg = T.TripeptideConfig(n_points_refine=2000, top_m=4)
    a = T.scan_residue("LYS", (1.4,), cfg, phipsi=grid)
    b = T.scan_residue("LYS", (1.4,), cfg, phipsi=grid[::-1].copy())
    assert a.max_total[1.4] == pytest.approx(b.max_total[1.4], rel=1e-12)
    assert a.max_side[1.4] == pytest.approx(b.max_side[1.4], rel=1e-12)
    assert a.n_clash_rejected == b.n_clash_rejected
