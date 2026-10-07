import warnings

import numpy as np
import pytest

from accessfold import ReferenceTable, build_empirical_reference, get_reference, register_reference
from accessfold.accessibility.reference import STANDARD_RESIDUES, TIEN2013_THEORETICAL_1P4, residue_sasa
from conftest import make_fake_protein


def test_tien_table_is_complete_and_ordered_sensibly():
    t = TIEN2013_THEORETICAL_1P4.max_asa
    assert set(t) == set(STANDARD_RESIDUES)
    assert min(t, key=t.get) == "GLY" and max(t, key=t.get) == "TRP"


def test_defaults_exist_for_the_2a_2b_radii_and_nowhere_else():
    for r in (1.4, 2.5, 4.0, 6.0):
        assert get_reference(r).probe_radius == r
    with pytest.raises(ValueError, match="Do NOT reuse"):
        get_reference(3.0)
    with pytest.raises(ValueError, match="Do NOT reuse"):
        get_reference(1.41)  # radii are matched exactly, never "close enough"


def test_literature_table_is_registered_by_name_but_is_not_the_default():
    lit = get_reference(1.4, "tien2013_theoretical")
    assert lit is TIEN2013_THEORETICAL_1P4
    assert get_reference(1.4).name != "tien2013_theoretical"


def test_empirical_reference_percentile_and_min_count():
    structures = [make_fake_protein(n_res=60, seed=s) for s in range(6)]
    ref = build_empirical_reference(structures, 4.0, percentile=100.0, min_count=5)
    # percentile=100 -> the max observed per residue type; verify against a direct recomputation
    obs = {}
    for s in structures:
        tot, _ = residue_sasa(s, 4.0)
        for i in np.flatnonzero(s.resolved_mask):
            obs.setdefault(str(s.residue_names[i]), []).append(tot[i])
    for k, v in ref.max_asa.items():
        assert v == pytest.approx(max(obs[k]))
    assert ref.probe_radius == 4.0 and ref.radii_table == "bondi"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        big = build_empirical_reference(structures, 4.0, min_count=10**6)
    assert big.max_asa == {} and w


def test_empirical_reference_roundtrips_through_json_and_registry(tmp_path):
    structures = [make_fake_protein(n_res=60, seed=s) for s in range(4)]
    ref = build_empirical_reference(structures, 2.5, min_count=5, name="emp_test")
    ref.to_json(tmp_path / "ref.json")
    loaded = ReferenceTable.from_json(tmp_path / "ref.json")
    assert dict(loaded.max_asa) == dict(ref.max_asa) and loaded.probe_radius == 2.5
    register_reference(loaded)
    assert get_reference(2.5, "emp_test").name == "emp_test"


# ---------------------------------------------------------------- bundled Gly-X-Gly tables
BUNDLED_RADII = (1.4, 2.5, 4.0, 6.0)


def test_bundled_tables_are_complete_and_consistent():
    for r in BUNDLED_RADII:
        t = get_reference(r)
        assert t.probe_radius == r and set(t.max_asa) == set(STANDARD_RESIDUES)
        assert "GLY" not in t.sidechain_max_asa and set(t.sidechain_max_asa) == set(STANDARD_RESIDUES) - {"GLY"}
        assert all(t.sidechain_max_asa[k] <= t.max_asa[k] for k in t.sidechain_max_asa)
        assert t.radii_table == "bondi" and t.extra["rotamers"]["source"] == "staggered_chi_grid_v1"
        assert min(t.max_asa, key=t.max_asa.get) == "GLY" and max(t.max_asa, key=t.max_asa.get) == "TRP"
    tabs = [get_reference(r).max_asa for r in BUNDLED_RADII]
    assert all(tabs[0][k] < tabs[1][k] < tabs[2][k] < tabs[3][k] for k in STANDARD_RESIDUES)


def test_only_one_nonfinite_conformer_source_is_recorded_and_it_is_met():
    per = get_reference(1.4).extra["per_residue"]
    assert {k for k, v in per.items() if v["n_nonfinite_rejected"]} == {"MET"}


def test_bundled_1p4_table_reproduces_the_published_values_approximately():
    """Not exact by design: Bondi radii vs DSSP radii, exhaustive vs subsampled rotamers, staggered chi grid,
    PeptideBuilder geometry, steric filter. Observed at build time: -5.2% .. +0.6%, mean |diff| 2.7%."""
    ours, paper = get_reference(1.4).max_asa, TIEN2013_THEORETICAL_1P4.max_asa
    rel = {k: (ours[k] - paper[k]) / paper[k] for k in paper}
    assert max(abs(v) for v in rel.values()) < 0.08, rel
    a = np.array([ours[k] for k in sorted(paper)]); b = np.array([paper[k] for k in sorted(paper)])
    assert np.corrcoef(a, b)[0, 1] > 0.995


def test_duplicate_bundled_radius_is_an_error_not_a_silent_choice(tmp_path):
    from accessfold.accessibility.reference import load_bundled_references
    ReferenceTable("a", 7.5, {"ALA": 1.0}, "x").to_json(tmp_path / "a.json")
    ReferenceTable("b", 7.5, {"ALA": 2.0}, "x").to_json(tmp_path / "b.json")
    with pytest.raises(ValueError, match="two bundled"):
        load_bundled_references(tmp_path)
