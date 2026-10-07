import numpy as np
import pytest

from accessfold import (AtomicStructure, compute_accessibility, compute_accessibility_profile, ReferenceTable,
                        register_method)
from accessfold.accessibility.compute import AccessibilityResult, _flank_mask
from conftest import make_fake_protein


def test_shapes_and_masks_with_unresolved_residues():
    s = make_fake_protein(n_res=30, drop=(10, 11, 12))
    r = compute_accessibility(s, "absolute_sasa", 1.4)
    assert r.absolute.shape == (30,) and r.mask.shape == (30,)
    assert np.all(np.isnan(r.y[[10, 11, 12]]))
    assert not r.mask[[10, 11, 12]].any() and r.mask[[0, 20]].all()
    assert np.all(r.y[r.mask] > 0)


def test_residue_sums_equal_atom_sums():
    from accessfold.accessibility.radii import radii_from_elements
    from accessfold.accessibility.sasa import atom_sasa
    s = make_fake_protein()
    r = compute_accessibility(s, "absolute_sasa", 1.4)
    per_atom = atom_sasa(s.coords, radii_from_elements(s.elements), 1.4)
    assert np.nansum(r.absolute) == pytest.approx(per_atom.sum())


def test_glycine_has_no_sidechain_value_and_others_do():
    s = make_fake_protein(n_res=40)
    r = compute_accessibility(s, "absolute_sasa", 1.4)
    gly = s.residue_names == "GLY"
    assert gly.any() and (~gly).any()
    assert np.all(np.isnan(r.sidechain_absolute[gly]))
    assert np.all(np.isfinite(r.sidechain_absolute[~gly]))
    assert np.all(r.sidechain_absolute[~gly] <= r.absolute[~gly] + 1e-9)


def test_relative_equals_absolute_over_table_value():
    s = make_fake_protein()
    r = compute_accessibility(s, "relative_sasa", 1.4)
    from accessfold import get_reference
    ref = get_reference(1.4).max_asa  # whichever table is the registered default at 1.4 A
    expected = r.absolute / np.array([ref[n] for n in s.residue_names])
    assert np.allclose(r.relative, expected)
    assert r.primary == "relative" and np.allclose(r.y, r.values("relative"), equal_nan=True)


def test_relative_without_reference_at_other_radius_raises_but_absolute_works():
    s = make_fake_protein()
    with pytest.raises(ValueError, match="3.0"):
        compute_accessibility(s, "relative_sasa", 3.0)  # no table registered for 3.0 A
    r = compute_accessibility(s, "absolute_sasa", 3.0)
    assert np.all(np.isnan(r.relative)) and np.isfinite(r.y[r.mask]).all()


def test_reference_probe_mismatch_is_rejected():
    s = make_fake_protein()
    ref = ReferenceTable("x", 1.4, {"ALA": 100.0}, "test")
    with pytest.raises(ValueError, match="probe"):
        compute_accessibility(s, "relative_sasa", 4.0, reference=ref)


def test_unreferenced_residue_types_are_reported_not_silently_dropped():
    s = make_fake_protein()
    ref = ReferenceTable("partial", 1.4, {"ALA": 129.0}, "test")
    r = compute_accessibility(s, "relative_sasa", 1.4, reference=ref)
    assert set(r.metadata["unreferenced_residues"]) == {"GLY", "LEU", "SER"} & set(s.residue_names.tolist())
    assert np.all(np.isnan(r.relative[s.residue_names != "ALA"]))


def test_clip_caps_relative_values():
    s = make_fake_protein()
    ref = ReferenceTable("tiny", 1.4, {k: 1.0 for k in ("ALA", "GLY", "LEU", "SER")}, "test")
    r = compute_accessibility(s, "relative_sasa", 1.4, reference=ref, clip=1.0)
    assert np.nanmax(r.relative) == 1.0


def test_gap_flank_masks_neighbours_of_internal_gaps_only():
    s = make_fake_protein(n_res=20, drop=(0, 9, 10))  # residue 0 missing = terminal truncation
    r0 = compute_accessibility(s, "absolute_sasa", 1.4)
    r1 = compute_accessibility(s, "absolute_sasa", 1.4, gap_flank=1)
    assert r0.mask[8] and r0.mask[11]
    assert not r1.mask[8] and not r1.mask[11]
    assert r1.mask[1] and r1.mask[19]  # terminal truncation does not trigger flank masking


def test_flank_mask_no_wraparound():
    resolved = np.array([1, 1, 1, 1, 0, 1], bool)  # gap at 4; must not affect position 0 via roll
    m = _flank_mask(resolved, 1)
    assert m.tolist() == [True, True, True, False, False, False]


def test_with_coords_keeps_topology_and_changes_values():
    s = make_fake_protein()
    moved = s.with_coords(s.coords * 3.0)  # spread out: everything becomes more exposed
    a = compute_accessibility(s, "absolute_sasa", 1.4).absolute
    b = compute_accessibility(moved, "absolute_sasa", 1.4).absolute
    assert np.nansum(b) > np.nansum(a)
    with pytest.raises(ValueError):
        s.with_coords(s.coords[:-1])


def test_profile_returns_one_result_per_radius():
    s = make_fake_protein()
    prof = compute_accessibility_profile(s, (1.4, 2.5, 4.0, 6.0), method="absolute_sasa")
    assert list(prof) == [1.4, 2.5, 4.0, 6.0]
    assert all(isinstance(v, AccessibilityResult) for v in prof.values())


def test_unknown_method_and_registry_extension():
    s = make_fake_protein()
    with pytest.raises(ValueError, match="unknown method"):
        compute_accessibility(s, "nope")

    @register_method("constant_test")
    def _const(structure, *, probe_radius=1.4, **kw):
        n = structure.n_residues
        return AccessibilityResult(np.ones(n), np.ones(n), np.ones(n), np.ones(n), structure.resolved_mask,
                                   structure.residue_names, "constant_test", probe_radius, "absolute")
    assert np.all(compute_accessibility(s, "constant_test").y[s.resolved_mask] == 1.0)


def test_structure_validation():
    with pytest.raises(ValueError):
        AtomicStructure(np.zeros((2, 3)), ["C", "C"], ["CA", "CB"], [0, 5], ["ALA"])  # residue_index out of range
    with pytest.raises(ValueError):
        AtomicStructure(np.array([[np.nan, 0, 0]]), ["C"], ["CA"], [0], ["ALA"])
