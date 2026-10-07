import numpy as np
import pytest

from accessfold.accessibility.sasa import atom_sasa, unit_sphere_points


def rot(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_unit_sphere_points_are_unit_and_balanced():
    p = unit_sphere_points(1000)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)
    assert np.linalg.norm(p.mean(axis=0)) < 1e-2


@pytest.mark.parametrize("probe", [0.0, 1.4, 4.0])
def test_isolated_sphere_matches_analytic_area(probe):
    r = 1.7
    a = atom_sasa(np.zeros((1, 3)), np.array([r]), probe)
    assert a[0] == pytest.approx(4 * np.pi * (r + probe) ** 2, rel=1e-12)


@pytest.mark.parametrize("probe", [1.4, 2.5, 6.0])
@pytest.mark.parametrize("d", [2.5, 3.2, 4.0])
def test_two_overlapping_spheres_match_spherical_cap_formula(probe, d):
    r1, r2 = 1.7, 1.52
    R1, R2 = r1 + probe, r2 + probe
    x1 = (d * d + R1 * R1 - R2 * R2) / (2 * d)  # distance from centre 1 to the intersection plane
    x2 = d - x1
    expected = 4 * np.pi * R1**2 - 2 * np.pi * R1 * (R1 - x1) + 4 * np.pi * R2**2 - 2 * np.pi * R2 * (R2 - x2)
    a = atom_sasa(np.array([[0, 0, 0], [d, 0, 0.0]]), np.array([r1, r2]), probe, n_points=4000)
    assert a.sum() == pytest.approx(expected, rel=0.01)


def test_fully_buried_atom_has_zero_area():
    shell = unit_sphere_points(60) * 2.0  # 60 atoms on a shell around a central atom
    coords = np.vstack([[0, 0, 0], shell])
    radii = np.full(len(coords), 1.7)
    a = atom_sasa(coords, radii, 1.4)
    assert a[0] == 0.0
    assert a[1:].min() > 0


def test_translation_invariance_is_exact(blob):
    radii = np.full(len(blob), 1.7)
    a = atom_sasa(blob, radii, 1.4)
    b = atom_sasa(blob + np.array([13.0, -7.0, 101.0]), radii, 1.4)
    assert np.allclose(a, b, atol=1e-8)


@pytest.mark.parametrize("probe", [1.4, 4.0])
def test_rotation_invariance_within_sampling_error(blob, probe):
    # Numerical Shrake-Rupley is not exactly rotation invariant: the sampling lattice rotates
    # relative to the geometry. Require agreement of the total to ~1%.
    radii = np.full(len(blob), 1.7)
    a = atom_sasa(blob, radii, probe, n_points=1000)
    b = atom_sasa(blob @ rot(3).T, radii, probe, n_points=1000)
    assert abs(a.sum() - b.sum()) / a.sum() < 0.01


def test_permutation_invariance(blob):
    radii = np.linspace(1.5, 1.9, len(blob))
    perm = np.random.default_rng(0).permutation(len(blob))
    a = atom_sasa(blob, radii, 1.4)
    b = atom_sasa(blob[perm], radii[perm], 1.4)
    assert np.allclose(a[perm], b)


def test_inputs_validated():
    with pytest.raises(ValueError):
        atom_sasa(np.zeros((3, 2)), np.ones(3))
    with pytest.raises(ValueError):
        atom_sasa(np.zeros((3, 3)), np.ones(2))
    assert atom_sasa(np.zeros((0, 3)), np.zeros(0)).shape == (0,)


def test_agrees_with_biotite_across_probe_radii(blob):
    struc = pytest.importorskip("biotite.structure")
    radii = np.linspace(1.5, 1.9, len(blob))
    arr = struc.AtomArray(len(blob))
    arr.coord = blob.astype(np.float32)
    arr.element = np.full(len(blob), "C")
    arr.res_name = np.full(len(blob), "ALA")
    arr.hetero[:] = False
    for probe in (1.4, 2.5, 4.0, 6.0):
        theirs = struc.sasa(arr, probe_radius=probe, point_number=1000, vdw_radii=radii, ignore_ions=False)
        ours = atom_sasa(blob, radii, probe, n_points=1000)
        assert ours.sum() == pytest.approx(theirs.sum(), rel=0.01), probe
        assert np.corrcoef(ours, theirs)[0, 1] > 0.99, probe


def _brute_force_atom_sasa(coords, radii, probe, n_points):
    """Dense O(N^2) reference with no neighbour search: catches cutoffs that only hold at 1.4 A."""
    expanded = radii + probe
    unit = unit_sphere_points(n_points)
    out = np.empty(len(coords))
    for i in range(len(coords)):
        pts = coords[i] + expanded[i] * unit
        d = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=-1)
        d[:, i] = np.inf
        buried = (d < expanded[None, :]).any(axis=1)
        out[i] = 4 * np.pi * expanded[i] ** 2 * (1 - buried.mean())
    return out


@pytest.mark.parametrize("probe", [0.0, 1.4, 2.5, 4.0, 6.0, 9.0])
def test_kdtree_neighbour_search_matches_brute_force_at_every_probe_radius(probe):
    coords = __import__("conftest").make_blob(n_atoms=150, box=16.0, seed=4)
    radii = np.linspace(1.4, 1.9, len(coords))  # heterogeneous radii exercise the per-pair cutoff
    fast = atom_sasa(coords, radii, probe, n_points=300)
    slow = _brute_force_atom_sasa(coords, radii, probe, 300)
    assert np.allclose(fast, slow, atol=1e-9)


def test_rotation_noise_shrinks_with_point_count_and_is_small_at_default():
    """Measured discretisation noise: spread of total SASA over random rotations, and error vs a dense reference."""
    from accessfold.accessibility.sasa import DEFAULT_N_POINTS
    coords = __import__("conftest").make_blob(n_atoms=250, box=18.0, seed=2)
    radii = np.full(len(coords), 1.7)
    rots = [rot(s) for s in range(10)]
    for probe in (1.4, 4.0):
        truth = np.mean([atom_sasa(coords @ R.T, radii, probe, 20000).sum() for R in rots[:3]])
        cv, err = {}, {}
        for n in (100, 400, 1600):
            tot = np.array([atom_sasa(coords @ R.T, radii, probe, n).sum() for R in rots])
            cv[n] = tot.std() / tot.mean()
            err[n] = abs(tot.mean() - truth) / truth
        assert cv[1600] < cv[100] and cv[1600] < 0.005, (probe, cv)
        assert err[1600] < 0.005, (probe, err)
    tot = np.array([atom_sasa(coords @ R.T, radii, 1.4, DEFAULT_N_POINTS).sum() for R in rots])
    assert tot.std() / tot.mean() < 0.005  # default resolution: total SASA repeatable to < 0.5% across orientations
