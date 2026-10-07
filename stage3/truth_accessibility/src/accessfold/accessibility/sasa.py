"""Numerical solvent-accessible surface area (Shrake & Rupley 1973, J Mol Biol 79:351).

Pure NumPy/SciPy on plain arrays, so the same call works on truth, predicted candidates and
intermediate diffusion coordinates. This is the *reference* (non-differentiable) implementation
that a Stage 5 PyTorch surrogate is validated against.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Default sample points per atom sphere. Shrake-Rupley with a fixed point lattice is not exactly
#: rotation invariant; ~1000 points keeps that discretisation noise near the percent level (measured in
#: tests/test_sasa.py). Truth and candidate structures must be compared at the SAME n_points.
DEFAULT_N_POINTS = 1000

_UNIT_SPHERE_CACHE: dict[int, np.ndarray] = {}


def unit_sphere_points(n_points: int) -> np.ndarray:
    """[P, 3] quasi-uniform points on the unit sphere (golden-spiral / Fibonacci lattice)."""
    cached = _UNIT_SPHERE_CACHE.get(n_points)
    if cached is not None:
        return cached
    i = np.arange(n_points, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
    _UNIT_SPHERE_CACHE[n_points] = pts
    return pts


def atom_sasa(coords: np.ndarray, radii: np.ndarray, probe_radius: float = 1.4,
              n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Per-atom solvent-accessible surface area in Angstrom^2.

    Args:
        coords: [A, 3] atom positions.
        radii: [A] van der Waals radii.
        probe_radius: probe sphere radius (1.4 = water).
        n_points: test points per atom sphere; area resolution is ~ 4*pi*R^2 / n_points.

    Convention: area_i = 4*pi*(r_i + probe)^2 * (fraction of test points on the expanded sphere
    that lie outside every other expanded sphere). All atoms occlude all others.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or radii.shape != (coords.shape[0],):
        raise ValueError("expected coords [A, 3] and radii [A]")
    if probe_radius < 0 or n_points < 1:
        raise ValueError("probe_radius must be >= 0 and n_points >= 1")
    n_atoms = coords.shape[0]
    if n_atoms == 0:
        return np.zeros(0)

    expanded = radii + probe_radius
    unit = unit_sphere_points(n_points)
    tree = cKDTree(coords)
    neighbours = tree.query_ball_point(coords, r=expanded + expanded.max())
    sq_norm_all = np.einsum("ij,ij->i", coords, coords)

    areas = np.empty(n_atoms)
    for i in range(n_atoms):
        idx = np.asarray(neighbours[i], dtype=np.int64)
        idx = idx[idx != i]
        if idx.size:
            d = np.linalg.norm(coords[idx] - coords[i], axis=1)
            idx = idx[d < expanded[i] + expanded[idx]]  # only spheres that actually overlap
        full = 4.0 * np.pi * expanded[i] ** 2
        if idx.size == 0:
            areas[i] = full
            continue
        pts = coords[i] + expanded[i] * unit  # [P, 3]
        # squared distances point-neighbour via |p|^2 - 2 p.c + |c|^2 (BLAS matmul)
        d2 = (np.einsum("ij,ij->i", pts, pts)[:, None] - 2.0 * pts @ coords[idx].T + sq_norm_all[idx][None, :])
        buried = (d2 < expanded[idx][None, :] ** 2).any(axis=1)
        areas[i] = full * (1.0 - buried.mean())
    return areas
