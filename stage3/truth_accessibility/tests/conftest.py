import numpy as np
import pytest

from accessfold import AtomicStructure


def make_blob(n_atoms=400, box=22.0, min_dist=1.3, seed=0):
    """Dense pseudo-random atom cloud with a minimum separation (no real chemistry)."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_atoms:
        p = rng.uniform(0, box, 3)
        if not pts or np.min(np.linalg.norm(np.array(pts) - p, axis=1)) > min_dist:
            pts.append(p)
    return np.array(pts)


def make_fake_protein(n_res=40, seed=1, drop=()):
    """Synthetic 'protein': every residue has N, CA, C, O, CB (ALA-like, GLY gets no CB).

    Only used to exercise indexing/masking/aggregation, not to claim real geometry.
    `drop`: residue indices to leave unresolved (no atoms).
    """
    rng = np.random.default_rng(seed)
    names_pool = np.array(["ALA", "GLY", "LEU", "SER"])
    resnames = names_pool[rng.integers(0, len(names_pool), n_res)]
    coords, elems, atoms, ridx = [], [], [], []
    ca = np.cumsum(rng.normal(0, 1.0, (n_res, 3)) * 2.0, axis=0)
    for i in range(n_res):
        if i in drop:
            continue
        for name, el, off in (("N", "N", (-1.2, 0, 0)), ("CA", "C", (0, 0, 0)), ("C", "C", (1.2, 0.4, 0)),
                              ("O", "O", (1.8, 1.3, 0)), ("CB", "C", (0, -1.2, 0.8))):
            if name == "CB" and resnames[i] == "GLY":
                continue
            coords.append(ca[i] + np.array(off))
            elems.append(el); atoms.append(name); ridx.append(i)
    return AtomicStructure(np.array(coords), elems, atoms, ridx, resnames)


@pytest.fixture
def blob():
    return make_blob()
