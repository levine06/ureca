"""Physically checkable expectations on PeptideBuilder-built model peptides (no experimental structure needed).

These stand in for the 'manually inspected buried/exposed residues' tests until real structures from the
Stage 1 dataset are available (drop them into tests/data and extend this file).
"""
import numpy as np
import pytest

PB = pytest.importorskip("PeptideBuilder")

from accessfold import AtomicStructure, compute_accessibility
from accessfold.accessibility.tripeptide import THREE_TO_ONE


def build_peptide(seq, phi, psi):
    from PeptideBuilder import Geometry as G, PeptideBuilder as PBmod
    geos = []
    for aa in seq:
        g = G.geometry(aa)
        g.phi, g.psi_im1 = phi, psi
        geos.append(g)
    st = PBmod.make_structure_from_geos(geos)
    coords, names, elems, res, resnames = [], [], [], [], []
    for i, residue in enumerate(st[0]["A"]):
        resnames.append(residue.get_resname())
        for a in residue:
            coords.append(a.get_coord()); names.append(a.get_name()); elems.append(a.element); res.append(i)
    return AtomicStructure(np.array(coords), elems, names, res, resnames)


HELIX = build_peptide("A" * 24, -57.0, -47.0)
STRAND = build_peptide("A" * 24, -120.0, 130.0)


def test_helix_termini_are_more_exposed_than_the_middle():
    rel = compute_accessibility(HELIX, "relative_sasa", 1.4).relative
    assert rel[:3].mean() > rel[10:14].mean() and rel[-3:].mean() > rel[10:14].mean()


def test_extended_strand_is_more_exposed_than_helix_in_the_middle():
    h = compute_accessibility(HELIX, "relative_sasa", 1.4).relative[8:16].mean()
    s = compute_accessibility(STRAND, "relative_sasa", 1.4).relative[8:16].mean()
    assert s > h


@pytest.mark.parametrize("probe", [1.4, 2.5, 4.0, 6.0])
def test_relative_values_are_in_a_sane_range_and_finite_at_every_radius(probe):
    rel = compute_accessibility(HELIX, "relative_sasa", probe).relative
    assert np.isfinite(rel).all() and rel.min() >= 0.0 and rel.max() < 1.5


def test_a_large_probe_sees_less_of_a_helix_middle_than_of_its_ends():
    """Coarser probes should still separate ends from the middle (the sweep should not destroy the signal)."""
    for probe in (2.5, 4.0):
        rel = compute_accessibility(HELIX, "relative_sasa", probe).relative
        assert rel[:3].mean() > rel[10:14].mean()


def test_isolated_extended_tripeptide_is_close_to_its_own_reference_maximum():
    tri = build_peptide("GAG", -120.0, 140.0)
    rel = compute_accessibility(tri, "relative_sasa", 1.4).relative[1]
    assert 0.6 < rel <= 1.02  # extended Gly-Ala-Gly is nearly the definition of 'fully exposed'


@pytest.mark.parametrize("probe", [1.4, 2.5, 4.0, 6.0])
def test_agrees_with_biopython_shrake_rupley_on_a_realistic_peptide_at_every_radius(probe):
    """Independent implementation (Bio.PDB.SASA) with the same Bondi-style element radii."""
    from Bio.PDB.SASA import ShrakeRupley
    from PeptideBuilder import Geometry as G, PeptideBuilder as PBmod

    seq = "ACDEFGHIKLMNPQRSTVWY"
    geos = []
    for aa in seq:
        g = G.geometry(aa)
        g.phi, g.psi_im1 = -57.0, -47.0
        geos.append(g)
    st = PBmod.make_structure_from_geos(geos)
    ShrakeRupley(probe_radius=probe, n_points=2000).compute(st, level="R")
    theirs = np.array([res.sasa for res in st[0]["A"]])

    coords, names, elems, res_idx, resnames = [], [], [], [], []
    for i, residue in enumerate(st[0]["A"]):
        resnames.append(residue.get_resname())
        for a in residue:
            coords.append(a.get_coord()); names.append(a.get_name()); elems.append(a.element); res_idx.append(i)
    ours = compute_accessibility(AtomicStructure(np.array(coords), elems, names, res_idx, resnames),
                                 "absolute_sasa", probe, n_points=2000).absolute
    assert ours.sum() == pytest.approx(theirs.sum(), rel=0.01)
    assert np.corrcoef(ours, theirs)[0, 1] > 0.995
    assert np.abs(ours - theirs).max() < 0.05 * np.abs(theirs).max() + 3.0


@pytest.mark.parametrize("probe", [1.4, 2.5, 4.0, 6.0])
def test_agrees_with_freesasa_lee_richards_at_every_radius(probe):
    """FreeSASA's Lee-Richards (1000 slices) integrates the area along slices instead of sampling points,
    so it checks absolute areas, not just agreement between two point-sampling codes.
    Measured at build time (1000 points): total error +0.06 / -0.24 / +0.07 / +0.16 % at 1.4 / 2.5 / 4.0 / 6.0 A.
    Install: pip install freesasa --use-pep517   (the legacy setup.py path fails on Debian's patched setuptools)."""
    freesasa = pytest.importorskip("freesasa")
    from PeptideBuilder import Geometry as G, PeptideBuilder as PBmod
    from accessfold.accessibility.radii import radii_from_elements
    from accessfold.accessibility.sasa import atom_sasa

    geos = []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        g = G.geometry(aa)
        g.phi, g.psi_im1 = -57.0, -47.0
        geos.append(g)
    st = PBmod.make_structure_from_geos(geos)
    coords, elems, fs = [], [], freesasa.Structure()
    for i, residue in enumerate(st[0]["A"]):
        for a in residue:
            x = a.get_coord()
            coords.append(x)
            elems.append(a.element)
            fs.addAtom(f" {a.get_name():<3}", residue.get_resname(), str(i + 1), "A", float(x[0]), float(x[1]), float(x[2]))
    coords = np.array(coords)
    radii = radii_from_elements(np.array(elems))
    fs.setRadii(radii.tolist())  # identical radii on both sides: only the algorithm differs
    res = freesasa.calc(fs, freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": probe,
                                                 "n-slices": 1000}))
    exact = np.array([res.atomArea(i) for i in range(len(coords))])
    ours = atom_sasa(coords, radii, probe, 1000)
    assert abs(ours.sum() - exact.sum()) / exact.sum() < 0.005
    assert np.abs(ours - exact).max() < 4.0  # Angstrom^2 per atom (measured max 2.5)
