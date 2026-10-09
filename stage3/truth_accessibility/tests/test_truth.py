"""mmCIF chain loader and truth-accessibility pipeline, on a synthetic mmCIF built from real PeptideBuilder geometry."""
import gzip
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
PB = pytest.importorskip("PeptideBuilder")

from accessfold import AtomicStructure, compute_accessibility
from accessfold.structures.mmcif import gap_distance, load_chain_from_mmcif
from accessfold.truth import truth_arrays

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"
HEADER = """data_TEST
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.pdbx_PDB_model_num
"""


def model_atoms():
    """[(seq_id, resname, atom, element, xyz)] for a real-geometry helix-like peptide."""
    from Bio.PDB import PDBIO  # noqa: F401
    s = PB.make_structure(SEQ, [-60.0] * (len(SEQ) - 1), [-45.0] * (len(SEQ) - 1))
    atoms = []
    for res in s[0]["A"]:
        for a in res:
            if a.element != "H":
                atoms.append((res.id[1], res.get_resname(), a.get_name(), a.element.upper(), np.array(a.coord, float)))
    return atoms


def line(group, el, name, alt, comp, asym, seq, xyz, occ=1.0, mdl=1):
    return f"{group} {el} {name} {alt} {comp} {asym} {seq} {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} {occ} {mdl}\n"


@pytest.fixture(scope="module")
def cif_and_truth(tmp_path_factory):
    atoms = model_atoms()
    gap = {8, 9}                       # internal gap (positions 8, 9 unobserved)
    zero_occ = 20                      # coordinates present, occupancy 0 (placed but unobserved)
    sep_pos = 13                       # Ser modelled as phosphoserine: extra P / O atoms
    partial = 5                        # side chain beyond CB missing
    out, kept = [], []
    for seq_id, comp, name, el, xyz in atoms:
        if seq_id in gap:
            continue
        if seq_id == partial and name not in ("N", "CA", "C", "O", "CB"):
            continue
        if seq_id == zero_occ:
            out.append(line("ATOM", el, name, ".", comp, "A", seq_id, xyz, 0.0))
            continue
        if seq_id == sep_pos:
            comp = "SEP"
        if seq_id == 1:                # Met modelled as selenomethionine HETATM
            comp2, name2, el2, grp = "MSE", ("SE" if name == "SD" else name), ("SE" if name == "SD" else el), "HETATM"
        else:
            comp2, name2, el2, grp = comp, name, el, "ATOM"
        if seq_id == 3 and name == "CB":  # altloc: A (0.6, the chosen one) and B (0.4, displaced)
            out.append(line(grp, el2, name2, "A", comp2, "A", seq_id, xyz, 0.6))
            out.append(line(grp, el2, name2, "B", comp2, "A", seq_id, xyz + 0.9, 0.4))
        else:
            out.append(line(grp, el2, name2, ".", comp2, "A", seq_id, xyz))
        kept.append((seq_id, comp, name2, el2, np.round(xyz, 3)))
        if seq_id == sep_pos and name == "OG":      # phosphate group on the serine oxygen
            cb = [x[4] for x in atoms if x[0] == seq_id and x[2] == "CB"][0]
            u = (xyz - cb) / np.linalg.norm(xyz - cb)
            pxyz = xyz + 1.6 * u
            others = [("O1P", np.array([1.0, 0, 0])), ("O2P", np.array([0, 1.0, 0])), ("O3P", np.array([0, 0, 1.0]))]
            for nm, e in [("P", None)] + others:
                c = pxyz if e is None else pxyz + 1.5 * (u + e) / np.linalg.norm(u + e)
                el_ = "P" if e is None else "O"
                out.append(line("HETATM", el_, nm, ".", "SEP", "A", seq_id, c))
                kept.append((seq_id, "SEP", nm, el_, np.round(c, 3)))
    out.append(line("ATOM", "H", "HA", ".", "ALA", "A", 4, np.array([0.1, 0.2, 0.3])))            # hydrogen: dropped
    out.append(line("ATOM", "C", "CB", ".", "ALA", "A", 4, np.array([0.1, 0.2, 0.3]), 0.0))       # occupancy 0: dropped
    out.append(line("HETATM", "O", "O", ".", "HOH", "C", ".", np.array([1.0, 1.0, 1.0])))         # water
    out.append(line("HETATM", "ZN", "ZN", ".", "ZN", "D", ".", np.array([2.0, 2.0, 2.0])))        # ion
    for seq_id, comp, name, el, xyz in atoms[:30]:                                                  # second chain, same ids
        out.append(line("ATOM", el, name, ".", comp, "B", seq_id, xyz + 50.0))
    d = tmp_path_factory.mktemp("cif")
    path = d / "TEST.cif.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(HEADER + "".join(out))
    return path, kept


def test_loader_rules(cif_and_truth):
    path, kept = cif_and_truth
    s, complete, qc = load_chain_from_mmcif(path, "A", SEQ)
    n = len(SEQ)
    assert s.n_residues == n and "".join(s.residue_names[:3]) == "METLYSTHR"  # names come from the sequence
    resolved = s.resolved_mask
    assert not resolved[7] and not resolved[8] and not resolved[19] and resolved.sum() == n - 3   # gap 8-9, occupancy-0 residue 20
    assert qc["longest_internal_gap"] == 2 and qc["n_resolved"] == n - 3
    # coverage: coordinate records exist for the occupancy-0 residue but no usable atoms; reported and warned about
    assert qc["n_with_coordinate_records"] == n - 2 and qc["n_record_but_unusable"] == 1
    assert any("occupancy 0" in w for w in qc["warnings"])
    assert not complete[4] and complete[0] and complete[5]       # residue 5 lost its side chain
    # waters, ions, chain B, hydrogens, occupancy-0 atoms are gone; MSE kept and not a name mismatch
    assert s.n_atoms == len(kept) and qc["n_selenomethionine"] == 1       # MSE is the one accepted exception
    assert qc["n_modified_residues"] == 1 and qc["n_name_mismatch_vs_sequence"] == 1   # SEP is NOT relabelled as Ser
    m = qc["modified_residues"][0]
    assert (m["position"], m["comp_id"], m["kind"], m["sequence_residue"]) == (13, "SEP", "modified", "SER")
    assert m["extra_atoms"] == ["O1P", "O2P", "O3P", "P"] and any("13:SEP" in w for w in qc["warnings"])
    assert "SE" in set(s.elements.tolist()) and qc["n_residues_with_altloc"] == 1 and qc["n_altloc_atoms_dropped"] == 1
    # the 0.6 occupancy altloc was kept, not the displaced 0.4 one
    cb3 = s.coords[(s.residue_index == 2) & (s.atom_names == "CB")][0]
    ref = [x for x in kept if x[0] == 3 and x[2] == "CB"][0][4]
    assert np.allclose(cb3, ref)


def test_loader_equals_direct_structure_and_masks(cif_and_truth):
    path, kept = cif_and_truth
    s, complete, _ = load_chain_from_mmcif(path, "A", SEQ)
    direct = AtomicStructure.from_atoms(np.array([k[4] for k in kept]), [k[3] for k in kept], [k[2] for k in kept],
                                        [k[0] - 1 for k in kept], s.residue_names)
    _, _, qc = load_chain_from_mmcif(path, "A", SEQ)
    a = truth_arrays(s, complete, (1.4, 6.0), modified=qc["modified_residues"])
    b = compute_accessibility(direct, "relative_sasa", 1.4)
    ok = a["mask"]
    assert not ok[4] and not ok[7] and not ok[8] and not ok[19] and ok[0]  # incomplete, unresolved masked; MSE kept
    assert a["modified"][12] and not ok[12]                                # phosphoserine masked, its atoms kept
    assert np.isnan(a["rel_1.4"][12]) and s.n_atoms == direct.n_atoms
    assert np.allclose(a["abs_1.4"][ok], b.absolute[ok]) and np.allclose(a["rel_1.4"][ok], b.relative[ok])
    assert np.isnan(a["rel_6"][~ok]).all() and np.isfinite(a["rel_6"][ok]).all()
    # residue 5's atoms still occlude: its neighbours keep their values (they are in the mask)
    assert ok[3] and (ok[5] or a["shadow"][5])   # residue 6 may be shadowed by the phosphoserine 7 positions later


def test_shadow_mask_matches_independent_recomputation(cif_and_truth):
    path, _ = cif_and_truth
    s, complete, qc = load_chain_from_mmcif(path, "A", SEQ)
    a = truth_arrays(s, complete, (1.4, 6.0), modified=qc["modified_residues"], shadow_threshold=5.0)
    drop = (s.residue_index == 12) & np.isin(s.atom_names, ["P", "O1P", "O2P", "O3P"])
    plain = AtomicStructure(s.coords[~drop], s.elements[~drop], s.atom_names[~drop], s.residue_index[~drop], s.residue_names)
    expect = np.zeros(s.n_residues, bool)
    for r in (1.4, 6.0):
        full = compute_accessibility(s, "absolute_sasa", r).absolute
        stripped = compute_accessibility(plain, "absolute_sasa", r).absolute
        expect |= np.nan_to_num(np.abs(full - stripped), nan=0.0) > 5.0
    expect[12] = False
    assert np.array_equal(a["shadow"], expect) and a["shadow"].sum() >= 1, np.flatnonzero(a["shadow"])
    assert not a["mask"][a["shadow"]].any()
    # a huge threshold shadows nobody, a zero threshold shadows every residue the phosphate touches at all
    assert not truth_arrays(s, complete, (1.4,), modified=qc["modified_residues"], shadow_threshold=1e6)["shadow"].any()
    assert a["gap_distance"][7] == 0 and a["gap_distance"][6] == 1 and a["gap_distance"][9] == 1


def test_gap_distance_ignores_terminal_truncation():
    r = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=bool)
    d = gap_distance(r)
    assert d[4] == 0 and d[3] == 1 and d[5] == 1 and d[2] == 2 and d[6] == 2 and d[0] == 4 and d[7] == 3   # distance to the one internal gap (position 4)
    assert gap_distance(np.array([0, 1, 1, 0], dtype=bool)).tolist() == [9999] * 4


def test_wrong_chain_and_wrong_sequence_raise(cif_and_truth):
    path, _ = cif_and_truth
    with pytest.raises(ValueError, match="no polymer atoms"):
        load_chain_from_mmcif(path, "Z", SEQ)
    with pytest.raises(ValueError, match="does not fit"):
        load_chain_from_mmcif(path, "A", SEQ[:10])


def test_cli_end_to_end_offline(cif_and_truth, tmp_path):
    path, _ = cif_and_truth
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "TEST.cif.gz").write_bytes(path.read_bytes())
    csv_path = tmp_path / "ds.csv"
    header = "pdb_id,chain_id,label_chain_id,sequence,length,resolution,resolved_fraction,training_or_test_split,release_date\n"
    csv_path.write_text(header + f"TEST,A,A,{SEQ},{len(SEQ)},1.5,0.9,development,2000-01-01\n"
                                 f"NOPE,A,A,{SEQ},{len(SEQ)},1.5,0.9,development,2000-01-01\n")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "compute_truth_accessibility.py"
    r = subprocess.run([sys.executable, str(script), "--csv", str(csv_path), "--cache-dir", str(cache), "--out", str(out),
                        "--offline", "--workers", "2", "--radii", "1.4", "6.0"], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr          # NOPE has no cached file -> reported, batch continues
    assert "NOPE_A" in (out / "failures.csv").read_text() and "offline" in (out / "failures.csv").read_text()
    z = np.load(out / "TEST_A.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["split"] == "development" and meta["qc"]["n_resolved"] == len(SEQ) - 3
    assert z["rel_1.4"].shape == (len(SEQ),) and np.isnan(z["rel_1.4"][7]) and np.isfinite(z["rel_6"][0])
    summ = (out / "summary.csv").read_text().splitlines()
    assert len(summ) == 2 and "mean_rsa_1.4" in summ[0] and "13:SEP" in summ[1] and "n_shadow_masked" in summ[0]
    with gzip.open(out / "residues.csv.gz", "rt") as fh:
        exported = list(csv.DictReader(fh))
    assert len(exported) == len(SEQ)
    for prefix, decimals in (("abs", 3), ("rel", 4), ("sc_abs", 3), ("sc_rel", 4)):
        for radius in (1.4, 6.0):
            key = f"{prefix}_{radius:g}"
            for i, row in enumerate(exported):
                value = z[key][i]
                assert row[key] == ("" if np.isnan(value) else f"{value:.{decimals}f}")
    # resumable: a second run recomputes nothing and gives the same bytes in the result
    before = (out / "TEST_A.npz").stat().st_mtime_ns
    subprocess.run([sys.executable, str(script), "--csv", str(csv_path), "--cache-dir", str(cache), "--out", str(out),
                    "--offline", "--workers", "1", "--radii", "1.4", "6.0"], capture_output=True, text=True)
    assert (out / "TEST_A.npz").stat().st_mtime_ns == before
