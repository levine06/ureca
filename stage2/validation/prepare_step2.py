"""Reconstruct stored reference winners and verify normalization arithmetic."""
import csv
import json
from pathlib import Path
import numpy as np
from biotite.structure.io.pdb import PDBFile
from accessfold import AtomicStructure, compute_accessibility, get_reference
from accessfold.accessibility import tripeptide as T
from accessfold.accessibility.reference import STANDARD_RESIDUES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reference_conformers"
OUT.mkdir(exist_ok=True)
records = []
for probe in (1.4, 2.5, 4.0, 6.0):
    table = get_reference(probe)
    assert set(table.max_asa) == set(STANDARD_RESIDUES)
    assert all(np.isfinite(v) and v > 0 for v in table.max_asa.values())
    for name in STANDARD_RESIDUES:
        winner = table.extra["per_residue"][name]["argmax"]
        base, names, elems, indices = T._structure_arrays(T._build_conformer_structure(name, "G", -120, 140))
        topo = T._make_topology(name, base, names, elems, indices, T.TripeptideConfig())
        coords = T._structure_arrays(T._build_conformer_structure(name, "G", winner["phi"], winner["psi"]))[0]
        coords = T._apply_chi(coords[None], topo, np.asarray(winner["chi"], dtype=float)[None])[0]
        assert T._clash_free(coords[None], topo)[0], (name, probe)
        structure = AtomicStructure(coords, elems, names, indices, ["GLY", name, "GLY"])
        result = compute_accessibility(structure, probe_radius=probe, n_points=10000)
        assert np.allclose(result.relative, result.absolute / table.lookup(structure.residue_names))
        error = abs(result.absolute[1] / table.max_asa[name] - 1)
        assert error < 0.005, (name, probe, error)
        path = OUT / f"{name}_{probe:g}.pdb"
        # PDB output is for independent PyMOL measurement, with explicit residue IDs.
        from biotite.structure import AtomArray
        arr = AtomArray(len(coords))
        arr.coord = coords
        arr.atom_name = names
        arr.element = elems
        arr.chain_id[:] = "A"
        arr.res_id = indices + 1
        arr.res_name = np.asarray(["GLY", name, "GLY"])[indices]
        pdb = PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(path))
        records.append({"name": name, "probe": probe, "file": path.relative_to(ROOT).as_posix(),
                        "stored_max_A2": table.max_asa[name], "recomputed_max_A2": float(result.absolute[1]),
                        "relative_at_winner": float(result.relative[1])})
(ROOT / "reference_winners.json").write_text(json.dumps(records, indent=2))
print(f"Verified {len(records)} stored total-residue maxima and normalization identities", flush=True)

targets = {"1CRN": (16, 30), "1LYZ": (25, 108, 98)}
rows = []
for pdb_id, residues in targets.items():
    from accessfold.structures.adapters import from_biotite
    arr = PDBFile.read(str(ROOT / f"{pdb_id}_clean.pdb")).get_structure(model=1)
    structure = from_biotite(arr, "A")
    for probe in (0.0, 0.5, 1.0, 1.4):
        result = compute_accessibility(structure, method="absolute_sasa", probe_radius=probe, n_points=4000)
        for residue in residues:
            rows.append({"pdb_id":pdb_id,"residue":residue,"probe":probe,"accessfold_A2":float(result.absolute[residue-1])})
with (ROOT / "step2_accessfold.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Prepared 20 residue/probe measurements for PyMOL comparison", flush=True)
