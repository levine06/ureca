"""Independent numerical check while PyMOL runtime access is unavailable."""
import csv
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from biotite.structure.io.pdb import PDBFile
from accessfold import compute_accessibility
from accessfold.structures.adapters import from_biotite
ROOT = Path(__file__).resolve().parent
rows = []
for pdb_id in ("1UBQ", "1CRN", "1LYZ"):
    pdb = PDBFile.read(str(ROOT / f"{pdb_id}.pdb"))
    arr = pdb.get_structure(model=1, altloc="first")
    arr = arr[(arr.chain_id == "A") & ~arr.hetero & (arr.element != "H")]
    clean = PDBFile()
    clean.set_structure(arr)
    clean.write(str(ROOT / f"{pdb_id}_clean.pdb"))
    structure = from_biotite(arr, "A")
    bio = PDBParser(QUIET=True).get_structure(pdb_id, str(ROOT / f"{pdb_id}_clean.pdb"))
    for probe in (1.4, 2.5, 4.0, 6.0):
        ShrakeRupley(probe_radius=probe, n_points=1000,
                     radii_dict={"C":1.70,"N":1.55,"O":1.52,"S":1.80}).compute(bio, level="R")
        theirs = np.array([r.sasa for r in bio[0]["A"]])
        result = compute_accessibility(structure, probe_radius=probe, n_points=1000)
        ours = result.absolute
        error = abs(ours.sum()-theirs.sum())/theirs.sum()
        corr = np.corrcoef(ours,theirs)[0,1]
        maximum = np.max(np.abs(ours-theirs))
        assert np.isfinite(result.y).all()
        assert error < 0.02 and corr > 0.99 and maximum < 8
        rows.append({"pdb_id":pdb_id,"probe":probe,"residues":len(ours),
                     "total_error_percent":100*error,"correlation":corr,"max_residue_error_A2":maximum})
        print(f"{pdb_id} probe={probe}: passed", flush=True)
with (ROOT / "biopython_checks.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
