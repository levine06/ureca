"""Export observations alongside scores without modifying labels or test criteria."""
import csv
from pathlib import Path
from biotite.structure.io.pdb import PDBFile
from accessfold import compute_accessibility
from accessfold.structures.adapters import from_biotite

ROOT = Path(__file__).resolve().parent
with (ROOT / "manual_labels.csv").open() as handle:
    rows = list(csv.DictReader(handle))
scores = {}
for pdb_id in ("1UBQ", "1CRN", "1LYZ"):
    arr = PDBFile.read(str(ROOT / f"{pdb_id}_clean.pdb")).get_structure(model=1)
    scores[pdb_id] = compute_accessibility(from_biotite(arr, "A"), n_points=4000)
for row in rows:
    result = scores[row["pdb_id"]]
    index = int(row["residue"]) - 1
    row["absolute_sasa_A2"] = float(result.absolute[index])
    row["relative_sasa"] = float(result.relative[index])
with (ROOT / "manual_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Saved manual_comparison.csv; original labels unchanged")
