"""Run with the installed PyMOL Python; export SASA and neutral inspection sessions."""
import csv
import json
import os
from pathlib import Path
import pymol

ROOT = Path(__file__).resolve().parent
GUI = os.environ.get("ACCESSFOLD_GUI") == "1"
if not GUI:
    pymol.finish_launching(["pymol", "-cq"])
from pymol import cmd

rows = []
for record in json.loads((ROOT / "manifest.json").read_text())["structures"]:
    pdb_id = record["pdb_id"]
    cmd.reinitialize()
    cmd.load(str(ROOT / f"{pdb_id}.pdb"), "source")
    cmd.create("protein", "source and chain A and polymer.protein and not hydro and (alt ''+A)", 1, 1)
    cmd.delete("source")
    cmd.alter("protein", "alt=''")
    # Explicit element radii match AccessFold's Bondi model.
    cmd.alter("protein", "vdw=radii[elem.upper()]", space={"radii": {"C":1.70,"N":1.55,"O":1.52,"S":1.80}})
    cmd.sort()
    cmd.save(str(ROOT / f"{pdb_id}_clean.pdb"), "protein", state=1)
    cmd.set("dot_solvent", 1)
    cmd.set("dot_density", 4)
    for probe in (1.4, 2.5, 4.0, 6.0):
        cmd.set("solvent_radius", probe)
        cmd.get_area("protein", state=1, load_b=1)
        residues = {}
        for atom in cmd.get_model("protein", state=1).atom:
            key = (atom.resi, atom.resn)
            residues[key] = residues.get(key, 0.0) + atom.b
        rows.extend({"pdb_id":pdb_id,"residue":key[0],"name":key[1],"probe":probe,"pymol_sasa":area}
                    for key, area in residues.items())
    # No computed exposure colours or scores in the manual inspection view.
    cmd.alter("protein", "b=0")
    cmd.set("solvent_radius", 1.4)
    cmd.hide("everything")
    cmd.show("cartoon", "protein")
    cmd.show("surface", "protein")
    cmd.color("gray70", "protein")
    cmd.set("transparency", 0.35)
    cmd.bg_color("white")
    cmd.orient("protein")
    cmd.save(str(ROOT / f"{pdb_id}_inspect.pse"))
with (ROOT / "pymol_sasa.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Exported {len(rows)} residue/probe measurements and three inspection sessions")
def inspect_residue(residue):
    """Highlight a residue while keeping the complete protein as context."""
    cmd.hide("sticks", "protein")
    cmd.hide("labels", "protein")
    cmd.color("gray70", "protein")
    cmd.select("focus", f"protein and resi {int(residue)}")
    cmd.show("sticks", "focus")
    cmd.color("magenta", "focus")
    cmd.label("focus and name CA", "resn + resi")
    cmd.zoom("focus", buffer=8)
cmd.extend("inspect_residue", inspect_residue)
if GUI:
    cmd.reinitialize()
    cmd.load(str(ROOT / "1UBQ_inspect.pse"))
else:
    cmd.quit()
