"""Run inside PyMOL; writes independent areas without changing the current object."""
import csv
import json
import inspect
from pathlib import Path
from pymol import cmd

ROOT = Path(inspect.currentframe().f_code.co_filename).resolve().parent
measurement_objects = []
rows = []
try:
    def measure(path, probe):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        measurement_object = cmd.get_unused_name("afcheck")
        measurement_objects.append(measurement_object)
        # Direct PDB ingestion avoids file-loader splitting and object-name reuse.
        cmd.read_pdbstr(path.read_text(), measurement_object, state=1, zoom=0, multiplex=0)
        loaded_objects = cmd.get_names("objects")
        if measurement_object not in loaded_objects:
            raise RuntimeError(f"Expected {measurement_object} after reading {path.name}; found {loaded_objects}")
        selection = "%" + measurement_object
        if cmd.count_atoms(selection) == 0:
            raise RuntimeError(f"No atoms loaded from {path}; objects: {cmd.get_names('objects')}")
        # Keep the object enabled for area calculation, but hide its representations.
        cmd.enable(measurement_object)
        cmd.hide("everything", selection)
        cmd.alter(selection, "vdw=radii[elem.upper()]", space={"radii":{"C":1.70,"N":1.55,"O":1.52,"S":1.80}})
        cmd.set("dot_solvent", 1, measurement_object)
        cmd.set("dot_density", 4, measurement_object)
        cmd.set("solvent_radius", probe, measurement_object)
        print(f"STEP2: {path.name}, probe={probe}, object={measurement_object}, atoms={cmd.count_atoms(selection)}")
        cmd.get_area(selection, state=1, load_b=1)
        areas = {}
        for atom in cmd.get_model(selection, state=1).atom:
            areas[atom.resi] = areas.get(atom.resi, 0.0) + atom.b
        return areas
    for pdb_id, residues in {"1CRN":(16,30),"1LYZ":(25,108,98)}.items():
        for probe in (0.0,0.5,1.0,1.4):
            areas = measure(ROOT / f"{pdb_id}_clean.pdb", probe)
            for residue in residues:
                rows.append({"kind":"residue","id":pdb_id,"residue":residue,"probe":probe,"pymol_A2":areas[str(residue)]})
    for record in json.loads((ROOT / "reference_winners.json").read_text()):
        areas = measure(ROOT / record["file"], record["probe"])
        rows.append({"kind":"reference","id":record["name"],"residue":2,"probe":record["probe"],"pymol_A2":areas["2"]})
    with (ROOT / "step2_pymol.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"STEP 2 EXPORT COMPLETE: {len(rows)} measurements saved to step2_pymol.csv")
finally:
    for measurement_object in measurement_objects:
        cmd.delete(measurement_object)
