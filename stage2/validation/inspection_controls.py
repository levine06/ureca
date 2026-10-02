"""Restore inspection commands after opening a saved PyMOL session."""
from pymol import cmd


def inspect_residue(residue):
    """Highlight a residue with the complete protein retained as context."""
    residue = int(residue)
    cmd.hide("sticks", "protein")
    cmd.hide("labels", "protein")
    cmd.color("gray70", "protein")
    cmd.select("focus", f"protein and resi {residue}")
    cmd.show("sticks", "focus")
    cmd.color("magenta", "focus")
    cmd.label("focus and name CA", "resn + resi")
    cmd.zoom("focus", buffer=8)


cmd.extend("inspect_residue", inspect_residue)
print("Inspection command restored. Enter: inspect_residue 13")
