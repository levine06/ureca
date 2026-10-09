"""Load ONE chain of a PREDICTED structure (OpenFold3 or similar) as an `AtomicStructure` indexed by sequence position.

Predicted files are simpler than experimental ones, but a wrong file must never be scored silently, so correspondence
with the input sequence is VERIFIED, not assumed:
  * `.cif` / `.cif.gz`  -> the experimental mmCIF loader (`label_seq_id` / `label_asym_id`); any residue whose name differs
    from the sequence (the loader only counts these as "modified"/"mismatching") is an error here, except
    selenomethionine written for Met;
  * `.pdb` / `.pdb.gz`  -> a minimal fixed-column PDB reader. Required: residue numbers 1..N along the sequence, blank
    insertion codes, the residue NAME at every position equal to the sequence residue (MSE accepted for Met), and no
    duplicate (residue, atom name) records (alternate locations are not expected in a prediction). Hydrogens and
    occupancy-0 atoms are dropped.
Any violation raises ValueError (the candidate is then reported as failed, never scored). The residue names of the
returned structure always come from `sequence`, exactly as for the truth.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Tuple

import numpy as np

from accessfold.accessibility.radii import RADII_TABLES
from accessfold.structures.atomic import AtomicStructure
from accessfold.structures.mmcif import EXPECTED_HEAVY_ATOMS, ISOSTERIC, ONE_TO_THREE, load_chain_from_mmcif


def _expected_names(sequence: str):
    return [ONE_TO_THREE.get(c, "UNK") for c in sequence.upper()]


def _load_pdb(path: Path, chain: str, sequence: str) -> Tuple[AtomicStructure, np.ndarray, dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    xyz, el, nm, pos, seen_names = [], [], [], [], {}
    seen_keys = set()
    first_model_done = False
    supported = set(RADII_TABLES["bondi"])
    n = len(sequence)
    expected = _expected_names(sequence)
    with opener(path, "rt") as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                first_model_done = True
            if first_model_done or rec not in ("ATOM  ", "HETATM") or line[21] != chain:
                continue
            name = line[12:16].strip()
            element = (line[76:78].strip() or name[:1]).upper()
            if element in ("H", "D") or element not in supported:
                continue
            try:
                occ = float(line[54:60]) if line[54:60].strip() else 1.0
            except ValueError:
                occ = 1.0
            if occ <= 0:
                continue
            resname = line[17:20].strip().upper()
            icode = line[26].strip()
            resseq = int(line[22:26])
            if icode:
                raise ValueError(f"{path.name}: insertion code {icode!r} at residue {resseq}; predicted chains must be "
                                 "numbered 1..N without insertion codes")
            if resseq < 1 or resseq > n:
                raise ValueError(f"{path.name}: residue number {resseq} does not fit the {n}-residue sequence "
                                 "(predicted PDB files must be numbered 1..N along the sequence)")
            want = expected[resseq - 1]
            if resname != want and ISOSTERIC.get(resname) != want:
                raise ValueError(f"{path.name}: residue {resseq} is {resname} but the sequence has {want} there "
                                 "(wrong chain, sequence or numbering)")
            if seen_names.setdefault(resseq, resname) != resname:
                raise ValueError(f"{path.name}: residue {resseq} has more than one residue name")
            if (resseq, name) in seen_keys:
                raise ValueError(f"{path.name}: duplicate atom {name} in residue {resseq} (alternate locations or a "
                                 "repeated record); a prediction must have one position per atom")
            seen_keys.add((resseq, name))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            el.append(element)
            nm.append(name)
            pos.append(resseq)
    if not xyz:
        raise ValueError(f"{path.name}: no atoms for chain {chain!r}")
    pos = np.asarray(pos)
    res_names = np.array(expected, dtype="<U3")
    s = AtomicStructure(np.asarray(xyz), np.asarray(el), np.asarray(nm), pos - 1, res_names, chain_id=chain)
    have = {}
    for p, a in zip(s.residue_index, s.atom_names):
        have.setdefault(int(p), set()).add(str(a))
    complete = np.zeros(n, bool)
    for p, atoms in have.items():
        need = EXPECTED_HEAVY_ATOMS.get(str(res_names[p]))
        if res_names[p] == "MET" and "SE" in atoms:
            atoms = (atoms - {"SE"}) | {"SD"}
        complete[p] = need is not None and need <= atoms
    qc = {"file": path.name, "format": "pdb", "n_resolved": int(s.resolved_mask.sum()), "warnings": []}
    return s, complete, qc


def load_predicted_chain(path: str | Path, chain: str, sequence: str, model: int | None = None):
    """(structure, complete, qc) for one predicted chain, with its correspondence to `sequence` verified."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz")):
        s, complete, qc = load_chain_from_mmcif(p, chain, sequence, model=model)
        if qc["n_modified_residues"]:
            bad = ", ".join(f"{m['position']}:{m['comp_id']} (sequence {m['sequence_residue']})"
                            for m in qc["modified_residues"][:5])
            raise ValueError(f"{p.name}: residue identities differ from the sequence at {qc['n_modified_residues']} "
                             f"position(s): {bad} (wrong chain, sequence or numbering)")
        return s, complete, qc
    if name.endswith((".pdb", ".pdb.gz", ".ent", ".ent.gz")):
        return _load_pdb(p, chain, sequence)
    raise ValueError(f"unsupported candidate file type: {p.name} (use .cif/.cif.gz or .pdb/.pdb.gz)")
