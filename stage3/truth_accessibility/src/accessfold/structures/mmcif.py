"""Load ONE polymer chain of a PDBx/mmCIF file as an `AtomicStructure` indexed by SEQRES position.

This replaces the provisional numbering policy of `adapters.from_biotite` for Stage 1/3 data: mmCIF stores
`label_seq_id`, the 1-based position of each residue in the entity's full sequence (SEQRES), so there is no
guessing from author numbering, insertion codes or numbering gaps. Position i of every returned array is
residue i+1 of the sequence you pass in; residues that were not observed simply have no atoms.

Rules (all counted in the returned QC dict):
  * only the requested `label_asym_id` (one chain, isolated monomer: other chains, waters, ligands, ions dropped);
  * polymer residues only (`label_seq_id` present). Modified residues bonded into the chain (e.g. MSE) are kept;
  * hydrogens/deuterium dropped; atoms with occupancy 0 dropped; one model (the first by default);
  * alternate locations: per (residue, atom name) the highest-occupancy conformer is kept (ties: first listed);
    if one position has several residue names (microheterogeneity), the highest total occupancy wins;
  * residues whose structure name differs from the sequence residue are reported in qc["modified_residues"] (identity,
    parent, extra atoms), except selenomethionine (MSE -> MET); they are NOT silently relabelled as the parent;
  * residue names come from `sequence` (canonical one-letter code -> three-letter), NOT from the structure, so
    truth and predicted structures share residue_names exactly; disagreements are counted, not hidden.
Requires biotite (only to read the CIF text); imported lazily.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from accessfold.accessibility.radii import RADII_TABLES
from accessfold.structures.atomic import AtomicStructure

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS",
    "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

#: heavy atoms (excluding OXT) a complete residue of each type must have
EXPECTED_HEAVY_ATOMS: Dict[str, frozenset] = {k: frozenset(v.split()) for k, v in {
    "ALA": "N CA C O CB", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2", "ASN": "N CA C O CB CG OD1 ND2",
    "ASP": "N CA C O CB CG OD1 OD2", "CYS": "N CA C O CB SG", "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2", "GLY": "N CA C O", "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "ILE": "N CA C O CB CG1 CG2 CD1", "LEU": "N CA C O CB CG CD1 CD2", "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "N CA C O CB CG CD",
    "SER": "N CA C O CB OG", "THR": "N CA C O CB OG1 CG2",
    "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH",
    "VAL": "N CA C O CB CG1 CG2",
}.items()}

#: the ONLY non-standard residue name treated as its parent amino acid: selenomethionine replaces one sulfur by
#: selenium and keeps the shape. Every other non-standard residue changes the chemistry (extra atoms or groups), so its
#: value is masked and its neighbours' values are checked for shielding by the extra atoms (see accessfold.truth).
ISOSTERIC = {"MSE": "MET"}

_NO_GAP = 9999


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _col(block_cat, name: str, dtype=str) -> np.ndarray:
    return block_cat[name].as_array(dtype)


def gap_distance(resolved: np.ndarray) -> np.ndarray:
    """[N] int: distance in sequence positions to the nearest INTERNAL unresolved residue (0 = is one itself).

    Missing residues at the N/C-terminal ends are chain truncation, not gaps (same rule as compute._flank_mask).
    `_NO_GAP` (9999) where there is none. Lets a downstream user pick any `gap_flank` without recomputing.
    """
    n = resolved.size
    out = np.full(n, _NO_GAP, dtype=np.int64)
    if not resolved.any():
        return out
    pos = np.flatnonzero(resolved)
    internal = np.flatnonzero(~resolved)
    internal = internal[(internal > pos[0]) & (internal < pos[-1])]
    for g in internal:
        out = np.minimum(out, np.abs(np.arange(n) - g))
    return out


def load_chain_from_mmcif(
    cif_path: str | Path, label_asym_id: str, sequence: str, model: Optional[int] = None,
) -> Tuple[AtomicStructure, np.ndarray, dict]:
    """Return (structure, complete, qc).

    structure: AtomicStructure indexed by sequence position (unresolved residues have no atoms).
    complete:  [N] bool, True where the residue has every heavy atom its type should have (side chain included).
    qc:        plain-valued dict of counts/warnings (JSON-serialisable).
    See the module docstring for the rules applied."""
    from biotite.structure.io.pdbx import CIFFile

    path = Path(cif_path)
    with _open_text(path) as fh:
        cif = CIFFile.read(fh)
    block = cif[next(iter(cif.keys()))]
    atom = block["atom_site"]

    chain = _col(atom, "label_asym_id")
    seq_raw = _col(atom, "label_seq_id")
    model_num = _col(atom, "pdbx_PDB_model_num") if "pdbx_PDB_model_num" in atom else np.full(chain.shape, "1")
    elem = np.char.upper(_col(atom, "type_symbol"))
    qc: dict = {"cif": path.name, "warnings": []}

    models = sorted(set(model_num.tolist()), key=lambda s: int(s))
    qc["n_models_in_file"] = len(models)
    use_model = str(model) if model is not None else models[0]
    base = (chain == label_asym_id) & (model_num == use_model) & ~np.isin(seq_raw, [".", "?"])
    if not base.any():
        raise ValueError(f"{path.name}: no polymer atoms for label_asym_id={label_asym_id!r} (model {use_model})")

    qc["n_with_coordinate_records"] = len(set(seq_raw[base].astype(int).tolist()))   # any atom record, any occupancy
    keep = base & ~np.isin(elem, ["H", "D"])
    occ = _col(atom, "occupancy", float) if "occupancy" in atom else np.ones(chain.shape)
    qc["n_atoms_occupancy_zero_dropped"] = int((keep & (occ <= 0)).sum())
    keep &= occ > 0
    supported = np.isin(elem, list(RADII_TABLES["bondi"]))
    qc["n_atoms_unsupported_element_dropped"] = int((keep & ~supported).sum())
    keep &= supported

    idx = np.flatnonzero(keep)
    seq_id = seq_raw[idx].astype(int)
    n_seq = len(sequence)
    if seq_id.size == 0:
        raise ValueError(f"{path.name}: chain {label_asym_id!r} has no usable atoms")
    if seq_id.min() < 1 or seq_id.max() > n_seq:
        raise ValueError(f"{path.name}: label_seq_id range {seq_id.min()}..{seq_id.max()} does not fit the "
                         f"{n_seq}-residue sequence supplied (wrong entity/sequence?)")
    comp = _col(atom, "label_comp_id")[idx]
    names = _col(atom, "label_atom_id")[idx]
    names = np.char.strip(names, '"')
    xyz = np.stack([_col(atom, c, float)[idx] for c in ("Cartn_x", "Cartn_y", "Cartn_z")], axis=1)
    occ_i, elem_i = occ[idx], elem[idx]
    alt = _col(atom, "label_alt_id")[idx] if "label_alt_id" in atom else np.full(idx.shape, ".")

    # microheterogeneity: choose the residue name with the larger total occupancy at each position
    tot: Dict[Tuple[int, str], float] = defaultdict(float)
    for s, c, o in zip(seq_id, comp, occ_i):
        tot[(int(s), str(c))] += float(o)
    chosen: Dict[int, str] = {}
    for (s, c), t in tot.items():
        if s not in chosen or t > tot[(s, chosen[s])]:
            chosen[s] = c
    qc["n_microheterogeneous_positions"] = int(sum(1 for s in chosen if sum(1 for (s2, _) in tot if s2 == s) > 1))

    best: Dict[Tuple[int, str], int] = {}
    tie_pos = set()
    for k in range(idx.size):
        if comp[k] != chosen[int(seq_id[k])]:
            continue
        key = (int(seq_id[k]), str(names[k]))
        if key not in best or occ_i[k] > occ_i[best[key]]:
            best[key] = k
        elif occ_i[k] == occ_i[best[key]]:
            tie_pos.add(int(seq_id[k]))   # equal-occupancy altlocs: the first listed is kept
    qc["n_residues_altloc_tie"] = len(tie_pos)
    sel = np.array(sorted(best.values()), dtype=int)
    n_same_name = sum(1 for k in range(idx.size) if comp[k] == chosen[int(seq_id[k])])
    qc["n_altloc_atoms_dropped"] = int(n_same_name - sel.size)
    alt_pos = {int(seq_id[k]) for k in range(idx.size) if str(alt[k]) not in (".", "?")}
    qc["n_residues_with_altloc"] = len(alt_pos)

    res_names = np.array([ONE_TO_THREE.get(ch, "UNK") for ch in sequence.upper()], dtype="<U3")
    qc["n_nonstandard_sequence_letters"] = int((res_names == "UNK").sum())
    structure = AtomicStructure(xyz[sel], elem_i[sel], names[sel], seq_id[sel] - 1, res_names, chain_id=label_asym_id)

    # per-position bookkeeping ------------------------------------------------------------------------
    resolved = structure.resolved_mask
    atoms_by_pos: Dict[int, set] = defaultdict(set)
    for p, nme in zip(structure.residue_index, structure.atom_names):
        atoms_by_pos[int(p)].add(str(nme))
    complete = np.zeros(n_seq, dtype=bool)
    for p, have in atoms_by_pos.items():
        need = EXPECTED_HEAVY_ATOMS.get(str(res_names[p]))
        if res_names[p] == "MET" and "SE" in have:  # selenomethionine: SE takes the place of SD
            have = (have - {"SE"}) | {"SD"}
        complete[p] = need is not None and need <= have
    # chemistry check: structure residue name vs sequence residue -------------------------------------------
    parents, chem_names = _modres_info(block, label_asym_id)
    standard3 = set(ONE_TO_THREE.values())
    modified, n_seleno = [], 0
    for sid, c in sorted(chosen.items()):
        seq_name = str(res_names[sid - 1])
        if c == seq_name:
            continue
        if ISOSTERIC.get(c) == seq_name:
            n_seleno += 1
            continue
        pos_atoms = atoms_by_pos.get(sid - 1, set())
        std_atoms = EXPECTED_HEAVY_ATOMS.get(seq_name, frozenset()) | {"OXT"}
        occs = [float(occ_i[k]) for k in sel if int(seq_id[k]) == sid]
        modified.append({
            "index": sid - 1, "position": sid, "comp_id": c, "sequence_residue": seq_name,
            "kind": "sequence_mismatch" if c in standard3 else "modified",
            "parent": parents.get(sid), "name": chem_names.get(c),
            "extra_atoms": sorted(pos_atoms - std_atoms) if seq_name in EXPECTED_HEAVY_ATOMS else [],
            "mean_occupancy": round(float(np.mean(occs)), 3) if occs else None,
        })
    n_records = qc["n_with_coordinate_records"]
    qc.update({
        "n_sequence": n_seq, "n_atoms": structure.n_atoms, "n_resolved": int(resolved.sum()),
        "resolved_fraction": float(resolved.mean()), "n_incomplete_residues": int((resolved & ~complete).sum()),
        "n_record_but_unusable": n_records - int(resolved.sum()),
        "records_fraction": n_records / n_seq,
        "n_modified_residues": len(modified), "n_selenomethionine": n_seleno, "modified_residues": modified,
        "n_name_mismatch_vs_sequence": len(modified),
        "longest_internal_gap": _longest_internal_gap(resolved),
    })
    if n_records != int(resolved.sum()):
        qc["warnings"].append(f"{n_records - int(resolved.sum())} residue(s) have coordinate records but no usable "
                              f"atoms (occupancy 0): usable coverage {resolved.mean():.4f} vs records {n_records / n_seq:.4f}")
    if modified:
        qc["warnings"].append("masked non-standard/mismatching residue(s): " + ", ".join(
            f"{m['position']}:{m['comp_id']}" + (f"(parent {m['parent']})" if m["parent"] else "") for m in modified))
    return structure, complete, qc


def _modres_info(block, label_asym_id: str):
    """({seq_id: parent comp id}, {comp id: chemical name}) from pdbx_struct_mod_residue / chem_comp, if present."""
    parents, names = {}, {}
    try:
        if "pdbx_struct_mod_residue" in block:
            t = block["pdbx_struct_mod_residue"]
            for asym, sid, par in zip(_col(t, "label_asym_id"), _col(t, "label_seq_id"), _col(t, "parent_comp_id")):
                if asym == label_asym_id and sid not in (".", "?"):
                    parents[int(sid)] = str(par) if par not in (".", "?") else None
        if "chem_comp" in block:
            t = block["chem_comp"]
            for cid, nm in zip(_col(t, "id"), _col(t, "name")):
                names[str(cid)] = str(nm)
    except Exception:  # noqa: BLE001 - identity is informational only
        pass
    return parents, names


def _longest_internal_gap(resolved: np.ndarray) -> int:
    if not resolved.any():
        return 0
    pos = np.flatnonzero(resolved)
    inner = ~resolved[pos[0]: pos[-1] + 1]
    best = run = 0
    for v in inner:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)
