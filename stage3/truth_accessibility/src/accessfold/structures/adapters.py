"""Optional adapters from third-party structure objects. Requires `biotite` (extra: accessfold[biotite]).

PROVISIONAL: the residue-numbering policy here (sequence index = res_id - first res_id, so numbering
gaps become unresolved residues named "UNK") is a placeholder until the Stage 1 dataset fixes the
PDB-numbering <-> SEQRES mapping. Replace `_sequence_index` when that mapping exists.
"""

from __future__ import annotations

import numpy as np

from accessfold.structures.atomic import AtomicStructure


def from_biotite(atom_array, chain_id: str | None = None) -> AtomicStructure:
    """Convert a biotite AtomArray (single model) into an AtomicStructure.

    Keeps amino-acid, non-hetero, non-hydrogen atoms of one chain. Raises on insertion codes and
    on multiple altlocs-as-duplicates (resolve those upstream, e.g. `biotite` `altloc="first"`).
    """
    import biotite.structure as struc

    arr = atom_array
    if chain_id is None:
        chain_id = str(arr.chain_id[0])
    keep = (arr.chain_id == chain_id) & struc.filter_amino_acids(arr) & (arr.element != "H") & (~arr.hetero)
    arr = arr[keep]
    if arr.array_length() == 0:
        raise ValueError(f"no amino-acid heavy atoms found for chain {chain_id!r}")
    if np.any(np.char.strip(arr.ins_code.astype(str)) != ""):
        raise ValueError("insertion codes present; resolve numbering (Stage 1 mapping) before conversion")

    res_index = _sequence_index(arr.res_id)
    n_res = int(res_index.max()) + 1
    names = np.full(n_res, "UNK", dtype="<U3")
    names[res_index] = arr.res_name  # duplicates within a residue write the same name
    keys = list(zip(res_index.tolist(), arr.atom_name.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (residue, atom name) pairs: unresolved altlocs or microheterogeneity")
    return AtomicStructure(arr.coord, np.char.upper(arr.element.astype(str)), arr.atom_name, res_index, names, chain_id)


def _sequence_index(res_id: np.ndarray) -> np.ndarray:
    return np.asarray(res_id, dtype=np.int64) - int(np.min(res_id))
