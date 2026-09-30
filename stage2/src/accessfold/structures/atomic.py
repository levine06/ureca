"""Array-level structure container: the input contract for every accessibility function.

Design intent: later stages call the forward model on (a) the true PDB chain, (b) OpenFold3
candidate structures, and (c) intermediate diffusion coordinates. All three share one
*topology* (which atom is which, which residue it belongs to) and differ only in coordinates,
so topology and coordinates are kept separate and `with_coords` swaps coordinates cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

# Atoms counted as "backbone" when splitting a residue into backbone / side chain.
BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


@dataclass(frozen=True)
class AtomicStructure:
    """Heavy-atom structure of ONE chain, indexed by sequence position.

    Shapes (A = number of atoms present, N = number of residues in the *full* sequence):
        coords         [A, 3]  float, Angstrom
        elements       [A]     str, upper-case element symbol ("C", "N", "O", "S", "SE")
        atom_names     [A]     str, PDB atom names ("CA", "CB", ...)
        residue_index  [A]     int in [0, N): sequence position of the residue each atom belongs to
        residue_names  [N]     str, three-letter codes ("ALA", ...) for EVERY sequence position

    Residues with no atoms (unresolved in the experiment) are legal and are reported as
    masked-out by the accessibility functions. Use sequence (SEQRES-style) indexing, not PDB
    numbering, so truth and predicted structures line up position-by-position.
    """

    coords: np.ndarray
    elements: np.ndarray
    atom_names: np.ndarray
    residue_index: np.ndarray
    residue_names: np.ndarray
    chain_id: str = "A"

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float64)
        elements = np.char.upper(np.asarray(self.elements, dtype=str))
        atom_names = np.char.upper(np.asarray(self.atom_names, dtype=str))
        residue_index = np.asarray(self.residue_index, dtype=np.int64)
        residue_names = np.char.upper(np.asarray(self.residue_names, dtype=str))

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape [A, 3], got {coords.shape}")
        a = coords.shape[0]
        for name, arr in (("elements", elements), ("atom_names", atom_names), ("residue_index", residue_index)):
            if arr.shape != (a,):
                raise ValueError(f"{name} must have shape [{a}], got {arr.shape}")
        if residue_names.ndim != 1:
            raise ValueError("residue_names must be 1-D [N]")
        if not np.all(np.isfinite(coords)):
            raise ValueError("coords contain NaN/inf; drop or mask those atoms before building the structure")
        if a and (residue_index.min() < 0 or residue_index.max() >= residue_names.shape[0]):
            raise ValueError("residue_index out of range for residue_names")

        for name, arr in (
            ("coords", coords),
            ("elements", elements),
            ("atom_names", atom_names),
            ("residue_index", residue_index),
            ("residue_names", residue_names),
        ):
            object.__setattr__(self, name, arr)

    # ------------------------------------------------------------------ basic properties
    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_residues(self) -> int:
        return int(self.residue_names.shape[0])

    @property
    def resolved_mask(self) -> np.ndarray:
        """[N] bool: True where the residue has at least one atom."""
        mask = np.zeros(self.n_residues, dtype=bool)
        mask[self.residue_index] = True
        return mask

    @property
    def is_backbone(self) -> np.ndarray:
        """[A] bool: atom is N, CA, C, O or OXT."""
        return np.isin(self.atom_names, list(BACKBONE_ATOM_NAMES))

    @property
    def is_sidechain(self) -> np.ndarray:
        """[A] bool: complement of `is_backbone` (CB and beyond; none for glycine)."""
        return ~self.is_backbone

    # ------------------------------------------------------------------ utilities
    def with_coords(self, coords: np.ndarray) -> "AtomicStructure":
        """Same topology, new coordinates [A, 3] (e.g. a predicted candidate or a diffusion step)."""
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape != self.coords.shape:
            raise ValueError(f"expected coords of shape {self.coords.shape}, got {coords.shape}")
        return replace(self, coords=coords)

    @classmethod
    def from_atoms(
        cls,
        coords: np.ndarray,
        elements: Sequence[str],
        atom_names: Sequence[str],
        residue_index: Sequence[int],
        residue_names: Sequence[str],
        chain_id: str = "A",
    ) -> "AtomicStructure":
        return cls(coords, np.asarray(elements), np.asarray(atom_names), np.asarray(residue_index),
                   np.asarray(residue_names), chain_id)
