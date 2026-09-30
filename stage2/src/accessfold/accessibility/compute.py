"""Public entry point: structure -> residue-level accessibility vector.

    result = compute_accessibility(structure, method="relative_sasa", probe_radius=1.4)
    y = result.y            # [N] float, NaN where the value is not defined
    ok = result.valid_mask  # [N] bool

Methods live in a registry so that later forward models (e.g. Stage 2C CpK/rotamer accessibility)
can be added with `register_method` without changing any caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from accessfold.accessibility.reference import ReferenceTable, get_reference, residue_sasa
from accessfold.accessibility.sasa import DEFAULT_N_POINTS
from accessfold.structures.atomic import AtomicStructure


@dataclass(frozen=True)
class AccessibilityResult:
    """All arrays are [N] over the full sequence. Unresolved / masked residues are NaN.

    absolute:            residue SASA, Angstrom^2
    relative:            absolute / max-ASA (NaN if no reference); can exceed 1, see `clip`
    sidechain_absolute:  side-chain-atom SASA (NaN for glycine)
    sidechain_relative:  only if the reference table has side-chain maxima
    mask:                residue resolved and not within `gap_flank` of an unresolved residue
    primary:             which array `.y` returns ("absolute" | "relative" | ...)
    """

    absolute: np.ndarray
    relative: np.ndarray
    sidechain_absolute: np.ndarray
    sidechain_relative: np.ndarray
    mask: np.ndarray
    residue_names: np.ndarray
    method: str
    probe_radius: float
    primary: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def values(self, kind: str) -> np.ndarray:
        try:
            arr = {
                "absolute": self.absolute, "relative": self.relative,
                "sidechain_absolute": self.sidechain_absolute, "sidechain_relative": self.sidechain_relative,
            }[kind]
        except KeyError as exc:
            raise ValueError(f"unknown kind {kind!r}") from exc
        return np.where(self.mask, arr, np.nan)

    @property
    def y(self) -> np.ndarray:
        """The method's headline per-residue vector, NaN where masked or undefined."""
        return self.values(self.primary)

    @property
    def valid_mask(self) -> np.ndarray:
        """[N] bool: residues where `.y` is finite and usable in an error metric."""
        return np.isfinite(self.y)


MethodFn = Callable[..., AccessibilityResult]
_METHODS: Dict[str, MethodFn] = {}


def register_method(name: str) -> Callable[[MethodFn], MethodFn]:
    """Decorator to add a forward model. Signature: fn(structure, *, probe_radius, **options)."""
    def deco(fn: MethodFn) -> MethodFn:
        _METHODS[name] = fn
        return fn
    return deco


def _flank_mask(resolved: np.ndarray, gap_flank: int) -> np.ndarray:
    """resolved AND not within `gap_flank` positions of an unresolved residue *inside* the chain.

    Residues missing at the N/C-terminal ends are chain truncation, not a gap, and do not count.
    """
    if gap_flank <= 0 or not resolved.any():
        return resolved.copy()
    n = resolved.size
    pos = np.flatnonzero(resolved)
    interior_gap = ~resolved.copy()
    interior_gap[: pos[0]] = False
    interior_gap[pos[-1] + 1:] = False
    near = interior_gap.copy()
    for k in range(1, gap_flank + 1):
        near[k:] |= interior_gap[:-k]   # gap k positions before
        near[:-k] |= interior_gap[k:]   # gap k positions after
    return resolved & ~near


def _build(structure: AtomicStructure, *, method: str, primary: str, probe_radius: float,
           reference: Optional[ReferenceTable], need_reference: bool, n_points: int, radii_table: str,
           gap_flank: int, clip: Optional[float]) -> AccessibilityResult:
    total, side = residue_sasa(structure, probe_radius, n_points, radii_table)
    resolved = structure.resolved_mask
    n = structure.n_residues
    has_side = np.bincount(structure.residue_index, weights=structure.is_sidechain.astype(float), minlength=n) > 0

    absolute = np.where(resolved, total, np.nan)
    sc_abs = np.where(resolved & has_side, side, np.nan)

    relative = np.full(n, np.nan)
    sc_rel = np.full(n, np.nan)
    meta: Dict[str, object] = {"n_points": n_points, "radii_table": radii_table, "gap_flank": gap_flank, "clip": clip}
    if need_reference and reference is None:
        reference = get_reference(probe_radius)  # raises a clear error if none exists for this radius
    elif reference is None:
        try:
            reference = get_reference(probe_radius)
        except ValueError:
            reference = None
    if reference is not None:
        if not np.isclose(reference.probe_radius, probe_radius):
            raise ValueError(f"reference built for probe {reference.probe_radius} A, requested {probe_radius} A")
        # Only the radii table is physically meaningful to match. `reference.n_points` is deliberately NOT
        # compared: tables are built with a higher point count than routine structure calls (a maximum over
        # many noisy estimates is otherwise biased upwards), so the two normally differ.
        if reference.radii_table not in (None, radii_table):
            meta["reference_config_mismatch"] = True  # surfaced, not fatal: see tests/test_reference.py
        relative = absolute / reference.lookup(structure.residue_names)
        if reference.sidechain_max_asa is not None:
            sc_rel = sc_abs / reference.lookup(structure.residue_names, sidechain=True)
        if clip is not None:
            relative = np.minimum(relative, clip)
            sc_rel = np.minimum(sc_rel, clip)
        meta["reference"] = reference.name
        unref = sorted({str(r) for r in structure.residue_names[resolved & ~np.isfinite(relative)]})
        meta["unreferenced_residues"] = unref
    return AccessibilityResult(
        absolute=absolute, relative=relative, sidechain_absolute=sc_abs, sidechain_relative=sc_rel,
        mask=_flank_mask(resolved, gap_flank), residue_names=structure.residue_names, method=method,
        probe_radius=float(probe_radius), primary=primary, metadata=meta,
    )


@register_method("absolute_sasa")
def _absolute_sasa(structure: AtomicStructure, *, probe_radius: float = 1.4, reference: Optional[ReferenceTable] = None,
                   n_points: int = DEFAULT_N_POINTS, radii_table: str = "bondi", gap_flank: int = 0,
                   clip: Optional[float] = None) -> AccessibilityResult:
    return _build(structure, method="absolute_sasa", primary="absolute", probe_radius=probe_radius,
                  reference=reference, need_reference=False, n_points=n_points, radii_table=radii_table,
                  gap_flank=gap_flank, clip=clip)


@register_method("relative_sasa")
def _relative_sasa(structure: AtomicStructure, *, probe_radius: float = 1.4, reference: Optional[ReferenceTable] = None,
                   n_points: int = DEFAULT_N_POINTS, radii_table: str = "bondi", gap_flank: int = 0,
                   clip: Optional[float] = None) -> AccessibilityResult:
    return _build(structure, method="relative_sasa", primary="relative", probe_radius=probe_radius,
                  reference=reference, need_reference=True, n_points=n_points, radii_table=radii_table,
                  gap_flank=gap_flank, clip=clip)


def compute_accessibility(structure: AtomicStructure, method: str = "relative_sasa", probe_radius: float = 1.4,
                          **options) -> AccessibilityResult:
    """Compute residue-level accessibility for one structure.

    Args:
        structure: AtomicStructure (truth, candidate, or intermediate coordinates via `with_coords`).
        method: "relative_sasa" (needs a max-ASA reference for `probe_radius`) or "absolute_sasa".
        probe_radius: Angstrom. 1.4 = water; larger values probe coarser surface shape.
        **options: reference, n_points, radii_table, gap_flank, clip (see method docstrings).

    Returns: AccessibilityResult; use `.y` and `.valid_mask` for the headline vector.
    """
    try:
        fn = _METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unknown method {method!r}; available: {sorted(_METHODS)}") from exc
    return fn(structure, probe_radius=probe_radius, **options)


def compute_accessibility_profile(structure: AtomicStructure, probe_radii: Iterable[float] = (1.4, 2.5, 4.0, 6.0),
                                  method: str = "relative_sasa", references: Optional[Dict[float, ReferenceTable]] = None,
                                  **options) -> Dict[float, AccessibilityResult]:
    """Stage 2B convenience: the same structure at several probe radii. `references` maps radius -> table."""
    out = {}
    for r in probe_radii:
        opts = dict(options)
        if references and float(r) in references:
            opts["reference"] = references[float(r)]
        out[float(r)] = compute_accessibility(structure, method=method, probe_radius=r, **opts)
    return out
