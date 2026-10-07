"""Theoretical maximum accessibility from Gly-X-Gly tripeptides (after Tien et al. 2013).

Reference: Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013) "Maximum allowed solvent
accessibilities of residues in proteins", PLoS ONE 8(11):e80635.

What the paper does (as read from its Methods and Table 1):
  * tripeptide Gly-X-Gly, omega = 180 deg, bond lengths/angles = averages mined from crystal structures;
  * phi and psi of X swept exhaustively in 10 deg steps;
  * for each (phi, psi), side-chain rotamers are enumerated in 120 deg sectors of each chi angle
    (residues with > 10 rotamers were randomly subsampled to 10);
  * A_max(X) = the largest ASA of X over everything evaluated (whole residue, 1.4 A probe, DSSP).

What this module does, and where it DIFFERS (each is a recorded modelling choice, see
`TripeptideConfig` and the `extra` field of the resulting `ReferenceTable`):
  * geometry: PeptideBuilder defaults (same lab; averages over crystal structures), not Engh & Huber;
  * chi values: a staggered grid (`StaggeredChiGrid`), not the Dunbrack library. Because A_max is a
    MAXIMUM, rotamer populations are irrelevant and only the allowed chi values matter. Swap in a
    backbone-dependent library later by implementing `RotamerSource`;
  * NO random subsampling: every grid point is evaluated (so maxima can exceed the paper's for
    Arg, Lys, Gln, Glu, Met, Ile, Leu, Asn);
  * sterically impossible conformers are rejected with a hard-sphere filter (`clash_scale`);
  * the probe radius is a parameter, and SASA uses this package's own engine and radii table
    (not DSSP), so the 1.4 A table agrees with the paper only approximately;
  * the maximum is refined: the top candidates found with `n_points` sample points are re-evaluated with
    `n_points_refine` points, because a maximum over ~10^5 noisy estimates is biased upwards.

Flanking glycines stay at PeptideBuilder's default backbone angles (phi=-120, psi=140).
The C-terminal glycine has no OXT atom.

Known limitations (measured/reviewed, none changes the reported maxima by more than ~1%):
  * the maximum is refined only over the `top_m` best coarse candidates, so it can be biased low by a
    fraction of a percent when many conformers are within the coarse-estimate noise of the top;
  * Arg chi4 (planar N-C bond) is sampled on the sp3 grid; a finer chi4 sweep raised Arg at 6 A by ~0.6%;
  * Pro keeps PeptideBuilder's rigid ring at every phi (only its unrealistic phi values are affected,
    and its maximum sits at a physically sensible phi ~ -50);
  * for almost every residue the maximising backbone lies near (phi, psi) ~ (70, -70), at the edge of
    the region the hard-sphere filter allows, so A_max depends on `clash_scale` (see the README table).
"""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from accessfold.accessibility.radii import radii_from_elements
from accessfold.accessibility.reference import STANDARD_RESIDUES, ReferenceTable
from accessfold.accessibility.sasa import unit_sphere_points
from accessfold.structures.atomic import BACKBONE_ATOM_NAMES

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: (phi, psi) of the extended conformer used to derive atom connectivity (see `scan_residue`)
REFERENCE_PHI_PSI = (-120.0, 140.0)

# IUPAC side-chain chi definitions (heavy atoms). Pro is deliberately absent: its ring pucker is
# coupled to chi1/chi2, so it is evaluated only at PeptideBuilder's default ring geometry.
CHI_ATOMS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}

# chi angles whose distal group is planar and 2-fold symmetric in heavy atoms: 180-deg period
_SYMMETRIC_CHI = {("ASP", 1), ("GLU", 2), ("PHE", 1), ("TYR", 1)}
# chi angles into a planar but NON-symmetric group: full circle
_PLANAR_CHI = {("ASN", 1), ("GLN", 2), ("HIS", 1), ("TRP", 1)}


# --------------------------------------------------------------------------- rotamer sources
class RotamerSource(Protocol):
    """Supplies the side-chain chi conformations evaluated at each backbone (phi, psi)."""

    name: str

    def chi_sets(self, resname: str, phi: float, psi: float) -> np.ndarray:
        """[K, n_chi] target chi angles in DEGREES ([1, 0] when there is nothing to vary)."""
        ...

    def describe(self) -> dict: ...


@dataclass(frozen=True)
class StaggeredChiGrid:
    """Library-free rotamer set: staggered sp3-sp3 chi (+60/180/-60) and a 30-deg grid for planar chi.

    Stand-in for the Dunbrack library (needs registration, not available here). Because the paper's
    statistic is a maximum over rotamers, only which chi values are allowed matters.
    """

    sp3_values: Tuple[float, ...] = (60.0, 180.0, -60.0)
    planar_step: float = 30.0
    name: str = "staggered_chi_grid_v1"

    def chi_sets(self, resname: str, phi: float, psi: float) -> np.ndarray:
        defs = CHI_ATOMS.get(resname, [])
        if not defs:
            return np.zeros((1, 0))
        axes = []
        for j in range(len(defs)):
            if (resname, j) in _SYMMETRIC_CHI:
                axes.append(np.arange(0.0, 180.0, self.planar_step))
            elif (resname, j) in _PLANAR_CHI:
                axes.append(np.arange(-180.0, 180.0, self.planar_step))
            else:
                axes.append(np.array(self.sp3_values))
        return np.array(list(itertools.product(*axes)), dtype=np.float64)

    def describe(self) -> dict:
        return {"source": self.name, "sp3_values": list(self.sp3_values), "planar_step": self.planar_step,
                "symmetric_chi_period": 180, "backbone_dependent": False}


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TripeptideConfig:
    phi_psi_step: float = 10.0          # deg; the paper uses 10
    flank: str = "G"                     # one-letter flanking residue; the paper uses Gly
    clash_scale: float = 0.80            # reject if d < clash_scale * (r_i + r_j) for pairs >= 4 bonds apart
    radii_table: str = "bondi"
    n_points: int = 1000                 # coarse scan resolution
    n_points_refine: int = 10000         # resolution used to re-evaluate the best candidates (bundled tables)
    top_m: int = 24                      # candidates refined per (radius, quantity) (bundled tables)
    batch_size: int = 48
    dtype: str = "float32"               # coarse scan precision; refinement always float64


# --------------------------------------------------------------------------- geometry helpers
def dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """IUPAC dihedral angle (degrees) for [..., 3] point arrays."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def rotate_about_axis(points: np.ndarray, origin: np.ndarray, axis_unit: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of [B, M, 3] points about per-batch axes: origin/axis_unit [B, 3], angle [B]."""
    th = np.radians(angle_deg)[:, None, None]
    v = points - origin[:, None, :]
    k = axis_unit[:, None, :]
    kv = np.cross(k, v)
    kdv = np.sum(k * v, axis=-1, keepdims=True)
    out = v * np.cos(th) + kv * np.sin(th) + k * kdv * (1.0 - np.cos(th))
    return out + origin[:, None, :]


@dataclass
class _Topology:
    """Per-residue-type constants: atom layout, radii, chi machinery, clash pair mask."""

    resname: str
    atom_names: np.ndarray            # [A]
    elements: np.ndarray              # [A]
    residue_of_atom: np.ndarray       # [A] 0, 1, 2
    radii: np.ndarray                 # [A]
    target: np.ndarray                # [T] indices of atoms of residue X (the middle residue)
    target_is_side: np.ndarray        # [T] bool
    chi_idx: List[np.ndarray]         # per chi: 4 atom indices
    chi_moving: List[np.ndarray]      # per chi: indices of atoms that rotate
    clash_pairs: np.ndarray           # [A, A] bool, True where a steric check applies
    clash_dist: np.ndarray            # [A, A] minimum allowed distance for checked pairs (already scaled)


def _build_conformer_structure(resname: str, flank: str, phi: float, psi: float):
    from PeptideBuilder import Geometry as G, PeptideBuilder as PB  # lazy: only the reference builder needs it

    geos = [G.geometry(flank), G.geometry(THREE_TO_ONE[resname]), G.geometry(flank)]
    geos[1].phi = float(phi)
    geos[2].psi_im1 = float(psi)
    return PB.make_structure_from_geos(geos)


def _structure_arrays(structure):
    coords, names, elems, res = [], [], [], []
    for ri, residue in enumerate(structure[0]["A"]):
        for atom in residue:
            coords.append(atom.get_coord())
            names.append(atom.get_name())
            elems.append(atom.element.upper())
            res.append(ri)
    return np.array(coords, dtype=np.float64), np.array(names), np.array(elems), np.array(res)


def _graph_distances(coords: np.ndarray, cutoff: float = 1.9) -> np.ndarray:
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    adj = (d < cutoff) & ~np.eye(len(coords), dtype=bool)
    n = len(coords)
    gd = np.full((n, n), 99, dtype=np.int64)
    np.fill_diagonal(gd, 0)
    gd[adj] = 1
    for k in range(n):  # Floyd-Warshall; n <= ~40
        gd = np.minimum(gd, gd[:, k:k + 1] + gd[k:k + 1, :])
    return gd


def _make_topology(resname: str, base_coords, names, elems, res_of_atom, cfg: TripeptideConfig) -> _Topology:
    radii = radii_from_elements(elems, cfg.radii_table)
    target = np.flatnonzero(res_of_atom == 1)
    is_side = ~np.isin(names[target], list(BACKBONE_ATOM_NAMES))
    index = {(int(res_of_atom[i]), str(names[i])): i for i in range(len(names))}
    dmat = np.linalg.norm(base_coords[:, None] - base_coords[None], axis=-1)
    adj = (dmat < 1.9) & ~np.eye(len(names), dtype=bool)

    chi_idx, chi_moving = [], []
    for quad in CHI_ATOMS.get(resname, []):
        idx = np.array([index[(1, a)] for a in quad])
        b, c = idx[1], idx[2]
        seen, stack = {int(c)}, [int(c)]  # atoms reachable from c without crossing back through b
        while stack:
            cur = stack.pop()
            for nb in np.flatnonzero(adj[cur]):
                nb = int(nb)
                if nb == int(b) or nb in seen or res_of_atom[nb] != 1:
                    continue
                seen.add(nb)
                stack.append(nb)
        chi_idx.append(idx)
        chi_moving.append(np.array(sorted(seen - {int(c)}), dtype=np.int64))

    gd = _graph_distances(base_coords)
    pairs = gd >= 4
    sum_r = radii[:, None] + radii[None, :]
    return _Topology(resname, names, elems, res_of_atom, radii, target, is_side, chi_idx, chi_moving,
                     pairs, cfg.clash_scale * sum_r)


def _phi_psi_grid(step: float) -> np.ndarray:
    vals = np.arange(-180.0, 180.0, step)
    return np.array([(p, s) for p in vals for s in vals], dtype=np.float64)


def _enumerate(resname: str, source: RotamerSource, phipsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flat list of (phi/psi index, chi targets [n_chi]) over the whole scan."""
    p_idx, chis = [], []
    cache = None
    backbone_dependent = bool(source.describe().get("backbone_dependent", False))
    for p, (phi, psi) in enumerate(phipsi):
        if cache is None or backbone_dependent:
            cache = source.chi_sets(resname, float(phi), float(psi))
        p_idx.append(np.full(len(cache), p, dtype=np.int64))
        chis.append(cache)
    return np.concatenate(p_idx), np.concatenate(chis, axis=0)


def _apply_chi(coords: np.ndarray, topo: _Topology, targets: np.ndarray) -> np.ndarray:
    """Set each chi of [B, A, 3] conformers to `targets` [B, n_chi] (degrees), in order chi1, chi2, ..."""
    coords = coords.copy()
    for j, (idx, moving) in enumerate(zip(topo.chi_idx, topo.chi_moving)):
        cur = dihedral_deg(coords[:, idx[0]], coords[:, idx[1]], coords[:, idx[2]], coords[:, idx[3]])
        delta = targets[:, j] - cur
        origin = coords[:, idx[2]]
        axis = coords[:, idx[2]] - coords[:, idx[1]]
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        coords[:, moving] = rotate_about_axis(coords[:, moving], origin, axis, delta)
    return coords


def _clash_free(coords: np.ndarray, topo: _Topology) -> np.ndarray:
    """[B] bool: conformer has finite coordinates and no hard-sphere clash between atoms >= 4 bonds apart.

    Non-finite conformers MUST be rejected here: NaN compares False everywhere, so a NaN conformer would
    otherwise look clash-free and (in the SASA step) fully exposed, and could become the reported maximum.
    (PeptideBuilder does emit NaN coordinates for a few degenerate angles, e.g. Met at phi=0, psi=90.)
    """
    finite = np.isfinite(coords).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)
        bad = (d < topo.clash_dist[None]) & topo.clash_pairs[None]
    return finite & ~bad.any(axis=(1, 2))


def batch_target_sasa(coords: np.ndarray, radii: np.ndarray, target: np.ndarray, probe: float,
                      n_points: int, dtype=np.float64) -> np.ndarray:
    """Shrake-Rupley areas of the `target` atoms for [B, A, 3] conformers -> [B, T].

    Same algorithm and point set as `accessfold.accessibility.sasa.atom_sasa`, batched over conformers
    (all atoms occlude). Verified against `atom_sasa` in tests/test_tripeptide.py.
    """
    B, A, _ = coords.shape
    if not np.isfinite(coords).all():
        raise ValueError("non-finite coordinates passed to batch_target_sasa (NaN would read as 'fully exposed')")
    expanded = radii + probe
    unit = unit_sphere_points(n_points).astype(dtype)
    c = coords.astype(dtype)
    T = len(target)
    pts = c[:, target, None, :] + expanded[target][None, :, None, None].astype(dtype) * unit[None, None]  # [B,T,P,3]
    pts = pts.reshape(B, T * n_points, 3)
    cross = np.einsum("bpk,bak->bpa", pts, c)
    d2 = (pts ** 2).sum(-1)[:, :, None] - 2.0 * cross + (c ** 2).sum(-1)[:, None, :]
    d2 = d2.reshape(B, T, n_points, A)
    inside = d2 < (expanded.astype(dtype) ** 2)[None, None, None, :]
    inside[:, np.arange(T), :, target] = False  # a point is never buried by its own atom (indexing puts T first)
    buried = inside.any(axis=-1)                # [B, T, P]
    frac_open = 1.0 - buried.mean(axis=-1)
    return (4.0 * np.pi * expanded[target] ** 2)[None, :] * frac_open


def _conformers(topo: _Topology, base: np.ndarray, p_idx: np.ndarray, chi: np.ndarray) -> np.ndarray:
    coords = base[p_idx]
    if chi.shape[1]:
        coords = _apply_chi(coords, topo, chi)
    return coords


# --------------------------------------------------------------------------- per-residue scan
@dataclass
class ResidueScan:
    resname: str
    n_conformers: int
    n_clash_rejected: int          # includes the non-finite ones
    n_nonfinite: int = 0
    # per probe radius
    max_total: Dict[float, float] = field(default_factory=dict)
    max_side: Dict[float, float] = field(default_factory=dict)
    coarse_max_total: Dict[float, float] = field(default_factory=dict)
    p99_total: Dict[float, float] = field(default_factory=dict)
    argmax: Dict[float, dict] = field(default_factory=dict)
    seconds: float = 0.0


def scan_residue(resname: str, probe_radii: Sequence[float], cfg: TripeptideConfig = TripeptideConfig(),
                 source: Optional[RotamerSource] = None, phipsi: Optional[np.ndarray] = None) -> ResidueScan:
    """Scan one residue type over (phi, psi) x rotamers for every probe radius in `probe_radii`."""
    t0 = time.time()
    source = source or StaggeredChiGrid()
    phipsi = _phi_psi_grid(cfg.phi_psi_step) if phipsi is None else phipsi
    dtype = np.dtype(cfg.dtype)

    # Topology (bond graph, chi rotation sets, clash-pair mask) comes from a fixed, clash-free, always-finite
    # reference conformer, never from a grid point: a grid point can be degenerate (NaN) or clashing.
    ref_coords, names, elems, res = _structure_arrays(
        _build_conformer_structure(resname, cfg.flank, *REFERENCE_PHI_PSI))
    topo = _make_topology(resname, ref_coords, names, elems, res, cfg)
    bases = [_structure_arrays(_build_conformer_structure(resname, cfg.flank, phi, psi))[0] for phi, psi in phipsi]
    base = np.stack(bases)
    p_idx, chi = _enumerate(resname, source, phipsi)
    M = len(p_idx)
    radii = list(map(float, probe_radii))

    total = {r: np.full(M, np.nan) for r in radii}
    side = {r: np.full(M, np.nan) for r in radii}
    ok_all = np.zeros(M, dtype=bool)
    n_nonfinite = 0
    has_side = bool(topo.target_is_side.any())
    for s in range(0, M, cfg.batch_size):
        sl = slice(s, min(M, s + cfg.batch_size))
        with np.errstate(invalid="ignore"):
            coords = _conformers(topo, base, p_idx[sl], chi[sl])
        ok = _clash_free(coords, topo)
        ok_all[sl] = ok
        n_nonfinite += int((~np.isfinite(coords).all(axis=(1, 2))).sum())
        if not ok.any():
            continue
        good = coords[ok]
        rows = np.flatnonzero(ok) + s
        for r in radii:
            a = batch_target_sasa(good, topo.radii, topo.target, r, cfg.n_points, dtype)
            total[r][rows] = a.sum(axis=1)
            if has_side:
                side[r][rows] = a[:, topo.target_is_side].sum(axis=1)

    result = ResidueScan(resname, M, int((~ok_all).sum()), n_nonfinite)
    # refinement: re-evaluate the top candidates with many more sample points
    for r in radii:
        finite = np.flatnonzero(np.isfinite(total[r]))
        if finite.size == 0:
            raise RuntimeError(f"{resname}: every conformer was rejected as a steric clash; loosen clash_scale")
        cand = set(finite[np.argsort(total[r][finite])[-cfg.top_m:]].tolist())
        if has_side:
            fs = np.flatnonzero(np.isfinite(side[r]))
            cand |= set(fs[np.argsort(side[r][fs])[-cfg.top_m:]].tolist())
        cand = np.array(sorted(cand))
        coords = _conformers(topo, base, p_idx[cand], chi[cand])
        best_t, best_s, best_i = -1.0, -1.0, -1
        for k in range(len(cand)):
            a = batch_target_sasa(coords[k:k + 1], topo.radii, topo.target, r, cfg.n_points_refine, np.float64)[0]
            tot = float(a.sum())
            if tot > best_t:
                best_t, best_i = tot, k
            if has_side:
                best_s = max(best_s, float(a[topo.target_is_side].sum()))
        result.max_total[r] = best_t
        if has_side:
            result.max_side[r] = best_s
        result.coarse_max_total[r] = float(np.nanmax(total[r]))
        result.p99_total[r] = float(np.nanpercentile(total[r], 99))
        c = cand[best_i]
        result.argmax[r] = {"phi": float(phipsi[p_idx[c], 0]), "psi": float(phipsi[p_idx[c], 1]),
                            "chi": [float(x) for x in chi[c]]}
    result.seconds = time.time() - t0
    return result


def _scan_job(args):
    resname, radii, cfg, source = args
    return scan_residue(resname, radii, cfg, source)


def build_tripeptide_references(
    probe_radii: Sequence[float] = (1.4, 2.5, 4.0, 6.0),
    residues: Iterable[str] = STANDARD_RESIDUES,
    config: TripeptideConfig = TripeptideConfig(),
    source: Optional[RotamerSource] = None,
    n_workers: int = 1,
    name: Optional[str] = None,
    progress: Optional[Callable[[ResidueScan], None]] = None,
) -> Dict[float, ReferenceTable]:
    """Gly-X-Gly reference tables, one per probe radius, scanning each residue type once for all radii."""
    source = source or StaggeredChiGrid()
    residues = list(residues)
    radii = [float(r) for r in probe_radii]
    jobs = [(res, radii, config, source) for res in residues]
    scans: List[ResidueScan] = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for sc in ex.map(_scan_job, jobs):
                scans.append(sc)
                if progress:
                    progress(sc)
    else:
        for job in jobs:
            sc = _scan_job(job)
            scans.append(sc)
            if progress:
                progress(sc)

    try:
        import PeptideBuilder
        pb_version = getattr(PeptideBuilder, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        pb_version = "unavailable"
    tables: Dict[float, ReferenceTable] = {}
    base_name = name or f"gxg_{source.name}"
    for r in radii:
        max_asa = {sc.resname: sc.max_total[r] for sc in scans}
        side_max = {sc.resname: sc.max_side[r] for sc in scans if r in sc.max_side}
        extra = {
            "method": "Gly-X-Gly tripeptide maximum ASA after Tien et al. 2013 PLoS ONE 8:e80635",
            "geometry": f"PeptideBuilder {pb_version} default bond lengths/angles (averages over crystal structures)",
            "rotamers": source.describe(),
            "config": asdict(config),
            "flank_backbone": "PeptideBuilder defaults (phi=-120, psi=140); no OXT on the C-terminal residue",
            "per_residue": {
                sc.resname: {"n_conformers": sc.n_conformers, "n_clash_rejected": sc.n_clash_rejected,
                             "n_nonfinite_rejected": sc.n_nonfinite,
                             "coarse_max": sc.coarse_max_total[r], "p99_total": sc.p99_total[r],
                             "argmax": sc.argmax[r]} for sc in scans},
            "differs_from_paper": [
                "PeptideBuilder geometry, not the paper's own mined values",
                "staggered chi grid instead of the Dunbrack library (unless `rotamers` says otherwise)",
                "no random subsampling of rotamers", "steric filter with clash_scale",
                "this package's SASA engine and radii table instead of DSSP"],
        }
        tables[r] = ReferenceTable(
            name=f"{base_name}",
            probe_radius=r,
            max_asa=max_asa,
            sidechain_max_asa=side_max,
            provenance="Gly-X-Gly tripeptide scan (see extra)",
            radii_table=config.radii_table,
            n_points=config.n_points_refine,
            extra=extra,
        )
    return tables


def build_tripeptide_reference(probe_radius: float, **kwargs) -> ReferenceTable:
    """Single-radius convenience wrapper around `build_tripeptide_references`."""
    return build_tripeptide_references((probe_radius,), **kwargs)[float(probe_radius)]
