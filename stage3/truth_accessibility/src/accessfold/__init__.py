"""accessfold: forward models mapping a protein structure to residue-level accessibility."""

from accessfold.structures.atomic import AtomicStructure
from accessfold.accessibility.compute import (
    AccessibilityResult,
    compute_accessibility,
    compute_accessibility_profile,
    register_method,
)
from accessfold.accessibility.reference import (
    ReferenceTable,
    build_empirical_reference,
    get_reference,
    register_reference,
)
from accessfold.accessibility.tripeptide import (
    StaggeredChiGrid,
    TripeptideConfig,
    build_tripeptide_reference,
    build_tripeptide_references,
)

__all__ = [
    "AtomicStructure",
    "AccessibilityResult",
    "compute_accessibility",
    "compute_accessibility_profile",
    "register_method",
    "ReferenceTable",
    "build_empirical_reference",
    "get_reference",
    "register_reference",
    "StaggeredChiGrid",
    "TripeptideConfig",
    "build_tripeptide_reference",
    "build_tripeptide_references",
]
__version__ = "0.1.0"
