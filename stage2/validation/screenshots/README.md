# Screenshot evidence

The planned collection is complete: two views each for ubiquitin residues 48 and 15, crambin residues 26 and 7, and lysozyme residues 17 and 2 (12 images). See INDEX.md for provenance and image limitations. These are representative supporting views, not screenshots of all 36 observations. Earlier conversation screenshots were unavailable at their original temporary paths during consolidation. The instructions below describe how to capture additional or replacement views.

Capture at least two substantially different angles per selected anchor. Retain the complete protein context and identify the protein, chain, residue, representation, and viewing angle in filenames and notes. Screenshots illustrate recorded observations; do not change labels to fit the calculated score.

| Protein | Buried anchor | Exposed anchor |
| --- | --- | --- |
| 1UBQ chain A | 15 | 48 |
| 1CRN chain A | 26 | 7 |
| 1LYZ chain A | 17 | 2 |

Open each *_inspect.pse file explicitly, restore inspection_controls.py, then highlight an anchor. For a consistent packing view:

```pymol
inspect_residue 15
hide surface, protein
show spheres, protein
set sphere_scale, 1
orient protein
```

Substitute the appropriate residue number. Rotate and use File > Save Image As > PNG to save into this screenshots folder. Example names: 1UBQ_A_15_buried_view1.png and 1UBQ_A_15_buried_view2.png. Repeat for the exposed anchor and other proteins. A buried anchor may be hidden in the complete sphere view: save an additional transparent-surface or neighbourhood view to reveal its location, clearly noting that representation.

Optional follow-up images: lysozyme 25, 108, 98 and crambin 16, 30, using a surface view plus a packing view. Describe the visible geometry without inferring a solvent entry route from the image alone.
