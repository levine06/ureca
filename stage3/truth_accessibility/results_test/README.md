# Test A and Test B Results

`summary.csv` contains the summary for the 40 test chains (20 Test A and 20 Test B), and `residues.csv.gz`
the per-residue rows. The 40 `.npz` files preserve full-precision arrays, masks and run metadata; use these for
numerical candidate comparisons.

Settings are the same as for the [development and validation results](../results/README.md): probe radii
1.4, 2.5, 4.0 and 6.0 A; 1,000 points; gap flank zero; shadow threshold 5 A^2.

Coverage: 39 of 40 chains pass 90% usable-coordinate coverage. 8EXF_B has 167 of 217 usable positions (77%),
with a longest internal gap of 19 residues. 7VKC has SEP at position 59 and four shadow exclusions, 8WZY has
KCX at position 150 and one, and 9YP0 has OCS at position 100. 7T1M, 7TFQ, 7TLV, 7VKC, 7XCD, 7XWY and 7Y19
contain accepted selenomethionine.

Regenerate from `stage3/truth_accessibility/` with:

```bash
python compute_truth_accessibility.py \
  --csv ../../stage1/datasets/test_a.csv ../../stage1/datasets/test_b.csv \
  --cache-dir cif_cache --out results_test --offline --workers 4
```
