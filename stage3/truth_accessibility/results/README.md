# Development and Validation Results

`summary.csv` contains the updated cluster-run summary for 30 development and 15 validation chains.
It includes modified-residue/shadow masking and separate coordinate-record and usable coverage.
Test A and Test B were not included.

`residues.csv.gz` contains 9,224 per-residue rows with total and side-chain absolute/relative
accessibility values at all four radii, plus exclusion flags.
The 45 `.npz` files preserve full-precision arrays, masks and run metadata; use these for numerical
candidate comparisons. The CSV rounds absolute areas to three decimals and relative values to four.
All files were checked for matching entries, updated masking metadata and masked relative values.

Settings: probe radii 1.4, 2.5, 4.0 and 6.0 A; 1,000 points; gap flank zero; shadow threshold 5 A^2.
Copies remain on the cluster at:

```text
/home/mlee116/accessibility-run/accessfold/truth_dev_validation_updated/
```

Use outputs from that updated run; the earlier run predates the masking fixes.
