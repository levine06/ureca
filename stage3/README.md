# Stage 3: The information-content experiment

Stage 3 tests whether accessibility measurements carry enough information to tell a correct structure from an incorrect one. For a protein with true structure `X_true`:

1. Calculate the true synthetic accessibility: `y = f(X_true)`.
2. Generate candidate OpenFold3 structures: `X_1, ..., X_K`.
3. Calculate the predicted accessibility of each candidate: `ŷ^(k) = f(X_k)`.
4. Calculate an accessibility error for each candidate.
5. Calculate the true structural accuracy of each candidate.

The accessibility error is then plotted against the structural error for every candidate.

## Components

| Step | What | Where |
|---|---|---|
| 1 | True synthetic accessibility | [`truth_accessibility/`](truth_accessibility/README.md) |
| 2 | Candidate generation | [`candidate_generation/`](candidate_generation/README.md) |
| 3 | Candidate accessibility | [`candidate_accessibility/`](candidate_accessibility/README.md) |
| 4-5 | Accessibility error, structural accuracy | not in this folder yet |
