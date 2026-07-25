# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "nc1", "created_at": "2026-07-25T00:00:00+00:00", "title": "Controls that fail as intended"}
-->
Each claim has a negative control that is expected to (and does) fail, confirming the result is not circular:

- **C1**: an intentionally too-slow bandwidth h=n^{-1/(4(2α+d))} (bias-limited) gives a clearly flatter rate than optimal in 11/12 (α,d) cells.
- **C2**: a shallow construction (fixed width 8, only 3 GD steps) plateaus (slope −0.40) — the O(1/n) rate requires precision and GD depth to scale with n.
- **C6**: the normal-matrix condition number is constant in n (κ≈200), so reducing GD steps below Θ(log n) leaves the least-squares gap above 1/n.

See each claim page's checks table for the control's PASS/FAIL and the raw CSVs.
