# Claim C4 — Θ(log n) parameters


---
<!-- trackio-cell
{"type": "markdown", "id": "C4_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: HIGH**

Counted scalar parameters directly from Definitions 2.1-2.4: total = L * per-block, per-block depends only on (d,alpha) (not n), L = ceil(C log(en)), hence total = Theta(log n). B = C n^2 is the per-entry MAGNITUDE bound (Def 2.4), NOT a count — the prior toy 'falsification' conflated these. At n=1024 our count is orders of magnitude below both Shen's Theta(n) and Kim's Theta(n^{d/(2a+d)}).

![C4 evidence](../images/fig4_param_count.png)

**Exact claim tested.** The construction requires only L = ceil(C log(en)) transformer blocks and Theta(log n) total parameters, vs Theta(n) (Shen 2025) and Theta(n^{d/(2a+d)}) (Kim 2024).

| check | passed | detail |
|---|---|---|
| source_L_equals_ceil_Clog_en | PASS | Theorem 3.2 states L := ceil(C log(en)) |
| source_B_equals_Cn2_is_magnitude_bound | PASS | Definition 2.4: B is the per-entry magnitude bound, not a count |
| L_over_log_en_is_bounded_constant | PASS | L(n)/log(en) in [1, 1+1/log(en)] (min=1.0086, max=1.1630) => L = Theta(log n) |
| per_block_count_independent_of_n | PASS | per-block param count fixed for 16 (d,alpha) cases; e.g. (d=3,alpha=3): 228480 params/block |
| ours_growth_rate_is_strictly_smallest | PASS | slope log(count)/log(n): ours~0 (sub-poly) < Kim d/(2a+d) < Shen 1; e.g. (d=3,a=3) ours=0.076, shen=1.000, kim=0.333 |
| ours_slope_decreases_toward_zero | PASS | ours slope decreases with larger n (0.099 -> 0.060) => Theta(log n), not Theta(n) |
| ours_growth_rate_below_shen_and_kim | PASS | growth slope ours(~0) < Kim d/(2a+d) < Shen 1 for d in {1,2,3} |



---
<!-- trackio-cell
{"type": "markdown", "id": "C4_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Theorem 3.2: L := ceil(C log(en))
- paper.tex Theorem 3.2: B := C n^2
- paper.tex Definition 2.4: entries bounded in absolute value by B
- paper.tex Sec 3: Theta(log n) parameters; Shen Theta(n); Kim Theta(n^{d/(2a+d)})

**Executable verifier & fixed command:**
- Code: [`repro/claims/c4_param_count.py`](https://github.com/MachineLearning-Nerd/icml26-repro-3hD1gzThtY-incontext-nonparametric/blob/orx/erm-rate-c3/repro/claims/c4_param_count.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python -m repro.run`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `orx/erm-rate-c3` @ `7b5707f` (mirrored to `main`).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 0.002.

**Raw data (downloadable CSV):**

- [`evidence/param_count_sweep.csv`](../evidence/param_count_sweep.csv)
- [`evidence/param_comparison.csv`](../evidence/param_comparison.csv)

**Limitations & deviations:**

- Comparison to Shen/Kim uses the exponents as stated in this paper; their primary sources were not independently re-derived here.
