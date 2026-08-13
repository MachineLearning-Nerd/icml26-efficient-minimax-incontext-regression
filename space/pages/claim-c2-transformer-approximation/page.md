# Claim C2 — transformer approximation


---
<!-- trackio-cell
{"type": "markdown", "id": "C2_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: HIGH**

Built f_TF from the verified construction (exact ReLU kernel + ReLU monomial basis at width ~ n + ceil(C log(en)) attention-GD steps) and measured the pointwise RMS construction error ||f_TF - f_LocPol||_2. Its log-log slope in n is -2.432 (<= -0.9 => O(1/n)). Since |R(f_TF)-R(f_LocPol)| <= 4M*||f_TF-f_LocPol||_2 (Lipschitz, both bounded by M), the risk difference is O(1/n) as claimed. A shallow construction (fixed precision, few GD steps) plateaus, confirming O(1/n) requires the precision/GD to scale with n.

![C2 evidence](../images/fig2_construction_approx.png)

**Exact claim tested.** There exists a transformer f_TF with L=ceil(C log(en)) blocks such that |R(f_TF) - R(f_LocPol)| <= C/n.

| check | passed | detail |
|---|---|---|
| construction_error_is_O_inverse_n | PASS | pointwise RMS construction error ||f_TF - f_LocPol||_2 vs n has log-log slope -2.432 (<= -0.9), i.e. O(1/n) as required by Theorem 3.1; n=16:eps=3.31e-04, n=32:eps=5.62e-05, n=64:eps=1.04e-05, n=128:eps=2.04e-06, n=256:eps=3.79e-07 |
| risk_difference_bound_O_inverse_n | PASS | |R(f_TF)-R(f_LocPol)| <= 4M*eps(n) = O(1/n) (M=10.0); bound at n=256: 1.52e-05 vs C/n=1.95e-01 |
| negative_control_shallow_construction_fails | PASS | shallow construction (fixed width 8, 3 GD steps) plateaus (slope -0.395 > -0.5): O(1/n) needs the precision/GD to scale with n |



---
<!-- trackio-cell
{"type": "markdown", "id": "C2_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Theorem 3.1 (approximation): |R(f_TF)-R(f_LocPol)| <= C/n
- paper.tex Theorem 3.1: L = ceil(C log(en)), B = C n^2
- paper.tex Section 4: construction via ReLU FFN basis + linear-attention GD

**Executable verifier & fixed command:**
- Code: [`repro/claims/c2_transformer_approx.py`](https://github.com/MachineLearning-Nerd/icml26-efficient-minimax-incontext-regression/blob/main/repro/claims/c2_transformer_approx.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python repro/src/run_publication_gate.py`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `main` (publication surface).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 1.182.

**Raw data (downloadable CSV):**

- [`evidence/construction_error.csv`](../evidence/construction_error.csv)
- [`evidence/construction_control.csv`](../evidence/construction_control.csv)

**Limitations & deviations:**

- Measures pointwise construction error (which bounds the risk difference Lipschitzly); a direct Monte-Carlo of |R(f_TF)-R(f_LocPol)| is impractical because both risks are ~n^{-2a/(2a+d)} >> 1/n and their difference is below MC noise. Monomial precision uses width~n (error~n^{-2}); the proof's log-depth Lu-Shen-Yang-Zhang construction would give the same O(1/n) at lower width.
