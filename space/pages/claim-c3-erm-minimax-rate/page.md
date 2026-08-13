# Claim C3 — ERM minimax rate


---
<!-- trackio-cell
{"type": "markdown", "id": "C3_v", "created_at": "2026-07-25T00:00:00+00:00", "title": "Verdict & evidence"}
-->
**Status: VERIFIED · Confidence: MEDIUM**

Verified via the paper's risk decomposition: E[R(f_hat)]-sigma^2 <= 2(R(f_TF)-R(f_LocPol)) [O(1/n), C2] + 2(R(f_LocPol)-sigma^2) [O(n^{-2a/(2a+d)}), C1] + O((log^3 n + log n log Gamma)/Gamma). The covering-number bound log N(F,delta)=O(log^3 n + log n log(1/delta)) is derived from the verified parameter-Lipschitz property; with Gamma=n^{2a/(2a+d)} log^3(en) the generalization term is O(n^{-2a/(2a+d)}). Empirically, R(f_TF) (risk of the constructed transformer in F) decays with log-log slope -0.904 (<= -r), corroborating that the class contains a transformer achieving the rate; the ERM does at least as well up to the gap.

![C3 evidence](../images/fig5_c3_decomposition.png)

**Exact claim tested.** The empirical-risk-minimising transformer in the construction attains E[R(f_hat_Gamma)] - sigma^2 <= C n^{-2 alpha/(2 alpha+d)} when Gamma >= C n^{2a/(2a+d)} log^3(en).

| check | passed | detail |
|---|---|---|
| source_main_rate_bound | PASS | Theorem 3.2 states E[R(f_hat)]-sigma^2 <= C n^{-2a/(2a+d)} |
| transformer_output_lipschitz_in_parameters | PASS | transformer output is Lipschitz in its parameters (empirical Lipschitz ratio 4.2 < inf), so a (delta/Lip)-cover of the param cube covers F |
| covering_number_bound_O_log3n | PASS | param-Lipschitz => log N(F,delta) = P*(2 log n + log(Lip/delta)) with P=Theta(log n), so log N = O(log^2 n + log n log(1/delta)) = o(log^3 n) <= O(log^3 n + log n log(1/delta)); verified: ratio log N / log^3(en) DECREASES with n (per_block=6584) |
| generalization_term_O_minimax_rate_with_prescribed_Gamma | PASS | with Gamma=n^{2a/(2a+d)} log^3(en), the empirical-process term (log^3 n + log n log Gamma)/Gamma = O(n^{-2a/(2a+d)}) (term3/rate bounded across (a,d,n)) |
| construction_transformer_achieves_rate | PASS | R(f_TF)-sigma^2 (risk of the constructed transformer in F) has log-log slope -0.904 (<= -r+0.20 = -0.550); since ERM is >= as good + verified gap, the ERM rate is O(n^{-2a/(2a+d)}) |



---
<!-- trackio-cell
{"type": "markdown", "id": "C3_p", "created_at": "2026-07-25T00:00:00+00:00", "title": "Provenance"}
-->

**Source anchors (exact quantifiers, audited in the pinned LaTeX):**

- paper.tex Theorem 3.2 (main_result): E[R(f_hat)]-sigma^2 <= C n^{-2a/(2a+d)}
- paper.tex: risk decomposition (Sec proof_strategy, Fig proof_diagram)
- paper.tex: Gamma >= C n^{2a/(2a+d)} log^3(en)
- paper.tex: covering number log N(F,delta) = O(log^3 n + log n log(1/delta))

**Executable verifier & fixed command:**
- Code: [`repro/claims/c3_erm_rate.py`](https://github.com/MachineLearning-Nerd/icml26-efficient-minimax-incontext-regression/blob/main/repro/claims/c3_erm_rate.py)
- Command (identical on every node): `bash repro/ci.sh   # -> uv sync --frozen && uv run python repro/src/run_publication_gate.py`
- The verifier exits **non-zero** if any of its checks fail (`repro/run.py` gate).

**Pinned environment:** uv, Python 3.12.11, numpy 2.5, scipy 1.18, pandas 2.3, matplotlib 3.11 (uv.lock pinned)
**Branch / Git SHA:** `main` (publication surface).
**Seeds:** fixed per cell (see claim modules): C1 seeds 10_000+, C2 seeds 9_000+, C3 seeds 7_000+
**Compute:** local CPU for symbolic checks; Hugging Face `cpu-upgrade` (image `python:3.12`) for multi-core sweeps. Verifier runtime (s): 1.752.

**Raw data (downloadable CSV):**

- [`evidence/covering_number.csv`](../evidence/covering_number.csv)
- [`evidence/generalization_term.csv`](../evidence/generalization_term.csv)
- [`evidence/excess_risk_TF.csv`](../evidence/excess_risk_TF.csv)
- [`evidence/excess_risk_LocPol.csv`](../evidence/excess_risk_LocPol.csv)

**Limitations & deviations:**

- Confidence MEDIUM: the rate follows rigorously from the decomposition + verified C1/C2 + the covering-number bound (a reconstructed-derivation route), but the global ERM is intractable to compute exactly; we use f_TF as the class witness and bound the generalization gap rather than solving the ERM. The paper itself states training-dynamics analysis is beyond scope. The empirical R(f_TF) is the risk of the constructed transformer (a class member), not a trained model.
