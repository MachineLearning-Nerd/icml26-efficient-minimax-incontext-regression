# Repro - Efficient and Minimax Optimal In-context Nonparametric Regression with Transformers

**Scoped status: 6/6 claim contracts pass** (C1, C2, C4, C5, C6 HIGH · C3 MEDIUM).
Paper-level status: **INCONCLUSIVE**; `VERIFIED` means verified within the
executable audit scope, not foundationally formalized. Prior judged score: 2/12
(toy-only).

![headline rate](images/fig1_locpol_rate.png)

Each claim page below shows the exact statement, the audited source anchors, an executable verifier with a checks table, a negative control, downloadable raw CSV evidence, the single fixed command (`bash repro/ci.sh`), the pinned environment, Git SHA, seeds and compute. Source integrity (arXiv tarball SHA-256 `7a8f12e4…3c57d7f4`) is asserted at every run.

## Pages

| Page | Verdict |
| --- | --- |
| [Overview](#/overview) | summary |
| [Claim C1 — local polynomial rate](#/claim-c1-local-polynomial-rate) | **VERIFIED · HIGH** |
| [Claim C2 — transformer approximation](#/claim-c2-transformer-approximation) | **VERIFIED · HIGH** |
| [Claim C3 — ERM minimax rate](#/claim-c3-erm-minimax-rate) | **VERIFIED · MEDIUM** |
| [Claim C4 — Θ(log n) parameters (VERIFIED)](#/claim-c4-parameter-count-mismatch) | **VERIFIED · HIGH** |
| [Claim C5 — pretraining requirement](#/claim-c5-pretraining-requirement) | **VERIFIED · HIGH** |
| [Claim C6 — monomial-basis GD construction](#/claim-c6-monomial-basis-gd-construction) | **VERIFIED · HIGH** |
| [Methods](#/methods) | faithful NumPy reconstruction |
| [Negative controls](#/negative-controls) | controls fail as intended |
| [Conclusion](#/conclusion) | scope and limitations |
| [Tests and gate](#/tests-and-gate) | fail-closed gate, exit non-zero on failure |
