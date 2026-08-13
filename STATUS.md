# Status

- **Scoped reproduction:** 6/6 paper claim contracts pass.
- **Confidence:** C1, C2, C4, C5, and C6 HIGH; C3 MEDIUM.
- **Paper-level status:** INCONCLUSIVE; this audit does not solve the global
  ERM, reprove cited minimax lower bounds, or reproduce the authors' GPU
  training dynamics.
- **Pinned source:** `source/arxiv-2601.15014.tar`, SHA-256
  `7a8f12e4f42513a87ce747648cecb00a0fac21a9979b3035c49c7b7b3c57d7f4`.
- **Claim producers:** `repro/claims/`; the estimator, transformer, and risk
  code live under `repro/`.
- **Historical control:** C0 is retained as a low-confidence toy/source-anchor
  regression control and is not included in the six paper claims.
- **Historical evaluator record:** prior toy-only revision scored `2/12`; no
  new judge score is claimed here.
- **Branches:** five legacy `orx/*` roles are documented in
  [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md); final publication keeps `main` only.
