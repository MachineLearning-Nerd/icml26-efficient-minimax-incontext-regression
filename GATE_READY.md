# Publication gate record

- Scoped claim contracts: `6/6` pass.
- Claim confidence: C1, C2, C4, C5, C6 `HIGH`; C3 `MEDIUM`.
- Historical C0: retained as a low-confidence toy/source-anchor control; not
  counted in the six paper claims.
- Source archive SHA-256:
  `7a8f12e4f42513a87ce747648cecb00a0fac21a9979b3035c49c7b7b3c57d7f4`.
- Paper-level status: `INCONCLUSIVE` because the audit does not solve the
  global ERM, reprove cited lower bounds, or reproduce GPU training dynamics.
- Historical evaluator score: `2/12` for the earlier toy-only revision; no
  new score claimed.

Run the gate with:

```bash
bash repro/ci.sh
```

The compact summary is written to
[`outputs/publication_gate.json`](outputs/publication_gate.json).
