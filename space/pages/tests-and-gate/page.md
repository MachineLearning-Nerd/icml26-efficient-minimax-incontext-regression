# Tests and gate


---
<!-- trackio-cell
{"type": "markdown", "id": "tg1", "created_at": "2026-07-25T00:00:00+00:00", "title": "Fail-closed gate"}
-->
The single fixed command `bash repro/ci.sh` runs `uv sync --frozen` then `uv run python -m repro.run`, which:
1. asserts the arXiv source tarball SHA-256 (`7a8f12e4…3c57d7f4`) and the per-file LaTeX hashes;
2. runs every claim verifier (`repro/claims/c*.py`);
3. writes `outputs/claims/*.json` and raw CSVs;
4. **exits non-zero** if any VERIFIED/FALSIFIED claim has a failing check.

**Latest result (Hugging Face `cpu-upgrade`, run 65493e29):** C1 VERIFIED HIGH, C2 VERIFIED HIGH, C3 VERIFIED MEDIUM, C4 VERIFIED HIGH, C5 VERIFIED HIGH, C6 VERIFIED HIGH, C0 (toy control) LOW. GATE PASSED.

Smoke tests: `uv run pytest repro/tests` (source-integrity + framework).


---
<!-- trackio-cell
{"type": "markdown", "id": "tg2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Command"}
-->
````bash
$ bash repro/ci.sh
[ci] uv ...
[ci] running repro.run
... GATE PASSED
````
exit 0 · ~30–60 s CPU
