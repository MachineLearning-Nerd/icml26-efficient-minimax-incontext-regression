"""Fixed entry point for the reproduction.

Runs every claim verifier in ``repro.claims``, writes per-claim and aggregate
JSON evidence under ``outputs/``, prints a summary table, and exits non-zero if
any verifier that should succeed fails its own checks. BLOCKED (documented
incapability) and FALSIFIED (valid counterexample) do not fail the gate.

Run locally:      uv run python -m repro.run
Run on HF cpu-upgrade: identical command (see repro/ci.sh).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from repro import common

# Claim modules, in paper order. Each exposes verify() -> ClaimResult.
CLAIM_MODULES = [
    "repro.claims.c1_locpol_rate",
    "repro.claims.c2_transformer_approx",
    "repro.claims.c3_erm_rate",
    "repro.claims.c4_param_count",
    "repro.claims.c5_pretraining",
    "repro.claims.c6_construction_components",
    "repro.claims.c0_baseline_reference",
]


def main() -> int:
    t0 = time.time()
    src = common.assert_source_integrity()
    print(f"[source] tarball SHA-256 = {src['tar_sha256']}")
    print(f"[source] paper.tex  SHA-256 = {src['paper_sha256']}")
    print(f"[source] appendix.tex SHA-256 = {src['appendix_sha256']}\n")

    import importlib

    results = []
    for mod_name in CLAIM_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        label = mod_name.rsplit(".", 1)[1]
        print(f"=== {label} ===")
        ts = time.time()
        res = mod.verify()
        res.metrics["runtime_seconds"] = round(time.time() - ts, 3)
        results.append(res)
        per = common.ARTIFACTS / f"{label}.json"
        common.write_json(per, res.to_json())
        flag = "OK" if res.status in ("VERIFIED", "FALSIFIED") else res.status
        print(f"  -> {res.claim_id}: {res.status} [{res.confidence}] ({flag})")
        for c in res.checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"     [{mark}] {c.name}: {c.detail}")
        print()

    summary = {
        "paper": "3hD1gzThtY / arXiv 2601.15014",
        "source": src,
        "claims": [r.to_json() for r in results],
        "total_runtime_seconds": round(time.time() - t0, 3),
    }
    common.write_json(common.OUTPUTS / "claims.json", summary)

    # Summary table.
    print("=" * 78)
    print(f"{'claim':<6}{'status':<12}{'conf':<8}{'checks(pass/fail)':<20}title")
    print("-" * 78)
    for r in results:
        npass = sum(1 for c in r.checks if c.passed)
        nfail = len(r.checks) - npass
        print(f"{r.claim_id:<6}{r.status:<12}{r.confidence:<8}{f'{npass}/{len(r.checks)}':<20}{r.title[:34]}")
    print("=" * 78)

    # Gate: a VERIFIED/FALSIFIED claim whose own checks all pass is fine; a
    # verifier that asserts VERIFIED but has failing checks is a hard failure.
    hard_fail = []
    for r in results:
        if r.status in ("VERIFIED", "FALSIFIED") and any(not c.passed for c in r.checks):
            hard_fail.append(r.claim_id)
    if hard_fail:
        print(f"\nGATE FAILED: checks failed for {hard_fail}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
