"""Run the current six-claim reproduction and write a conservative gate record."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CLAIMS = ("C1", "C2", "C3", "C4", "C5", "C6")


def main() -> None:
    result = subprocess.run([sys.executable, "-m", "repro.run"], cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)

    summary = json.loads((ROOT / "outputs" / "claims.json").read_text())
    claims = {claim["claim_id"]: claim for claim in summary["claims"]}
    missing = [claim for claim in REQUIRED_CLAIMS if claim not in claims]
    if missing:
        raise AssertionError(f"missing claim results: {missing}")
    failed = [claim for claim in REQUIRED_CLAIMS if claims[claim]["status"] != "VERIFIED"]
    if failed:
        raise AssertionError(f"scoped claim contracts did not verify: {failed}")

    source = summary["source"]
    gate = {
        "paper": "3hD1gzThtY",
        "gate": "scoped_claims_passed",
        "tests_passed": True,
        "publication_gate_passed": True,
        "overall_status": "INCONCLUSIVE",
        "scoped_claims_supported": len(REQUIRED_CLAIMS),
        "scoped_claims_total": len(REQUIRED_CLAIMS),
        "paper_claims_verified": 0,
        "paper_claims_total": len(REQUIRED_CLAIMS),
        "falsified_claims": 0,
        "blocked_claims": 0,
        "current_claim_verdicts": {claim: claims[claim]["status"] for claim in REQUIRED_CLAIMS},
        "claim_confidence": {claim: claims[claim]["confidence"] for claim in REQUIRED_CLAIMS},
        "source_sha256": source["tar_sha256"],
        "evidence": "outputs/claims.json",
        "command": "bash repro/ci.sh",
        "status_note": "Six scoped claim contracts pass; the audit is not a foundational proof-assistant formalization and does not reproduce GPU training dynamics or solve the global ERM exactly.",
    }
    output = ROOT / "outputs" / "publication_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
