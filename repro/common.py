"""Shared utilities: deterministic seeding, paths, source-integrity check, IO.

The source tarball ``source/arxiv-2601.15014.tar`` is the pinned primary source
for every claim. Its SHA-256 is asserted at the start of every run so that all
evidence is bound to an exact, reproducible paper text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SOURCE_TAR = ROOT / "source" / "arxiv-2601.15014.tar"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "outputs" / "claims"

# SHA-256 of the pinned arXiv source tarball (paper.tex + appendix.tex + figures).
SOURCE_SHA256 = "7a8f12e4f42513a87ce747648cecb00a0fac21a9979b3035c49c7b7b3c57d7f4"
# Per-file hashes of the extracted LaTeX (independent anchors).
PAPER_TEX_SHA256 = "abbe2a52502fb09711c8295d665ff2e12338ca62106e2c666f422380e71b9331"
APPENDIX_TEX_SHA256 = "711afe3ec3b826dd219ead3cc97f1269352d237125c0bd94cc4902e9c1405839"


def assert_source_integrity() -> dict:
    """Verify the pinned source tarball and extracted LaTeX by SHA-256."""
    tar_bytes = SOURCE_TAR.read_bytes()
    got = hashlib.sha256(tar_bytes).hexdigest()
    if got != SOURCE_SHA256:
        raise AssertionError(f"source tarball SHA mismatch: {got} != {SOURCE_SHA256}")
    import tarfile

    with tarfile.open(SOURCE_TAR) as z:
        paper = z.extractfile("paper.tex").read()
        appendix = z.extractfile("appendix.tex").read()
    ph = hashlib.sha256(paper).hexdigest()
    ah = hashlib.sha256(appendix).hexdigest()
    if ph != PAPER_TEX_SHA256:
        raise AssertionError(f"paper.tex SHA mismatch: {ph}")
    if ah != APPENDIX_TEX_SHA256:
        raise AssertionError(f"appendix.tex SHA mismatch: {ah}")
    return {"tar_sha256": SOURCE_SHA256, "paper_sha256": ph, "appendix_sha256": ah}


# --- deterministic seeding --------------------------------------------------
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- claim result + IO ------------------------------------------------------
@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        self.passed = bool(self.passed)


@dataclass
class ClaimResult:
    claim_id: str
    title: str
    statement: str
    status: str  # VERIFIED | FALSIFIED | BLOCKED
    confidence: str  # HIGH | MEDIUM | LOW
    summary: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    evidence_files: list[str] = field(default_factory=list)
    source_anchors: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) if isinstance(c, Check) else c for c in self.checks]
        return _to_native(d)


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-serialisable Python."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def claim_dir(claim_id: str) -> Path:
    p = ARTIFACTS / claim_id
    p.mkdir(parents=True, exist_ok=True)
    return p
