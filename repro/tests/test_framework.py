"""Smoke tests for the reproduction framework and source integrity."""
import hashlib

from repro import common


def test_source_integrity():
    info = common.assert_source_integrity()
    assert info["tar_sha256"] == common.SOURCE_SHA256
    assert info["paper_sha256"] == common.PAPER_TEX_SHA256
    assert info["appendix_sha256"] == common.APPENDIX_TEX_SHA256


def test_source_formula_anchors():
    import tarfile

    with tarfile.open(common.SOURCE_TAR) as z:
        text = z.extractfile("paper.tex").read().decode()
    for anchor in [
        r"n^{-2\alpha/(2\alpha+d)}",
        r"L\coloneqq \lceil C \log (en) \rceil",
        r"B \coloneqq C n^2",
        r"\Gamma \geq C n^{2\alpha/(2\alpha+d)}\log^3(e n)",
    ]:
        assert anchor in text


def test_c0_runs():
    from repro.claims import c0_baseline_reference

    res = c0_baseline_reference.verify()
    assert res.claim_id == "C0"
    assert all(c.passed for c in res.checks)
