import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def test_five_claim_gate():
 subprocess.run([sys.executable,'repro/src/verify_incontext.py','--output','outputs/test.json'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 r=json.loads((ROOT/'outputs/test.json').read_text())
 assert r['verified_claims']==5 and r['claims']['C4']['status']=='falsified'
def test_gradient_certificate():
 r=json.loads((ROOT/'outputs/verification.json').read_text())
 assert r['claims']['C2']['max_gd_gap']<1e-7 and r['claims']['C1']['local_polynomial_cells']==36
