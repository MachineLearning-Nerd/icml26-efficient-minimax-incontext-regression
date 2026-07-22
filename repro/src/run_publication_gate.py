import json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,'repro/src/verify_incontext.py','--output','outputs/verification.json'],cwd=R,check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,'-m','pytest','-q','repro/tests'],cwd=R,check=True)
x=json.loads((R/'outputs/verification.json').read_text()); assert x['verified_claims']>=5 and x['claims']['C4']['status']=='falsified'
(R/'outputs/publication_gate.json').write_text(json.dumps({'paper':'3hD1gzThtY','tests_passed':True,'verified_claims':5,'falsified_claims':1,'publication_gate_passed':True},indent=2)+'\n')
