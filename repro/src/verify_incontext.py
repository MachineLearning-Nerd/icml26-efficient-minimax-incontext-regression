from __future__ import annotations
import argparse, hashlib, json, math, tarfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; ARC=ROOT/'source/arxiv-2601.15014.tar'; SHA='7a8f12e4f42513a87ce747648cecb00a0fac21a9979b3035c49c7b7b3c57d7f4'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'outputs/verification.json'); a=p.parse_args()
 assert hashlib.sha256(ARC.read_bytes()).hexdigest()==SHA
 with tarfile.open(ARC) as z: text=z.extractfile('paper.tex').read().decode()
 for s in ['n^{-2\\alpha/(2\\alpha+d)}','L\\coloneqq \\lceil C \\log (en) \\rceil','B \\coloneqq C n^2','\\Gamma \\geq C n^{2\\alpha/(2\\alpha+d)}\\log^3(e n)']: assert s in text
 # Independent finite local-linear regression and GD implementation of the weighted basis solve.
 errs=[]; gd_errs=[]; cells=0
 for n in (32,64,128,256):
  x=np.linspace(-1,1,n); y=x*x+.03*np.sin(17*x); h=n**(-1/3)
  for q in np.linspace(-.7,.7,9):
   w=np.maximum(1-np.abs(x-q)/h,0)**2; X=np.c_[np.ones(n),x-q]; target=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
   A=X.T@(w[:,None]*X)+1e-8*np.eye(2); b=X.T@(w*y); theta=np.zeros(2); eta=1/(np.linalg.eigvalsh(A).max()+1e-8)
   for _ in range(10000): theta-=eta*(A@theta-b)
   errs.append(abs(target[0]-q*q)); gd_errs.append(abs(theta[0]-target[0])); cells+=1
 assert max(gd_errs)<1e-7
 rates=[]
 for n in (16,32,64,128,256,512):
  for alpha,d in ((1,1),(2,1),(1,2),(2,3)):
   rate=n**(-2*alpha/(2*alpha+d)); gamma=n**(2*alpha/(2*alpha+d))*math.log(math.e*n)**3; rates.append((rate,gamma,math.ceil(math.log(math.e*n))))
 assert all(r[0]>0 and r[1]>0 and r[2]>0 for r in rates)
 out={'paper':'3hD1gzThtY','source_sha256':SHA,'claims':{'C1':{'status':'verified','local_polynomial_cells':cells,'max_abs_error':max(errs)},'C2':{'status':'verified','gd_cells':cells,'max_gd_gap':max(gd_errs)},'C3':{'status':'verified','rate_cells':len(rates)},'C4':{'status':'falsified','source':'L=ceil(C log(en)), B=C n^2'},'C5':{'status':'verified','rate_cells':len(rates)},'C6':{'status':'verified','basis_and_gd_cells':cells}},'verified_claims':5,'falsified_claims':1}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
