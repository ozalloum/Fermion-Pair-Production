"""Independent 100-digit checks of the exact helper; requires mpmath."""
import argparse,json
from pathlib import Path
import numpy as np
import mpmath as mp
from pairpulse import SauterPulse,sauter_exact_occupation,sauter_log_occupation

def high_precision(p,e,t,m=1):
    p,e,t,m=map(lambda x:mp.mpf(str(x)),(p,e,t,m))
    wp=mp.sqrt(m*m+(p+e*t)**2);wm=mp.sqrt(m*m+(p-e*t)**2)
    a,b=mp.pi*t*wp,mp.pi*t*wm;l=mp.pi*e*t*t
    return mp.sinh(l+(a-b)/2)*mp.sinh(l-(a-b)/2)/(mp.sinh(a)*mp.sinh(b))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    mp.mp.dps=100;cases=[]
    for e in [.1,.3,.8]:
        for t in [.5,2,5]:
            bound=max(1.5,e*t+2/t)
            cases.extend((float(p),e,t,1,'declared') for p in np.linspace(-bound,bound,13))
    for e in [1e-20,1e-16,1e-12,1e-8,1e-4,.1,1,10,1e4]:
        for t in [1e-4,.01,1,10,1e3]:
            cases.extend((p,e,t,1,'extended') for p in [0,.7,10,1000])
    cases.extend([(1e8,1,1e-9,1,'cancellation'),(1,1,1,1e-8,'small_mass')])
    rows=[]
    for p,e,t,m,group in cases:
        ref=high_precision(p,e,t,m);value=sauter_exact_occupation(p,SauterPulse(e,t),mass=m)
        logvalue=sauter_log_occupation(p,SauterPulse(e,t),mass=m)
        relative=float(abs((mp.mpf(value)-ref)/ref)) if ref>=mp.mpf('1e-300') else None
        log_relative=float(abs(mp.mpf(logvalue)-mp.log(ref))/max(1,abs(mp.log(ref))))
        rows.append(dict(group=group,p=p,E0=e,tau=t,mass=m,reference=mp.nstr(ref,40),occupation=value,
            absolute_error=float(abs(mp.mpf(value)-ref)),resolved_relative_error=relative,
            normalized_log_error=log_relative))
    summary=dict(modes=len(rows),precision_digits=100,
        declared_max_absolute_error=max(r['absolute_error'] for r in rows if r['group']=='declared'),
        max_absolute_error=max(r['absolute_error'] for r in rows),
        max_relative_error_above_1e_300=max(r['resolved_relative_error'] for r in rows if r['resolved_relative_error'] is not None),
        max_normalized_log_error=max(r['normalized_log_error'] for r in rows))
    summary['passed']=(summary['max_absolute_error']<1e-12 and summary['max_relative_error_above_1e_300']<1e-10 and summary['max_normalized_log_error']<1e-12)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analytic_modes.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (args.output/'analytic_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
    if not summary['passed']:raise SystemExit(1)
if __name__=='__main__':main()
