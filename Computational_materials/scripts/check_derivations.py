"""Check the manuscript's transformations and Sauter connection coefficients.

These algebra/limit checks supplement, rather than replace, the ODE study.
"""
import argparse,json
from pathlib import Path
import numpy as np
import mpmath as mp
from pairpulse import SauterPulse,sauter_exact_occupation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('derivation-check'))
    out=ap.parse_args().output;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2202);mapping_error=0.;eigen_error=0.
    sigma=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    for _ in range(200):
        kinetic,mass,electric=rng.normal(),rng.uniform(.1,3),rng.normal()
        omega=np.hypot(mass,kinetic);theta=np.arctan2(kinetic,mass);W=electric*mass/omega**2
        n=np.array([np.sin(theta),0,np.cos(theta)]);et=np.array([np.cos(theta),0,-np.sin(theta)])
        state=rng.normal(size=2)+1j*rng.normal(size=2);state/=np.linalg.norm(state)
        s=np.array([np.vdot(state,a@state).real for a in sigma]);H=kinetic*sigma[0]+mass*sigma[2]
        ds=np.array([(1j*np.vdot(state,(H@a-a@H)@state)).real for a in sigma])
        f=(1+n@s)/2;u=et@s;v=s[1]
        transformed=np.array([(W*et@s+n@ds)/2,-W*n@s+et@ds,ds[1]])
        qke=np.array([W*u/2,W*(1-2*f)-2*omega*v,2*omega*u])
        mapping_error=max(mapping_error,float(np.max(abs(transformed-qke))))
        up=np.array([np.cos(theta/2),np.sin(theta/2)]);um=np.array([-up[1],up[0]])
        eigen_error=max(eigen_error,float(np.max(abs(H@up-omega*up))),float(np.max(abs(H@um+omega*um))))
    mp.mp.dps=80;cases=[]
    for p,E,tau,m in [('0','.3','2','1'),('1','.3','2','1'),('-1','.3','2','1'),('.7','10','3','2'),('4','.001','.1','1')]:
        p,E,tau,m=map(mp.mpf,(p,E,tau,m));a=E*tau;lam=a*tau
        wm=mp.sqrt(m*m+(p-a)**2);wp=mp.sqrt(m*m+(p+a)**2)
        alpha=1j*tau*wm/2;beta=1j*tau*wp/2
        A=alpha+beta-1j*lam;B=alpha+beta+1+1j*lam;C=1+2*alpha
        coeff=mp.gamma(C)*mp.gamma(-2*beta)/(mp.gamma(C-A)*mp.gamma(C-B))
        gamma_f=abs(coeff)**2*wp*(wm-p+a)/(wm*(wp+p+a))
        xminus=mp.pi*tau*(a-(wp-wm)/2);xplus=mp.pi*tau*(a+(wp-wm)/2)
        product=mp.sinh(xminus)*mp.sinh(xplus)/(mp.sinh(mp.pi*tau*wm)*mp.sinh(mp.pi*tau*wp))
        residuals=[]
        def chi(y):return y**alpha*(1-y)**beta*mp.hyp2f1(A,B,C,y)
        for y in map(mp.mpf,['.2','.5','.8']):
            potential=tau*tau*(m*m+(p-a+2*a*y)**2)/(4*y*y*(1-y)**2)+1j*lam/(y*(1-y))
            terms=[mp.diff(chi,y,2),(1-2*y)/(y*(1-y))*mp.diff(chi,y),potential*chi(y)]
            residuals.append(abs(sum(terms))/max(1,*map(abs,terms)))
        cases.append(dict(p=str(p),E=str(E),tau=str(tau),mass=str(m),gamma_product_relative_error=float(abs(gamma_f-product)/product),max_scalar_equation_normalized_residual=float(max(residuals))))
    p,m,tau=.7,1.,2.;omega=np.hypot(m,p)
    weak=[]
    for E in [1e-3,1e-4,1e-5]:
        exact=sauter_exact_occupation(p,SauterPulse(E,tau));leading=(np.pi*E*tau*tau)**2*m*m/(omega*omega*np.sinh(np.pi*tau*omega)**2)
        weak.append(dict(E=E,relative_difference=abs(exact-leading)/exact))
    long=[]
    for tau in [10.,100.,1000.]:
        exact=sauter_exact_occupation(.7,SauterPulse(.3,tau));limit=np.exp(-np.pi/.3)
        long.append(dict(tau=tau,relative_difference=abs(exact-limit)/limit))
    passed=(mapping_error<1e-12 and eigen_error<1e-13 and all(c['gamma_product_relative_error']<1e-60 and c['max_scalar_equation_normalized_residual']<1e-60 for c in cases) and all(weak[i+1]['relative_difference']<weak[i]['relative_difference']/90 for i in range(2)) and long[-1]['relative_difference']<3e-5)
    report=dict(passed=bool(passed),bloch_random_states=200,max_mapping_absolute_error=mapping_error,max_eigenvector_residual=eigen_error,mp_digits=80,connection_cases=cases,weak_field_limit=weak,long_pulse_limit=long)
    (out/'derivation_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
