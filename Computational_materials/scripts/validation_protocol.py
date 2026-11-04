"""Versioned validation jobs and complete pulse specifications."""
import hashlib,json
import numpy as np
from scipy.special import erf
from pairpulse import SauterPulse,PulseTrain

PROTOCOL = {
    "version": 2,
    "occupation_absolute_tolerance": 1e-10,
    "resolved_occupation_floor": 1e-10,
    "resolved_relative_tolerance": 1e-5,
    "yield_relative_tolerance": 1e-4,
    "diagnostic_tolerance": 1e-8,
    "feature_step_fraction": 0.125,
    "interpretation": "Empirical checks, not certified error bounds. Refined numerical runs are not exact.",
    "integration": {"tail": 20, "rtol": [2e-8,2e-10,2e-12], "atol_over_rtol": .01,
                    "step_coefficients": [.18,.09,.045], "momenta": [-1.5,-.75,0,.75,1.5]},
    "window": {"tails": [12,16,20], "rtol": 2e-12, "coefficient": .045},
    "parameters": {"amplitudes": [.1,.3,.8], "durations": [.5,2,5],
                   "momenta": "13 equally spaced in +/- max(1.5,E0*tau+2/tau)",
                   "tail":20,"rtol":2e-12,"coefficient":.09},
    "quadrature": {"single_bounds": [1.5,3,4.5], "double_bounds":[1.8,3.6,5.4],
                   "fixed_domain_counts":[41,81,161],"extended_counts":[321,481],
                   "tail":20,"rtol":2e-12,"coefficient":.09},
    "qutip": {"method":"vern9", "normalize_output":False,"tail":20,"rtol":2e-12,
              "coefficient":.09,"modes_per_case":41},
    "custom_pulse": {"E":"0.3 exp(-(t/2)^2)","A":"-0.3 sqrt(pi) erf(t/2)",
                     "momenta":[-1.5,-.75,0,.75,1.5],"tails":[6,8],"coefficients":[.09,.045]},
}

class GaussianPulse:
    def __init__(self, amplitude=.3, duration=2., center=0.):
        self.amplitude, self.duration, self.center = amplitude, duration, center
    def field(self,t):
        return self.amplitude*np.exp(-((np.asarray(t)-self.center)/self.duration)**2)
    def potential(self,t):
        return -self.amplitude*self.duration*np.sqrt(np.pi)/2*erf((np.asarray(t)-self.center)/self.duration)
    def time_window(self,tail_factor=12):
        return self.center-self.duration*tail_factor,self.center+self.duration*tail_factor
    def resolution_scale(self): return self.duration

def pulse_spec(case,e0=.3,tau=2.):
    if case == "single": return {"kind":"sauter","amplitude":e0,"duration":tau,"center":0.}
    if case == "double": return {"kind":"train","pulses":[
        {"kind":"sauter","amplitude":.3,"duration":1.1,"center":-3.5},
        {"kind":"sauter","amplitude":.3,"duration":1.1,"center":3.5}]}
    if case == "gaussian": return {"kind":"gaussian","amplitude":.3,"duration":2.,"center":0.}
    raise ValueError(f"unknown pulse case: {case}")

def pulse_from_spec(spec):
    if isinstance(spec,str): spec=json.loads(spec)
    if spec["kind"] == "sauter": return SauterPulse(spec["amplitude"],spec["duration"],spec["center"])
    if spec["kind"] == "train": return PulseTrain(pulse_from_spec(p) for p in spec["pulses"])
    if spec["kind"] == "gaussian": return GaussianPulse(spec["amplitude"],spec["duration"],spec["center"])
    raise ValueError("unknown pulse kind")

def pulse_for(case,e0=.3,tau=2):
    return pulse_from_spec(pulse_spec(case,e0,tau))

JOB_FIELDS=("study","case","p","method","tail","rtol","coefficient","E0","tau","mass","charge","pulse_spec")
NUMERIC_JOB_FIELDS=("p","tail","rtol","coefficient","E0","tau","mass","charge")
def job_id(job):
    normalized={k:float(job[k]) if k in NUMERIC_JOB_FIELDS else job[k] for k in JOB_FIELDS}
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def jobs():
    out=[]
    def add(study,case,p,method,tail=20,rtol=2e-12,coef=.09,e0=.3,tau=2):
        spec = pulse_spec(case, e0, tau)
        if case == "double": e0, tau = .3, 1.1
        job = dict(study=study,case=case,p=float(p),method=method,tail=tail,
                   rtol=rtol,coefficient=coef,E0=e0,tau=tau,mass=1.,charge=1.,
                   pulse_spec=json.dumps(spec, sort_keys=True, separators=(",", ":")))
        job["job_id"] = job_id(job)
        out.append(job)
    for case in ["single","double"]:
        for p in PROTOCOL["integration"]["momenta"]:
            for method in ["dirac","qke"]:
                for tol in PROTOCOL["integration"]["rtol"]:
                    for coef in [.18,.09,.045]: add("integration",case,p,method,rtol=tol,coef=coef)
                for tail in [12,16,20]: add("window",case,p,method,tail=tail,coef=.045)
        bound=1.5 if case=="single" else 1.8
        for p in np.linspace(-bound,bound,41):
            for method in ["dirac","qke","qutip_vern9"]: add("software",case,p,method)
        # Entire coarse grid also checks integrated effects of endpoint and cap changes.
        for p in np.linspace(-bound,bound,81):
            for tail,coef in [(12,.09),(16,.09),(20,.045)]:
                add("yield_controls",case,p,"qke",tail=tail,coef=coef)
    for e0 in [.1,.3,.8]:
        for tau in [.5,2,5]:
            bound=max(1.5,e0*tau+2/tau)
            for p in np.linspace(-bound,bound,13):
                for method in ["dirac","qke"]: add("parameter", "single",p,method,e0=e0,tau=tau)
    # A nested grid permits exact subsampling at unchanged nodes; no assumed symmetry.
    for case,bound in [("single",4.5),("double",5.4)]:
        for p in np.linspace(-bound,bound,481): add("quadrature",case,p,"qke")
    for p in [-1.5,-.75,0,.75,1.5]:
        for tail in [6,8]:
            for coef in [.09,.045]:
                for method in ["dirac","qke","qutip_vern9"]:
                    add("custom", "gaussian",p,method,tail=tail,coef=coef)
    return out
