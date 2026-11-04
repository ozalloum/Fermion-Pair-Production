"""Derive compact tables from retained study-validation mode records."""
import argparse,csv,json,math
from collections import Counter
from validation_protocol import PROTOCOL, JOB_FIELDS, NUMERIC_JOB_FIELDS, jobs, job_id, pulse_from_spec
from pathlib import Path
import numpy as np
from pairpulse import SauterPulse,sauter_exact_occupation,integrated_density

def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def validate_records(raw_rows):
    """Fail closed on missing, duplicate, malformed or inconsistent evidence."""
    expected={r["job_id"]:r for r in jobs()}
    counts=Counter(r.get("job_id") for r in raw_rows)
    errors=[]
    missing=set(expected)-set(counts);extra=set(counts)-set(expected)
    if missing: errors.append(f"missing {len(missing)} expected jobs")
    if extra: errors.append(f"unexpected job identifiers: {len(extra)}")
    if any(n != 1 for n in counts.values()): errors.append("duplicate job identifiers")
    parsed=[]
    for index,raw in enumerate(raw_rows,1):
        try:
            r=dict(raw)
            if r.get("status") != "ok": raise ValueError("integration status is not ok")
            for k in NUMERIC_JOB_FIELDS:
                r[k]=float(r[k])
                if not math.isfinite(r[k]): raise ValueError(f"nonfinite {k}")
            if r["job_id"] not in expected: raise ValueError("unknown job")
            if job_id(r) != r["job_id"]: raise ValueError("job metadata or pulse specification mismatch")
            if any(r[k] != expected[r["job_id"]][k] for k in JOB_FIELDS):
                raise ValueError("unexpected job parameters")
            for k in ("occupation","diagnostic","nfev","t0","tf","wall_seconds"):
                r[k]=float(r[k])
                if not math.isfinite(r[k]): raise ValueError(f"nonfinite {k}")
            if r["diagnostic"] < 0 or r["wall_seconds"] < 0: raise ValueError("negative diagnostic or wall time")
            if r["nfev"] != int(r["nfev"]) or (r["nfev"] <= 0 and not(r["method"]=="qutip_vern9" and r["nfev"]==-1)):
                raise ValueError("invalid evaluation count")
            tol=PROTOCOL["occupation_absolute_tolerance"]
            if not -tol <= r["occupation"] <= 1+tol: raise ValueError("occupation outside fermionic bounds")
            pulse=pulse_from_spec(r["pulse_spec"])
            if not np.allclose([r["t0"],r["tf"]],pulse.time_window(r["tail"]),rtol=0,atol=1e-12):
                raise ValueError("endpoints inconsistent with pulse specification")
            # Never use precomputed error columns as the source of acceptance.
            exact=sauter_exact_occupation(r["p"],pulse,mass=r["mass"],charge=r["charge"]) if r["case"]=="single" else None
            error=abs(r["occupation"]-exact) if exact is not None else None
            relative=error/exact if exact is not None and exact>=PROTOCOL["resolved_occupation_floor"] else None
            for k,new in (("exact",exact),("absolute_error",error),("resolved_relative_error",relative)):
                old=r.get(k)
                if new is None:
                    if old not in (None,""): raise ValueError(f"unexpected {k}")
                else:
                    old=float(old)
                    if not math.isfinite(old): raise ValueError(f"nonfinite {k}")
                    allowance=64*np.finfo(float).eps*max(abs(exact),abs(r["occupation"]),1e-300)
                    if k=="resolved_relative_error": allowance/=exact
                    if not math.isclose(old,new,rel_tol=1e-12,abs_tol=allowance):
                        raise ValueError(f"inconsistent recorded {k}")
                r[k]=new
            parsed.append(r)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            errors.append(f"row {index}: {exc}")
    if errors: raise ValueError("; ".join(errors))
    return parsed


def summarize(data):
    data=Path(data)
    # Invalidate a previous success before any reads or calculations. If an
    # unexpected exception interrupts this invocation, it cannot leave a green
    # summary from an earlier run in place.
    (data/'summary.json').write_text(json.dumps({
        "criteria_passed":False,
        "validation_errors":["Validation has not completed for this invocation."]
    },indent=2)+'\n')
    raw=[]
    try:
        with (data/'mode_results.csv').open(encoding='utf-8',newline='') as f:
            raw=list(csv.DictReader(f,strict=True))
        recorded=json.loads((data/'protocol.json').read_text())
        if not isinstance(recorded,dict):
            raise ValueError("protocol.json must contain a JSON object")
        if any(recorded.get(k) != v for k,v in PROTOCOL.items()):
            raise ValueError("protocol.json does not match the versioned validation protocol")
        rows=validate_records(raw)
    except (ValueError,OSError,csv.Error) as exc:
        result={"criteria_passed":False,"completed_evaluations":len(raw),
                "expected_evaluations":len(jobs()),"validation_errors":[str(exc)]}
        (data/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    failures=[]
    def select(**kw): return [r for r in rows if r['status']=='ok' and all(r[k]==v for k,v in kw.items())]
    def worst(rs,k): return max((r[k] for r in rs if r[k] is not None),default=None)
    integration=[]
    for case in ('single','double'):
        for method in ('dirac','qke'):
            rs=select(study='integration',case=case,method=method)
            ref={r['p']:r['occupation'] for r in rs if r['rtol']==2e-12 and r['coefficient']==.045}
            for tol in (2e-8,2e-10,2e-12):
                for c in (.18,.09,.045):
                    part=[r for r in rs if r['rtol']==tol and r['coefficient']==c]
                    integration.append(dict(case=case,method=method,rtol=tol,coefficient=c,modes=len(part),
                        max_reference_difference=max(abs(r['occupation']-ref[r['p']]) for r in part),
                        max_exact_error=worst(part,'absolute_error'),max_diagnostic=worst(part,'diagnostic')))
    save(data/'integration_summary.csv',integration)
    windows=[]
    for case in ('single','double'):
        for method in ('dirac','qke'):
            rs=select(study='window',case=case,method=method)
            ref={r['p']:r['occupation'] for r in rs if r['tail']==20}
            for tail in (12,16,20):
                part=[r for r in rs if r['tail']==tail]
                windows.append(dict(case=case,method=method,tail=tail,max_reference_difference=max(abs(r['occupation']-ref[r['p']]) for r in part),max_exact_error=worst(part,'absolute_error')))
    save(data/'window_summary.csv',windows)
    params=[]
    for e in (.1,.3,.8):
        for t in (.5,2.,5.):
            rs=select(study='parameter',E0=e,tau=t)
            for method in ('dirac','qke'):
                part=[r for r in rs if r['method']==method]
                params.append(dict(E0=e,tau=t,method=method,modes=len(part),
                    pmax=max(abs(r['p']) for r in part),min_exact=min(r['exact'] for r in part),
                    max_exact=max(r['exact'] for r in part),max_absolute_error=worst(part,'absolute_error'),
                    resolved_modes=sum(r['resolved_relative_error'] is not None for r in part),
                    max_resolved_relative_error=worst(part,'resolved_relative_error'),
                    negative_occupations=sum(r['occupation']<0 for r in part),
                    out_of_bounds_beyond_1e_10=sum(r['occupation']<-1e-10 or r['occupation']>1+1e-10 for r in part),
                    max_diagnostic=worst(part,'diagnostic')))
    save(data/'parameter_summary.csv',params)
    software=[]
    for case in ('single','double'):
        rs=select(study='software',case=case)
        ref={r['p']:r['occupation'] for r in rs if r['method']=='qutip_vern9'}
        for method in ('dirac','qke','qutip_vern9'):
            part=[r for r in rs if r['method']==method]
            software.append(dict(case=case,method=method,modes=len(part),
                max_qutip_difference=max(abs(r['occupation']-ref[r['p']]) for r in part),
                max_exact_error=worst(part,'absolute_error'),max_diagnostic=worst(part,'diagnostic')))
    save(data/'software_summary.csv',software)
    quad=[];control=[]
    for case,base in [('single',1.5),('double',1.8)]:
        rs=sorted(select(study='quadrature',case=case),key=lambda r:r['p'])
        p=np.array([r['p'] for r in rs]);f=np.array([r['occupation'] for r in rs])
        for bound,n in [(base,41),(base,81),(base,161),(2*base,321),(3*base,481)]:
            target=np.linspace(-bound,bound,n)
            indices=[int(np.argmin(abs(p-x))) for x in target]
            assert np.max(abs(p[indices]-target))<1e-12
            num=integrated_density(target,f[indices])
            exact=integrated_density(target,[sauter_exact_occupation(x,SauterPulse(.3,2)) for x in target]) if case=='single' else None
            quad.append(dict(case=case,pmax=bound,points=n,spacing=2*bound/(n-1),numerical_density=num,exact_grid_density=exact,
                             ode_density_difference=abs(num-exact) if exact is not None else None))
        for tail,coef in [(12,.09),(16,.09),(20,.045)]:
            cr=sorted(select(study='yield_controls',case=case,tail=float(tail),coefficient=coef),key=lambda r:r['p'])
            density=integrated_density([r['p'] for r in cr],[r['occupation'] for r in cr])
            ref=next(r['numerical_density'] for r in quad if r['case']==case and r['points']==81)
            control.append(dict(case=case,tail=tail,coefficient=coef,points=81,density=density,relative_reference_difference=abs(density-ref)/abs(ref)))
    save(data/'quadrature_summary.csv',quad);save(data/'yield_control_summary.csv',control)
    custom=select(study='custom');customdiff=[]
    for p in [-1.5,-.75,0,.75,1.5]:
        r=[r for r in custom if r['p']==p];ref=next(v['occupation'] for v in r if v['tail']==8 and v['coefficient']==.045 and v['method']=='qutip_vern9')
        customdiff.extend(abs(v['occupation']-ref) for v in r)
    summary={'expected_evaluations':len(jobs()),'global_finite_check':True,'global_bound_violations':0,'job_coverage_passed':True,'completed_evaluations':len(rows),'failed_evaluations':len(failures),
        'parameter_max_absolute_error':max(r['max_absolute_error'] for r in params),
        'parameter_max_resolved_relative_error':max(r['max_resolved_relative_error'] for r in params if r['max_resolved_relative_error'] is not None),
        'parameter_negative_occupations':sum(r['negative_occupations'] for r in params),
        'parameter_bound_violations':sum(r['out_of_bounds_beyond_1e_10'] for r in params),
        'integration_max_refinement_difference':max(r['max_reference_difference'] for r in integration),
        'software_max_difference':max(r['max_qutip_difference'] for r in software),
        'gaussian_max_difference_to_refined_qutip':max(customdiff),
        'yield_control_max_relative_difference':max(r['relative_reference_difference'] for r in control)}
    for case in ('single','double'):
        vals=[r for r in quad if r['case']==case]
        summary[case+'_fixed_grid_last_relative_change']=abs(vals[2]['numerical_density']-vals[1]['numerical_density'])/vals[2]['numerical_density']
        summary[case+'_last_domain_relative_change']=abs(vals[4]['numerical_density']-vals[3]['numerical_density'])/vals[4]['numerical_density']
        summary[case+'_final_density']=vals[4]['numerical_density']
    summary['window_max_refinement_difference']=max(r['max_reference_difference'] for r in windows)
    summary['global_max_diagnostic']=max(r['diagnostic'] for r in rows)
    summary['all_single_max_absolute_error']=max(r['absolute_error'] for r in rows if r['absolute_error'] is not None)
    summary['all_single_max_resolved_relative_error']=max(r['resolved_relative_error'] for r in rows if r['resolved_relative_error'] is not None)
    matched={}
    for r in rows:
        key=tuple(r[k] for k in JOB_FIELDS if k!='method')
        matched.setdefault(key,{})[r['method']]=r['occupation']
    summary['matched_formulation_max_difference']=max(abs(v['dirac']-v['qke']) for v in matched.values() if 'dirac' in v and 'qke' in v)
    absolute=PROTOCOL['occupation_absolute_tolerance'];relative=PROTOCOL['yield_relative_tolerance']
    checks={
        'complete_finite_bounded_records':len(rows)==len(jobs()) and not failures,
        'sauter_absolute':summary['all_single_max_absolute_error'] < absolute,
        'sauter_resolved_relative':summary['all_single_max_resolved_relative_error'] < PROTOCOL['resolved_relative_tolerance'],
        'diagnostics':summary['global_max_diagnostic'] < PROTOCOL['diagnostic_tolerance'],
        'formulations':summary['matched_formulation_max_difference'] < absolute,
        'integration':summary['integration_max_refinement_difference'] < absolute,
        'endpoints':summary['window_max_refinement_difference'] < absolute,
        'qutip':summary['software_max_difference'] < absolute,
        'gaussian':summary['gaussian_max_difference_to_refined_qutip'] < absolute,
        'density_time_controls':summary['yield_control_max_relative_difference'] < relative,
    }
    for case in ('single','double'):
        checks[case+'_spacing']=summary[case+'_fixed_grid_last_relative_change'] < relative
        checks[case+'_final_domain']=summary[case+'_last_domain_relative_change'] < relative
    summary['checks']=checks
    summary['criteria_passed']=bool(all(checks.values()))
    (data/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('validation'));args=ap.parse_args()
    result=summarize(args.data)
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['criteria_passed']: raise SystemExit(1)

if __name__=='__main__': main()
