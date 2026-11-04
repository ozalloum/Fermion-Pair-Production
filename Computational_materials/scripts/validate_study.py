"""Run the declared numerical validation; retain individual successes and failures.

Install PairPulse and requirements-validation.txt first. Run from any directory:
python scripts/validate_study.py --output validation --workers 4
No archived result file is overwritten. QuTiP is a validation-only dependency.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as im
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import subprocess
from datetime import datetime, timezone
import numpy as np
from scipy.special import erf
from pairpulse import SauterPulse, PulseTrain, solve_dirac_mode, solve_qke_mode, sauter_exact_occupation

from validation_protocol import PROTOCOL, jobs, pulse_from_spec, pulse_for


def qutip_mode(p,pulse,tail,rtol,coef):
    import qutip as qt
    t0,tf=pulse.time_window(tail)
    # Eigensystems and time evolution are built in QuTiP, without PairPulse's
    # RHS or eigenvector helper. The prescribed analytic potential is shared.
    h0=qt.sigmaz()+(p-float(pulse.potential(t0)))*qt.sigmax()
    hf=qt.sigmaz()+(p-float(pulse.potential(tf)))*qt.sigmax()
    initial=h0.eigenstates()[1][0]
    positive=hf.eigenstates()[1][-1]
    samples=np.linspace(t0,tf,257)
    cap=min(coef/np.sqrt(1+np.max(np.abs(p-pulse.potential(samples)))**2),
            pulse.resolution_scale()/8.)
    h=[qt.sigmaz()+p*qt.sigmax(),[qt.sigmax(),lambda t: -float(pulse.potential(t))]]
    sol=qt.sesolve(h,initial,[t0,tf],options={"method":"vern9","rtol":rtol,
        "atol":rtol/100,"max_step":cap,"nsteps":1000000,
        "normalize_output":False,"store_states":True,"progress_bar":""})
    state=sol.states[-1]
    return float(abs(positive.overlap(state))**2),float(abs(state.norm()**2-1)),-1,(t0,tf)

def run_job(job):
    row=dict(job); start=time.perf_counter()
    try:
        pulse=pulse_from_spec(job["pulse_spec"])
        args=dict(tail_factor=job["tail"],rtol=job["rtol"],atol=job["rtol"]/100,
                  max_step_coefficient=job["coefficient"])
        if job["method"]=="qutip_vern9":
            value,diag,nfev,span=qutip_mode(job["p"],pulse,job["tail"],job["rtol"],job["coefficient"])
        else:
            fun=solve_dirac_mode if job["method"]=="dirac" else solve_qke_mode
            result=fun(job["p"],pulse,**args)
            value,diag,nfev,span=result.occupation,result.diagnostic,result.nfev,result.time_span
        if not np.isfinite([value,diag,*span]).all():
            raise RuntimeError("nonfinite integration result")
        if not -PROTOCOL["occupation_absolute_tolerance"] <= value <= 1+PROTOCOL["occupation_absolute_tolerance"]:
            raise RuntimeError("occupation outside fermionic bounds")
        exact=sauter_exact_occupation(job["p"],pulse) if job["case"]=="single" else None
        error=abs(value-exact) if exact is not None else None
        relative=error/exact if exact is not None and exact>=PROTOCOL["resolved_occupation_floor"] else None
        row.update(occupation=value,diagnostic=diag,nfev=nfev,t0=span[0],tf=span[1],exact=exact,
                   absolute_error=error,resolved_relative_error=relative,status="ok",failure="")
    except Exception as exc:
        row.update(status="failed",failure=f"{type(exc).__name__}: {exc}")
    row["wall_seconds"]=time.perf_counter()-start
    return row


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path)
    ap.add_argument("--worker-input",type=Path);ap.add_argument("--worker-output",type=Path)
    ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
    if args.worker_input:
        rows=[]
        for job in json.loads(args.worker_input.read_text()):
            rows.append(run_job(job))
            if len(rows)%25==0: write_csv(args.worker_output,rows)
        write_csv(args.worker_output,rows)
        return
    if args.output is None or args.workers<1: ap.error("--output and positive --workers are required")
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"mode_results.csv").exists():
        raise SystemExit("Use a fresh output directory to preserve existing evidence.")
    protocol=dict(PROTOCOL,declared_at_utc=datetime.now(timezone.utc).isoformat())
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    todo=jobs();rows=[];started=time.perf_counter()
    metadata={"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),
              "cpu_count":os.cpu_count(),"workers":args.workers,"command":sys.argv,
              "packages":{p:im.version(p) for p in ["pairpulse","numpy","scipy","matplotlib","qutip","nbclient","ipykernel"]},
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "started_at_utc":datetime.now(timezone.utc).isoformat(),"planned_jobs":len(todo)}
    (args.output/"environment.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Declared protocol and {len(todo)} mode evaluations",flush=True)
    # File-backed subprocess shards also work where Windows named pipes are
    # unavailable. Each evaluation is independent; this is not a timing benchmark.
    worker_dir=args.output/"workers";worker_dir.mkdir(exist_ok=True)
    processes=[];handles=[]
    for index in range(args.workers):
        inp=worker_dir/f"jobs_{index}.json";out=worker_dir/f"results_{index}.csv"
        inp.write_text(json.dumps(todo[index::args.workers]))
        log=(worker_dir/f"worker_{index}.log").open("w");handles.append(log)
        processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
            "--worker-input",str(inp.resolve()),"--worker-output",str(out.resolve())],stdout=log,stderr=log))
    while any(p.poll() is None for p in processes):
        time.sleep(10)
        count=sum(max(0,len(p.read_text().splitlines())-1) for p in worker_dir.glob("results_*.csv"))
        print(f"Saved {count}/{len(todo)}; elapsed {time.perf_counter()-started:.1f}s",flush=True)
    for h in handles: h.close()
    if any(p.returncode for p in processes): raise RuntimeError("Worker failed; inspect retained worker logs")
    for index in range(args.workers):
        with (worker_dir/f"results_{index}.csv").open(newline="") as f: rows.extend(csv.DictReader(f))
    write_csv(args.output/"mode_results.csv",rows)
    metadata.update(elapsed_seconds=time.perf_counter()-started,completed_jobs=len(rows),
                    failures=sum(r["status"]!="ok" for r in rows))
    (args.output/"environment.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2),flush=True)
    if metadata["failures"]:
        raise SystemExit(1)

if __name__=="__main__": main()
