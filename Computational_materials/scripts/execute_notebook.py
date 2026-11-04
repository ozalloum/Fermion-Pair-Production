"""Execute every notebook cell in a real Jupyter kernel using installed PairPulse."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--notebook',type=Path,default=Path('notebooks/PairPulse_reproduction.ipynb'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kernel-name',default='python3')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    nb=nbformat.read(args.notebook,as_version=4)
    start=time.perf_counter()
    # Working outside src prevents an uninstalled source checkout masking errors.
    client=NotebookClient(nb,timeout=600,kernel_name=args.kernel_name,allow_errors=False,
                          resources={'metadata':{'path':str(args.output.resolve())}})
    client.execute()
    dest=args.output/'PairPulse_reproduction_executed.ipynb';nbformat.write(nb,dest)
    report={'status':'passed','actual_kernel':True,'code_cells':sum(c.cell_type=='code' for c in nb.cells),
            'execution_counts':[c.execution_count for c in nb.cells if c.cell_type=='code'],
            'error_outputs':[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error'],
            'elapsed_seconds':time.perf_counter()-start,
            'source_sha256':hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
            'executed_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (args.output/'notebook_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
