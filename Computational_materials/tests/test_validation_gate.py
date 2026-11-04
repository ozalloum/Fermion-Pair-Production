"""Synthetic-record integrity tests; these fixtures are not numerical evidence."""
from pathlib import Path
import copy,csv,json,sys,unittest,tempfile,uuid,shutil,subprocess
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validation_protocol import jobs,pulse_from_spec,PROTOCOL
from summarize_validation import validate_records,summarize
from pairpulse import sauter_exact_occupation


class ValidationGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]
        for job in jobs():
            pulse=pulse_from_spec(job['pulse_spec']);ti,tf=pulse.time_window(job['tail'])
            exact=sauter_exact_occupation(job['p'],pulse) if job['case']=='single' else None
            occupation=exact if exact is not None else .01*np.exp(-job['p']**2)
            cls.rows.append(dict(job,occupation=occupation,diagnostic=0.,nfev=-1 if job['method']=='qutip_vern9' else 1,
                t0=ti,tf=tf,wall_seconds=0.,exact=exact,absolute_error=0. if exact is not None else None,
                resolved_relative_error=0. if exact is not None and exact>=1e-10 else None,status='ok',failure=''))

    def test_complete_fixture_is_accepted(self):
        self.assertEqual(len(validate_records(self.rows)),2228)

    def test_missing_duplicate_and_unknown_jobs_are_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]], [dict(self.rows[0],job_id='unknown')]+self.rows[1:]):
            with self.assertRaises(ValueError):validate_records(rows)

    def test_invalid_window_records_are_rejected(self):
        i=next(i for i,r in enumerate(self.rows) if r['study']=='window' and r['method']=='qke')
        for key,value in [('occupation',2.),('occupation',np.nan),('diagnostic',np.inf),('status','failed'),('tau',9.),('t0',123.),('absolute_error',np.nan),('exact',0.),('pulse_spec','{}')]:
            rows=[dict(r) for r in self.rows];rows[i][key]=value
            with self.subTest(key=key,value=value):
                with self.assertRaises(ValueError):validate_records(rows)

    def test_summary_and_cli_fail_closed(self):
        # Default inherited directory permissions also work on Windows audit hosts.
        parent=Path(tempfile.gettempdir()).resolve();folder=parent/('pairpulse-test-'+uuid.uuid4().hex);folder.mkdir()
        try:
            (folder/"protocol.json").write_text(json.dumps(PROTOCOL))
            fields=list(self.rows[0])
            def save(rows):
                with (folder/'mode_results.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            save(self.rows)
            self.assertTrue(summarize(folder)['criteria_passed'])
            save(self.rows[:-1])
            self.assertFalse(summarize(folder)['criteria_passed'])
            script=Path(__file__).resolve().parents[1]/'scripts/summarize_validation.py'
            proc=subprocess.run([sys.executable,str(script),'--data',str(folder)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,1)
            # A structurally valid endpoint error must fail numerical acceptance.
            rows=copy.deepcopy(self.rows)
            row=next(r for r in rows if r['study']=='window' and r['method']=='qke' and r['tail']==12)
            row['occupation']+=2e-9;row['absolute_error']=abs(row['occupation']-row['exact'])
            row['resolved_relative_error']=row['absolute_error']/row['exact'] if row['exact']>=1e-10 else None
            save(rows);result=summarize(folder)
            self.assertFalse(result['criteria_passed']);self.assertFalse(result['checks']['endpoints'])
            # Once-successful evidence must not leave a stale green summary when
            # its input files later disappear or acquire a malformed structure.
            for broken in ('[]','null','42','"text"','{broken'):
                save(self.rows)
                (folder/'protocol.json').write_text(broken)
                (folder/'summary.json').write_text('{"criteria_passed":true}')
                result=summarize(folder)
                self.assertFalse(result['criteria_passed'])
                self.assertFalse(json.loads((folder/'summary.json').read_text())['criteria_passed'])
            (folder/'protocol.json').write_text(json.dumps(PROTOCOL))
            for broken in (None,b'\xff\xfe',b'job_id,status\n"unterminated'):
                (folder/'summary.json').write_text('{"criteria_passed":true}')
                csv_path=folder/'mode_results.csv'
                if broken is None:csv_path.unlink(missing_ok=True)
                else:csv_path.write_bytes(broken)
                proc=subprocess.run([sys.executable,str(script),'--data',str(folder)],capture_output=True,text=True)
                self.assertEqual(proc.returncode,1)
                self.assertFalse(json.loads((folder/'summary.json').read_text())['criteria_passed'])
            save(self.rows)
            (folder/'summary.json').write_text('{"criteria_passed":true}')
            with patch('summarize_validation.validate_records',side_effect=RuntimeError('interrupted calculation')):
                with self.assertRaises(RuntimeError):summarize(folder)
            self.assertFalse(json.loads((folder/'summary.json').read_text())['criteria_passed'])
        finally:
            assert folder.resolve().is_relative_to(parent) and folder.name.startswith('pairpulse-test-')
            shutil.rmtree(folder)

if __name__=='__main__':unittest.main()
