"""Run the CLI, standard-library verifier, and standalone export in subprocesses."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='tracesplit-smoke-') as temporary:
        out = Path(temporary)/'run'
        commands = [
            [sys.executable, '-m', 'tracesplit', 'check', 'examples/iris.tsml'],
            [sys.executable, '-m', 'tracesplit', 'run', 'examples/iris.tsml', '--root', '.', '--out', str(out), '--export'],
            [sys.executable, '-S', str(ROOT/'src/tracesplit/verifier.py'), str(out/'evidence.json'), '--data', str(ROOT/'datasets/iris.csv')],
            [sys.executable, str(out/'python_export/experiment.py'), '--output', str(out/'replay.json')],
            [sys.executable, '-m', 'tracesplit', 'compare', str(out/'evidence.json'), str(out/'replay.json')],
        ]
        records=[]
        for command in commands:
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
            records.append({'command':[str(x).replace(str(Path(temporary)),'<temporary>') for x in command], 'returncode':result.returncode})
            if result.returncode:
                print(result.stdout);print(result.stderr,file=sys.stderr)
                raise RuntimeError(f'Smoke command failed: {command[1:4]}')
        print(json.dumps({'ok':True,'commands':records},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
