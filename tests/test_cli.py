import json
from tracesplit.cli import main
from conftest import ROOT

def test_cli_check(capsys):
    assert main(['check',str(ROOT/'examples/iris.tsml')])==0
    assert json.loads(capsys.readouterr().out)['ok']

def test_cli_bad_program(tmp_path,capsys):
    source=tmp_path/'bad.tsml';source.write_text('import os\n')
    assert main(['check',str(source)])==2
    assert 'E_SYNTAX' in capsys.readouterr().err

def test_cli_run_verify_compare(tmp_path,capsys):
    run=tmp_path/'run'
    assert main(['run',str(ROOT/'examples/iris.tsml'),'--root',str(ROOT),'--out',str(run),'--export'])==0
    assert main(['verify',str(run/'evidence.json'),'--data',str(ROOT/'datasets/iris.csv')])==0
    assert main(['compare',str(run/'evidence.json'),str(run/'evidence.json')])==0
    assert '"ok": true' in capsys.readouterr().out
