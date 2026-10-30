import copy
import json
import subprocess
import sys
import pytest
from tracesplit import run_source
from tracesplit.evidence import digest
from tracesplit.verifier import verify

@pytest.fixture(scope="module")
def evidence():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return run_source((root/"examples"/"iris.tsml").read_text(), root)

def rehash(record):
    previous = "0"*64
    for i, event in enumerate(record["events"]):
        event["seq"] = i; event["previous"] = previous; event.pop("hash", None)
        event["hash"] = digest(event); previous = event["hash"]

@pytest.mark.parametrize("kind", ["prepare_fit", "fit", "evaluate", "assess"])
def test_wrong_scope_rehashed(evidence, kind):
    e = copy.deepcopy(evidence)
    event = next(x for x in e["events"] if x["kind"] == kind)
    role = "test" if event["role"] != "test" else "train"
    event["rows"] = e["partitions"][role]; event["role"] = role; rehash(e)
    assert not verify(e)["ok"]

def test_extra_assessment(evidence):
    e = copy.deepcopy(evidence); e["events"].append(copy.deepcopy(e["events"][-1])); rehash(e)
    assert not verify(e)["ok"]

def test_overlap(evidence):
    e = copy.deepcopy(evidence); e["partitions"]["test"][0] = e["partitions"]["train"][0]
    assert not verify(e)["ok"]

def test_wrong_winner(evidence):
    e = copy.deepcopy(evidence); e["selected"] = "tree" if e["selected"] != "tree" else "forest"
    assert not verify(e)["ok"]

def test_broken_chain(evidence):
    e = copy.deepcopy(evidence); e["events"][0]["hash"] = "0"*64
    assert not verify(e)["ok"]

def test_wrong_scores_require_data(evidence, root):
    e = copy.deepcopy(evidence); e["test"]["accuracy"] = .123456
    e["events"][-1]["payload"]["metrics"] = copy.deepcopy(e["test"]); rehash(e)
    assert verify(e)["ok"]  # Record consistency alone is intentionally weaker.
    assert not verify(e, root/"datasets"/"iris.csv")["ok"]

def test_standard_library_checker(evidence, tmp_path, root):
    path = tmp_path/"evidence.json"; path.write_text(json.dumps(evidence))
    r = subprocess.run([sys.executable, "-S", str(root/"src"/"tracesplit"/"verifier.py"), str(path), "--data", str(root/"datasets"/"iris.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr+r.stdout
    assert json.loads(r.stdout)["ok"]

@pytest.mark.parametrize("bad", [{}, [], None, {"format":"bad"}])
def test_malformed_evidence(bad):
    assert not verify(bad)["ok"]
