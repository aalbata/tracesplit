import json
import re
import threading
import urllib.request
import urllib.error
import pytest
from tracesplit.app import build_server

@pytest.fixture
def server(root):
    s = build_server(root, 0)
    thread = threading.Thread(target=s.serve_forever, daemon=True); thread.start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown(); s.server_close(); s.workspace.cleanup(); thread.join()

def test_browser_routes_and_execution(server):
    html = urllib.request.urlopen(server).read().decode()
    assert "TraceSplit" in html
    token = re.search("const token='([^']+)'", html).group(1)
    examples = json.load(urllib.request.urlopen(server+"/api/examples"))
    req = urllib.request.Request(server+"/api/run", data=json.dumps({"source": examples["iris"]}).encode(),
                                 headers={"Content-Type": "application/json", "X-TraceSplit-Token": token})
    result = json.load(urllib.request.urlopen(req, timeout=60))
    assert result["verified"] and result["selected"]
    binary = urllib.request.urlopen(server+"/download/"+result["id"]+"/bundle").read()
    assert binary[:2] == b"PK"

def test_csrf_and_unknown_routes(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(urllib.request.Request(server+"/api/run", data=b'{}', headers={"Content-Type":"application/json"}))
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(server+"/../../etc/passwd")
    assert error.value.code == 404
