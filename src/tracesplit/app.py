"""A dependency-free local browser playground. Not a public multi-user service."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import zipfile

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TraceSplit Playground</title><style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#152334;background:#f4f6f8;line-height:1.5}
body{margin:0}header{padding:25px 4%;background:#142b40;color:white}h1{font-size:27px;margin:0;font-weight:650}header p{margin:4px 0 0;color:#cad5df;font-size:14px}
main{max-width:1420px;margin:24px auto;padding:0 4%}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:18px}select,button,a.download{font:inherit;padding:10px 15px;border:1px solid #b9c6d1;border-radius:5px;background:white;color:#19344b;text-decoration:none}button{cursor:pointer;background:#193e5c;color:white;border-color:#193e5c}button:disabled{opacity:.55;cursor:wait}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.card{background:white;border:1px solid #d5dce2;border-radius:8px;padding:20px;min-width:0}h2{font-size:17px;margin:0 0 12px}textarea{width:100%;height:405px;box-sizing:border-box;font:13px/1.7 ui-monospace,Consolas,monospace;padding:14px;border:1px solid #cbd4dd;resize:vertical;background:#fafbfd;color:#18374d;tab-size:4}.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}.tabs button{background:#edf2f6;color:#29445b;border:0;font-size:13px;padding:8px 10px}.tabs button.active{background:#193e5c;color:white}pre{font:12px/1.65 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:470px;overflow:auto;margin:0}.status{font-size:14px;margin:0 0 18px;min-height:24px}.error{color:#a32a2a}.success{color:#166137}.note{font-size:12px;color:#5e6b76;margin-top:18px}.metricrow{display:flex;gap:15px;flex-wrap:wrap}.metric{background:#f0f4f7;border-radius:5px;padding:14px;min-width:130px}.metric b{display:block;font-size:22px;margin-top:3px}.small{font-size:12px;color:#5c6d7b}#downloads{display:none;gap:10px;flex-wrap:wrap;margin-top:20px}footer{margin:24px 0;font-size:12px;color:#697783}@media(max-width:900px){.grid{grid-template-columns:1fr}textarea{height:380px}}
</style></head><body><header><h1>TraceSplit</h1><p>Checked machine-learning workflows. Recorded evidence. Ordinary Python replay.</p></header>
<main><div class="bar"><label for="example">Example</label><select id="example" aria-label="Example dataset"></select><button id="run">Run workflow</button><span class="small">Local execution &middot; CPU only &middot; No external API</span></div>
<p class="status" id="status">Choose an example, inspect the program, and run it.</p>
<div class="grid"><section class="card"><h2>Workflow program</h2><textarea id="code" spellcheck="false" aria-label="Workflow program"></textarea><p class="note">Preprocessing and candidate fitting use training rows. Selection uses validation scores. Final test assessment is terminal within this run.</p></section>
<section class="card"><div class="tabs"><button data-tab="metrics" class="active">Results</button><button data-tab="explanation">Explanation</button><button data-tab="evidence">Evidence</button><button data-tab="python">Python</button></div><div id="result"><p class="small">No execution yet.</p></div></section></div>
<div id="downloads"></div><footer>Research prototype 0.1.0. A consistent evidence record is not proof that a computation occurred. The tool does not prevent test reuse across separate runs, identify target proxies, or establish that a chosen dataset is suitable for deployment.</footer></main>
<script>
const token='__TOKEN__';let examples={},data=null,tab='metrics';
function show(){const out=document.getElementById('result');out.replaceChildren();if(!data){out.textContent='No execution yet.';return;}if(tab==='metrics'){const title=document.createElement('h2');title.textContent='Selected candidate: '+data.selected;out.appendChild(title);const row=document.createElement('div');row.className='metricrow';for(const [k,v] of Object.entries(data.test)){const box=document.createElement('div');box.className='metric';box.textContent=k;const b=document.createElement('b');b.textContent=Number(v).toPrecision(6);box.appendChild(b);row.appendChild(box);}out.appendChild(row);const h=document.createElement('h2');h.style.marginTop='24px';h.textContent='Validation scores';out.appendChild(h);const pre=document.createElement('pre');pre.textContent=JSON.stringify(data.validation,null,2);out.appendChild(pre);}else{const pre=document.createElement('pre');pre.textContent=tab==='evidence'?JSON.stringify(data.evidence,null,2):data[tab];out.appendChild(pre);}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));show();});
fetch('/api/examples').then(r=>r.json()).then(x=>{examples=x;const select=document.getElementById('example');for(const k of Object.keys(x)){const o=document.createElement('option');o.value=k;o.textContent=k.replaceAll('_',' ');select.appendChild(o);}select.value='iris';document.getElementById('code').value=examples.iris;select.onchange=()=>document.getElementById('code').value=examples[select.value];});
document.getElementById('run').onclick=async()=>{const button=document.getElementById('run'),status=document.getElementById('status');button.disabled=true;status.className='status';status.textContent='Executing the declared workflow...';try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-TraceSplit-Token':token},body:JSON.stringify({source:document.getElementById('code').value})});const x=await response.json();if(!response.ok)throw new Error(x.error);data=x;status.className='status success';status.textContent='Execution completed. Independent evidence check: '+(x.verified?'passed':'failed');show();const d=document.getElementById('downloads');d.replaceChildren();d.style.display='flex';for(const [kind,label] of [['bundle','Download run and Python replay'],['evidence','Download evidence JSON']]){const a=document.createElement('a');a.className='download';a.href='/download/'+x.id+'/'+kind;a.textContent=label;d.appendChild(a);}}catch(e){status.className='status error';status.textContent=e.message;}finally{button.disabled=false;}};
</script></body></html>'''

def build_server(root: Path, port: int = 8765):
    root = Path(root).resolve(); token = secrets.token_hex(24)
    workspace = tempfile.TemporaryDirectory(prefix="tracesplit-playground-")
    completed_runs = {}; lock = threading.BoundedSemaphore(1)
    examples = {p.stem: p.read_text() for p in sorted((root/"examples").glob("*.tsml"))}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, status, body, content_type="application/json", filename=None):
            if isinstance(body, dict): body = json.dumps(body).encode()
            if isinstance(body, str): body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("Cache-Control", "no-store")
            if filename: self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            route = urllib.parse.urlsplit(self.path).path
            if route == "/": return self.send(200, PAGE.replace("__TOKEN__", token), "text/html; charset=utf-8")
            if route == "/api/examples": return self.send(200, examples)
            parts = route.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "download" and parts[1] in completed_runs:
                out = completed_runs[parts[1]]
                if parts[2] == "evidence": return self.send(200, (out/"evidence.json").read_bytes(), filename="evidence.json")
                if parts[2] == "bundle":
                    stream = io.BytesIO()
                    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
                        for path in out.rglob("*"):
                            if path.is_file(): z.write(path, path.relative_to(out))
                    return self.send(200, stream.getvalue(), "application/zip", "tracesplit_run.zip")
            self.send(404, {"error": "Not found."})
        def do_POST(self):
            if self.path != "/api/run": return self.send(404, {"error": "Not found."})
            if self.headers.get("X-TraceSplit-Token") != token: return self.send(403, {"error": "Invalid local token."})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json": return self.send(415, {"error": "JSON is required."})
            try: length = int(self.headers.get("Content-Length", "0"))
            except ValueError: return self.send(400, {"error": "Invalid content length."})
            if not 0 < length <= 100_000: return self.send(413, {"error": "Program request is too large or empty."})
            if not lock.acquire(blocking=False): return self.send(409, {"error": "One workflow is already executing in this local playground."})
            try:
                request = json.loads(self.rfile.read(length)); source = request["source"]
                if not isinstance(source, str): raise ValueError("Source must be text.")
                if len(completed_runs) >= 20: return self.send(429, {"error": "Restart the local playground after 20 runs to clear temporary results."})
                ident = secrets.token_hex(12); directory = Path(workspace.name)/ident; directory.mkdir()
                script = directory/"workflow.tsml"; script.write_text(source); out = directory/"result"
                env = dict(os.environ); env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                completed = subprocess.run([sys.executable, "-m", "tracesplit", "run", str(script), "--root", str(root), "--out", str(out), "--export"],
                                           capture_output=True, text=True, env=env, timeout=60)
                if completed.returncode:
                    return self.send(400, {"error": completed.stderr[-5000:].strip() or "The workflow failed."})
                evidence = json.loads((out/"evidence.json").read_text())
                from .verifier import verify
                from .engine import resolve_data_path
                report = verify(evidence, resolve_data_path(evidence["plan"]["dataset"], root))
                completed_runs[ident] = out
                self.send(200, {"id": ident, "verified": report["ok"], "selected": evidence["selected"], "test": evidence["test"],
                                "validation": evidence["validation"], "explanation": (out/"explanation.txt").read_text(),
                                "evidence": evidence, "python": (out/"python_export"/"experiment.py").read_text()})
            except subprocess.TimeoutExpired: self.send(408, {"error": "The local 60-second execution limit was reached."})
            except (ValueError, KeyError, OSError) as exc: self.send(400, {"error": str(exc)})
            finally: lock.release()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.workspace = workspace
    return server

def serve(root: Path, port: int):
    server = build_server(root, port)
    print(f"TraceSplit local playground: http://127.0.0.1:{server.server_port}", flush=True)
    print("Keep this terminal open. Stop with Ctrl+C. Do not expose this server publicly.", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close(); server.workspace.cleanup()
