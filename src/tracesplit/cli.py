from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .engine import run_source, save_result, ExecutionError
from .language import compile_source, LanguageError
from .exporter import export_bundle, compare_results
from .verifier import verify

def main(argv=None):
    parser = argparse.ArgumentParser(prog="tracesplit")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check"); check.add_argument("program", type=Path)
    run = sub.add_parser("run"); run.add_argument("program", type=Path); run.add_argument("--root", type=Path, default=Path(".")); run.add_argument("--out", type=Path, required=True); run.add_argument("--export", action="store_true")
    ver = sub.add_parser("verify"); ver.add_argument("evidence", type=Path); ver.add_argument("--data", type=Path)
    comp = sub.add_parser("compare"); comp.add_argument("first", type=Path); comp.add_argument("second", type=Path)
    serve = sub.add_parser("serve"); serve.add_argument("--root", type=Path, default=Path(".")); serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            p = compile_source(args.program.read_text()).as_dict(); print(json.dumps({"ok": True, "task": p["task"], "candidates": len(p["models"])})); return 0
        if args.command == "run":
            result = run_source(args.program.read_text(), args.root); save_result(result, args.out)
            if args.export: export_bundle(result, args.root, args.out/"python_export")
            print(json.dumps({"selected": result["selected"], "test": result["test"], "output": str(args.out)}, indent=2)); return 0
        if args.command == "verify": report = verify(json.loads(args.evidence.read_text()), args.data)
        elif args.command == "compare": report = compare_results(json.loads(args.first.read_text()), json.loads(args.second.read_text()))
        else:
            from .app import serve
            serve(args.root, args.port); return 0
        print(json.dumps(report, indent=2)); return 0 if report["ok"] else 1
    except (LanguageError, ExecutionError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr); return 2
