"""Export a frozen-split replay without importing TraceSplit at replay time."""
from __future__ import annotations
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import shutil
from .engine import resolve_data_path

def export_bundle(result: dict, root: str | Path, directory: str | Path) -> Path:
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    source = resolve_data_path(result["plan"]["dataset"], root)
    if hashlib.sha256(source.read_bytes()).hexdigest() != result["dataset_sha256"]:
        raise ValueError("The input CSV changed after execution; export was stopped.")
    spec = {key: result[key] for key in ["dataset_sha256", "partitions", "schema", "plan"]}
    template = files("tracesplit").joinpath("replay_template.py.txt").read_text()
    program = template.replace("__SPEC_JSON_LITERAL__", repr(spec))
    (out/"experiment.py").write_text(program, encoding="utf-8")
    if source.resolve() != (out/"data.csv").resolve(): shutil.copyfile(source, out/"data.csv")
    (out/"workflow.tsml").write_text(result["plan"]["source"])
    (out/"split.json").write_text(json.dumps(result["partitions"], indent=2)+"\n")
    (out/"README.txt").write_text("Run: python experiment.py --output replay_result.json\n"
                                 "Use the scientific-Python versions in the parent artifact's requirements-tested.txt.\n"
                                 "TraceSplit and Lark are not required for this replay.\n"
                                 "The CSV and split are frozen; this is not a proof of equivalence for unseen inputs.\n")
    return out

def compare_results(a: dict, b: dict, atol: float = 1e-10, rtol: float = 1e-10) -> dict:
    import numpy as np
    errors = []; max_error = 0.0
    if a["selected"] != b["selected"]: errors.append("Selected candidate differs.")
    for stage in ["validation", "test"]:
        first = a[stage] if stage == "validation" else {"test": a[stage]}
        second = b[stage] if stage == "validation" else {"test": b[stage]}
        if set(first) != set(second): errors.append(f"{stage} candidate names differ."); continue
        for name in first:
            if set(first[name]) != set(second[name]): errors.append(f"{stage} metric names differ."); continue
            for key in first[name]:
                delta = abs(first[name][key]-second[name][key]); max_error = max(max_error, delta)
                if not np.isclose(first[name][key], second[name][key], atol=atol, rtol=rtol): errors.append(f"{stage}/{name}/{key} differs.")
    for key in ["validation_predictions", "test_predictions"]:
        first = a[key] if isinstance(a[key], dict) else {"test": a[key]}
        second = b[key] if isinstance(b[key], dict) else {"test": b[key]}
        if set(first) != set(second): errors.append("Prediction keys differ."); continue
        for name in first:
            x, y = np.array(first[name]), np.array(second[name])
            if x.shape != y.shape: errors.append("Prediction shapes differ."); continue
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                max_error = max(max_error, float(np.max(np.abs(x-y))) if x.size else 0)
                if not np.allclose(x, y, atol=atol, rtol=rtol): errors.append(f"{key}/{name} differs.")
            elif not np.array_equal(x, y): errors.append(f"{key}/{name} differs.")
    return {"ok": not errors, "errors": errors, "max_absolute_difference": max_error, "atol": atol, "rtol": rtol}
