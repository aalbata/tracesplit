"""Deterministic encodings for inspectable evidence, not cryptographic attestation."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any
import numpy as np

def normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray): return normalize(value.tolist())
    if isinstance(value, np.generic): return normalize(value.item())
    if isinstance(value, dict): return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [normalize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numerical output cannot be placed in evidence.")
    return value

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def append_event(events: list[dict], kind: str, line: int, role: str, rows: list[int], **payload: Any) -> None:
    e = normalize({"seq": len(events), "kind": kind, "line": line, "role": role, "rows": rows,
                   "payload": payload, "previous": events[-1]["hash"] if events else "0"*64})
    e["hash"] = digest(e)
    events.append(e)

def estimator_state(model: Any) -> dict:
    name = type(model).__name__
    state: dict[str, Any] = {"class": name}
    for attr in ["coef_", "intercept_", "classes_", "n_features_in_"]:
        if hasattr(model, attr): state[attr] = normalize(getattr(model, attr))
    if hasattr(model, "tree_"):
        t = model.tree_
        state["tree"] = {a: normalize(getattr(t, a)) for a in
                         ["children_left", "children_right", "feature", "threshold", "value", "n_node_samples"]}
    if hasattr(model, "estimators_"):
        state["estimators"] = [estimator_state(m) for m in model.estimators_]
    return state

def preprocessing_state(preprocessor: Any) -> dict:
    state = {}
    for name, pipeline, columns in preprocessor.transformers_:
        if name == "remainder": continue
        part = {"columns": list(columns)}
        for step_name, step in pipeline.steps:
            part[step_name] = {key: normalize(getattr(step, key)) for key in
                              ["statistics_", "mean_", "var_", "scale_", "categories_"] if hasattr(step, key)}
        state[name] = part
    return state
