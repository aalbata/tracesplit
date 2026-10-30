"""Parsing and a closed-world, source-located workflow checker."""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from importlib.resources import files
import hashlib
import json
import math
from typing import Any
from lark import Lark, Tree, UnexpectedInput

class LanguageError(ValueError):
    def __init__(self, code: str, message: str, line: int = 0):
        self.code, self.line, self.message = code, line, message
        super().__init__(f"{code}" + (f" at line {line}" if line else "") + f": {message}")

PARSER = Lark(files("tracesplit").joinpath("grammar.lark").read_text(),
              parser="lalr", propagate_positions=True, maybe_placeholders=False)
CLASSIFIERS = {"logistic", "forest", "tree"}
REGRESSORS = {"ridge", "forest", "tree"}
METRICS = {"classification": {"accuracy", "balanced_accuracy", "macro_f1"},
           "regression": {"rmse", "mae", "r2"}}

_VALIDATION_SEAL = object()

@dataclass(frozen=True)
class Plan:
    """Serialized representation is copied at API boundaries, not a security capability."""
    encoded: str
    _validation_seal: object = field(default=None, repr=False, compare=False)
    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.encoded)

def fail(code: str, message: str, node: Tree) -> None:
    raise LanguageError(code, message, node.meta.line)

def unquote(value: Any) -> str:
    return json.loads(str(value))

def compile_source(source: str) -> Plan:
    if not isinstance(source, str) or not source.strip():
        raise LanguageError("E_EMPTY", "Write a workflow before running it.")
    if len(source.encode("utf-8")) > 100_000:
        raise LanguageError("E_SIZE", "Programs are limited to 100,000 UTF-8 bytes.")
    try:
        tree = PARSER.parse(source.rstrip() + "\n")
    except UnexpectedInput as exc:
        raise LanguageError("E_SYNTAX", "This command does not match the language grammar.", exc.line) from exc
    p: dict[str, Any] = {"version": 1, "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "drop": [], "models": [], "actions": [], "explain": False, "lines": {}}
    state = "empty"
    trained: set[str] = set()
    evaluated: set[str] = set()
    for node in tree.children:
        kind = str(node.data)
        a = node.children
        line = node.meta.line
        if state == "assessed" and kind != "explain":
            fail("E_TERMINAL", "Assessment is terminal. Start a separate study to change the workflow.", node)
        if kind == "read":
            if state != "empty": fail("E_ORDER", "Read exactly one dataset, as the first command.", node)
            name = unquote(a[0])
            if not name or "\x00" in name: fail("E_PATH", "Provide a nonempty CSV filename.", node)
            p["dataset"] = name; state = "read"
        elif kind == "predict":
            if state != "read": fail("E_ORDER", "Declare one target immediately after reading data.", node)
            p["target"], p["task"] = unquote(a[0]), str(a[1])
            if p["task"] not in METRICS: fail("E_TASK", "Use classification or regression.", node)
            state = "target"
        elif kind == "drop":
            if state != "target": fail("E_ORDER", "Drop columns after the target and before the split.", node)
            cols = [unquote(v) for v in a]
            if p["target"] in cols: fail("E_TARGET", "The target cannot be dropped.", node)
            if len(set(p["drop"] + cols)) != len(p["drop"] + cols): fail("E_DUPLICATE", "A column was dropped more than once.", node)
            p["drop"].extend(cols)
        elif kind == "split":
            if state != "target": fail("E_ORDER", "Split once, before any fitted transformation or model.", node)
            strategy = str(a[0]); offset = 1
            by = None
            if len(a) == 6: by = unquote(a[1]); offset = 2
            if strategy not in {"random", "stratified", "grouped", "temporal"}: fail("E_SPLIT", "Use random, stratified, grouped, or temporal.", node)
            if (strategy in {"grouped", "temporal"}) != (by is not None): fail("E_SPLIT_COLUMN", "Only grouped and temporal splits require a by column.", node)
            if by == p["target"]: fail("E_TARGET", "A split metadata column cannot be the target.", node)
            if by in p["drop"]: fail("E_SPLIT_COLUMN", "Keep the split metadata column; it is excluded from predictors automatically.", node)
            if strategy == "stratified" and p["task"] != "classification": fail("E_SPLIT_TASK", "Stratification is supported for classification only.", node)
            fractions = [Decimal(str(v)[:-1]) for v in a[offset:offset+3]]
            if any(v <= 0 for v in fractions) or sum(fractions) != Decimal(100): fail("E_FRACTIONS", "Use three positive percentages that sum to exactly 100%.", node)
            seed = int(a[offset+3])
            if not 0 <= seed < 2**32: fail("E_SEED", "The seed must be an integer from 0 through 2^32-1.", node)
            p["split"] = {"strategy": strategy, "by": by, "fractions": [float(v/100) for v in fractions], "seed": seed}
            state = "split"
        elif kind == "prepare":
            if state != "split": fail("E_ORDER", "Prepare once, after the split and before training.", node)
            numeric, scale, cat, role = map(str, a)
            if role != "train": fail("E_FIT_SCOPE", "Learn preprocessing parameters from train, not validation or test.", node)
            if numeric not in {"median", "mean", "most_frequent"}: fail("E_PREPARE", "Numeric imputation must be median, mean, or most_frequent.", node)
            if scale not in {"standard", "none"}: fail("E_PREPARE", "Scaling must be standard or none.", node)
            if cat not in {"most_frequent", "constant"}: fail("E_PREPARE", "Categorical imputation must be most_frequent or constant.", node)
            p["prepare"] = {"numeric": numeric, "scale": scale, "categorical": cat}
            state = "prepared"
        elif kind == "train":
            if state not in {"prepared", "training"}: fail("E_ORDER", "Train after preprocessing and before selection.", node)
            name, algorithm, role = str(a[0]), str(a[1]), str(a[-1])
            if role != "train": fail("E_FIT_SCOPE", "Models may be fitted on train only.", node)
            if name in trained or name == "best": fail("E_MODEL_NAME", "Use a new model name other than best.", node)
            if len(trained) >= 12: fail("E_MODEL_LIMIT", "At most 12 candidates are allowed per workflow.", node)
            allowed = CLASSIFIERS if p["task"] == "classification" else REGRESSORS
            if algorithm not in allowed: fail("E_ALGORITHM", f"Use one of {', '.join(sorted(allowed))} for this task.", node)
            params = {}
            for param in a[2:-1]:
                key, raw = str(param.children[0]), str(param.children[1])
                if key in params: fail("E_PARAMETER", "A parameter was specified twice.", node)
                value = float(raw)
                if not math.isfinite(value): fail("E_PARAMETER", "Parameters must be finite.", node)
                params[key] = value
            limits = {"logistic": {"C"}, "ridge": {"alpha"}, "forest": {"trees", "depth"}, "tree": {"depth"}}[algorithm]
            if set(params) - limits: fail("E_PARAMETER", "An unsupported parameter was supplied.", node)
            for key, value in params.items():
                if key in {"C", "alpha"} and not 0 < value <= 1e6: fail("E_PARAMETER", "Regularization parameters must be positive and at most 1e6.", node)
                if key in {"trees", "depth"}:
                    cap = 200 if key == "trees" else 30
                    if not value.is_integer() or not 1 <= value <= cap: fail("E_PARAMETER", f"{key} must be an integer between 1 and {cap}.", node)
                    params[key] = int(value)
            p["models"].append({"name": name, "algorithm": algorithm, "params": params, "line": line})
            p["actions"].append({"kind": "train", "model": name, "line": line})
            trained.add(name); state = "training"
        elif kind == "evaluate":
            name, role = map(str, a)
            if state != "training" or name not in trained: fail("E_MODEL", "Train this model before evaluating it.", node)
            if role != "validation": fail("E_EVAL_SCOPE", "Use validation for development; test is reserved for final assessment.", node)
            if name in evaluated: fail("E_DUPLICATE", "Each candidate is evaluated once in this fixed workflow.", node)
            p["actions"].append({"kind": "evaluate", "model": name, "line": line})
            evaluated.add(name)
        elif kind == "select":
            if state != "training" or not trained or trained != evaluated: fail("E_SELECTION", "Evaluate every trained candidate on validation before selecting.", node)
            metric = str(a[0])
            if metric not in METRICS[p["task"]]: fail("E_METRIC", "The selection metric does not match the task.", node)
            p["metric"] = metric; state = "selected"
        elif kind == "assess":
            if state != "selected": fail("E_ORDER", "Select the candidate before final assessment.", node)
            if str(a[0]) != "test": fail("E_ASSESS_SCOPE", "Final assessment uses test only.", node)
            state = "assessed"
        elif kind == "explain":
            if state != "assessed": fail("E_ORDER", "Explain a completed workflow after assessment.", node)
            if p["explain"]: fail("E_DUPLICATE", "Explain the completed workflow once.", node)
            p["explain"] = True
        p["lines"].setdefault(kind, line)
    if state != "assessed":
        raise LanguageError("E_INCOMPLETE", "Complete the workflow with selection and one test assessment.")
    return Plan(json.dumps(p, sort_keys=True, allow_nan=False), _VALIDATION_SEAL)
