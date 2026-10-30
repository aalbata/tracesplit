"""CPU-only execution of validated, bounded tabular workflows."""
from __future__ import annotations
import copy
import csv
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from .language import Plan, LanguageError, compile_source, _VALIDATION_SEAL
from .evidence import append_event, digest, estimator_state, normalize, preprocessing_state

class ExecutionError(ValueError):
    pass

def resolve_data_path(name: str, root: str | Path) -> Path:
    base = Path(root).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base):
        raise ExecutionError("Dataset paths must stay inside the supplied workspace root.")
    if candidate.suffix.lower() != ".csv" or not candidate.is_file():
        raise ExecutionError("A local CSV file is required.")
    return candidate

def load_data(path: Path, p: dict) -> tuple[pd.DataFrame, bytes]:
    if path.stat().st_size > 50_000_000: raise ExecutionError("CSV inputs are limited to 50 MB.")
    raw = path.read_bytes()
    try:
        header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        if len(header) != len(set(header)): raise ExecutionError("CSV column names must be unique.")
        df = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeError, pd.errors.ParserError, StopIteration) as exc:
        raise ExecutionError("The dataset must be a nonempty UTF-8 CSV with a header.") from exc
    if len(df) < 15 or len(df) > 100_000 or len(df.columns) > 200:
        raise ExecutionError("Use 15 to 100,000 rows and at most 200 columns.")
    required = {p["target"], *p["drop"]}
    if p["split"]["by"]: required.add(p["split"]["by"])
    if required - set(df.columns): raise ExecutionError(f"Missing columns: {sorted(required-set(df.columns))}.")
    if df[p["target"]].isna().any(): raise ExecutionError("Targets cannot be missing.")
    if p["task"] == "regression":
        y = pd.to_numeric(df[p["target"]], errors="coerce")
        if not np.isfinite(y).all(): raise ExecutionError("Regression targets must be finite numbers.")
        df[p["target"]] = y
    elif df[p["target"]].nunique() < 2:
        raise ExecutionError("Classification requires at least two target classes.")
    for col in df.select_dtypes(include=np.number):
        if np.isinf(df[col].to_numpy(dtype=float)).any(): raise ExecutionError(f"Column {col!r} contains infinity.")
    return df, raw

def make_split(df: pd.DataFrame, p: dict) -> dict[str, list[int]]:
    spec = p["split"]; n = len(df); ids = np.arange(n)
    a, b, c = spec["fractions"]; seed = spec["seed"]; strategy = spec["strategy"]
    try:
        if strategy in {"random", "stratified"}:
            strat = df[p["target"]] if strategy == "stratified" else None
            train, rest = train_test_split(ids, train_size=a, random_state=seed, stratify=strat)
            labels = df.iloc[rest][p["target"]] if strat is not None else None
            valid, test = train_test_split(rest, train_size=b/(b+c), random_state=seed, stratify=labels)
        elif strategy == "grouped":
            groups = df[spec["by"]]
            if groups.isna().any(): raise ExecutionError("Group identifiers cannot be missing.")
            first = GroupShuffleSplit(n_splits=1, train_size=a, random_state=seed)
            train, rest = next(first.split(ids, groups=groups))
            second = GroupShuffleSplit(n_splits=1, train_size=b/(b+c), random_state=seed)
            va, te = next(second.split(rest, groups=groups.iloc[rest]))
            valid, test = rest[va], rest[te]
        else:
            col = df[spec["by"]]
            if col.isna().any(): raise ExecutionError("Time values cannot be missing.")
            if pd.api.types.is_numeric_dtype(col): values = col.to_numpy(dtype=float)
            else:
                try: values = pd.to_datetime(col, utc=True, errors="raise").astype("int64").to_numpy()
                except (ValueError, TypeError) as exc: raise ExecutionError("Use numeric times or parseable UTC timestamps.") from exc
            order = np.argsort(values, kind="stable")
            cut1, cut2 = int(np.floor(a*n)), int(np.floor((a+b)*n))
            train, valid, test = order[:cut1], order[cut1:cut2], order[cut2:]
            if len(train) and len(valid) and len(test) and not (values[train].max() < values[valid].min() and values[valid].max() < values[test].min()):
                raise ExecutionError("A temporal boundary divides tied times. Change the split or aggregate tied observations.")
    except ValueError as exc:
        raise ExecutionError(f"Cannot create the requested split: {exc}") from exc
    parts = {"train": list(map(int, train)), "validation": list(map(int, valid)), "test": list(map(int, test))}
    validate_partitions(parts, n)
    if p["task"] == "classification" and df.iloc[parts["train"]][p["target"]].nunique() < 2:
        raise ExecutionError("The training partition must contain at least two classes.")
    return parts

def validate_partitions(parts: dict, n: int) -> None:
    if set(parts) != {"train", "validation", "test"}: raise ExecutionError("Three named partitions are required.")
    flat = [i for role in ("train", "validation", "test") for i in parts[role]]
    if any(type(i) is not int for i in flat) or len(flat) != n or sorted(flat) != list(range(n)):
        raise ExecutionError("Partitions must cover each input row exactly once.")
    if min(map(len, parts.values())) < 2: raise ExecutionError("Every partition must contain at least two rows.")

def make_preprocessor(df: pd.DataFrame, p: dict, train_ids: list[int]) -> tuple[ColumnTransformer, dict]:
    excluded = {p["target"], *p["drop"]}
    if p["split"]["by"]: excluded.add(p["split"]["by"])
    columns = [c for c in df.columns if c not in excluded]
    if not columns: raise ExecutionError("No predictor columns remain.")
    train = df.iloc[train_ids][columns]
    numeric = list(train.select_dtypes(include=np.number).columns)
    categorical = [c for c in columns if c not in numeric]
    # Dense one-hot encoding is deliberately bounded in this research prototype.
    if sum(train[c].nunique(dropna=True)+1 for c in categorical) + len(numeric) > 2_000:
        raise ExecutionError("One-hot output is limited to 2,000 columns.")
    if len(df)*(sum(train[c].nunique(dropna=True)+1 for c in categorical)+len(numeric)) > 10_000_000:
        raise ExecutionError("Dense transformed data would exceed the configured 10-million-cell limit.")
    transformers = []
    if numeric:
        steps: list = [("imputer", SimpleImputer(strategy=p["prepare"]["numeric"], keep_empty_features=True))]
        if p["prepare"]["scale"] == "standard": steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        imputer = SimpleImputer(strategy=p["prepare"]["categorical"], fill_value="__MISSING__", keep_empty_features=True)
        transformers.append(("categorical", Pipeline([("imputer", imputer), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(transformers, remainder="drop"), {"numeric": numeric, "categorical": categorical}

def make_estimator(m: dict, task: str, seed: int):
    a, params = m["algorithm"], m["params"]
    if a == "logistic": return LogisticRegression(C=params.get("C", 1.0), max_iter=2000, solver="lbfgs", random_state=seed)
    if a == "ridge": return Ridge(alpha=params.get("alpha", 1.0), solver="cholesky")
    if a == "forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=params.get("trees", 40), max_depth=params.get("depth", None), random_state=seed, n_jobs=1)
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(max_depth=params.get("depth", None), random_state=seed)

def score(y, pred, task: str) -> dict[str, float]:
    if task == "classification":
        return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    return {"rmse": float(root_mean_squared_error(y, pred)), "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}

def run_source(source: str, root: str | Path = ".") -> dict:
    start = time.perf_counter()
    plan = compile_source(source)
    parse_seconds = time.perf_counter()-start
    result = run_plan(plan, root)
    result["timings"]["compile_seconds"] = parse_seconds
    return result

def run_plan(plan: Plan, root: str | Path = ".", *, frozen_partitions: dict | None = None) -> dict:
    """The optional frozen partition argument is for controlled replication, not ordinary DSL programs."""
    start = time.perf_counter()
    # An immutable compiler result, not an unforgeable capability in hostile Python.
    if not isinstance(plan, Plan) or plan._validation_seal is not _VALIDATION_SEAL:
        raise ExecutionError("Compile source text before supplying a Plan to the execution API.")
    p = plan.as_dict()
    path = resolve_data_path(p["dataset"], root); df, raw = load_data(path, p)
    parts = copy.deepcopy(frozen_partitions) if frozen_partitions is not None else make_split(df, p)
    validate_partitions(parts, len(df))
    from .verifier import check_partition_semantics
    errors = check_partition_semantics(parts, p["split"], df.to_dict(orient="records"))
    if errors: raise ExecutionError("; ".join(errors))
    prep, schema = make_preprocessor(df, p, parts["train"])
    target = p["target"]; events: list[dict] = []
    append_event(events, "split", p["lines"]["split"], "all", list(range(len(df))), strategy=p["split"]["strategy"])
    validation: dict = {}; fitted: dict = {}; fingerprints: dict = {}; validation_predictions: dict = {}
    with threadpool_limits(limits=1):
        prep.fit(df.iloc[parts["train"]])
        prep_state = preprocessing_state(prep)
        append_event(events, "prepare_fit", p["lines"]["prepare"], "train", parts["train"], state_sha256=digest(prep_state), schema=schema)
        x_train = prep.transform(df.iloc[parts["train"]])
        append_event(events, "transform", p["lines"]["prepare"], "train", parts["train"])
        specs = {m["name"]: m for m in p["models"]}
        for action in p["actions"]:
            name = action["model"]; line = action["line"]
            if action["kind"] == "train":
                model = make_estimator(specs[name], p["task"], p["split"]["seed"])
                model.fit(x_train, df.iloc[parts["train"]][target])
                fitted[name] = model; fingerprints[name] = digest(estimator_state(model))
                append_event(events, "fit", line, "train", parts["train"], model=name, state_sha256=fingerprints[name])
            else:
                xv = prep.transform(df.iloc[parts["validation"]])
                append_event(events, "transform", line, "validation", parts["validation"], model=name)
                pred = fitted[name].predict(xv)
                validation_predictions[name] = normalize(pred)
                validation[name] = score(df.iloc[parts["validation"]][target], pred, p["task"])
                append_event(events, "evaluate", line, "validation", parts["validation"], model=name, metrics=validation[name], predictions_sha256=digest(normalize(pred)))
        metric = p["metric"]; direction = -1 if metric in {"rmse", "mae"} else 1
        # Python's max returns the first maximum; declaration order is the tie rule.
        winner = max([m["name"] for m in p["models"]], key=lambda name: direction*validation[name][metric])
        append_event(events, "select", p["lines"]["select"], "validation", parts["validation"], model=winner, metric=metric)
        xt = prep.transform(df.iloc[parts["test"]])
        append_event(events, "transform", p["lines"]["assess"], "test", parts["test"], model=winner)
        prediction = fitted[winner].predict(xt)
        final_metrics = score(df.iloc[parts["test"]][target], prediction, p["task"])
        append_event(events, "assess", p["lines"]["assess"], "test", parts["test"], model=winner, metrics=final_metrics, predictions_sha256=digest(normalize(prediction)))
    result = normalize({"format": "tracesplit-evidence-v1", "plan": p, "row_count": len(df), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                       "partitions": parts, "schema": schema, "events": events, "preprocessing": prep_state,
                       "model_fingerprints": fingerprints, "validation": validation, "validation_predictions": validation_predictions,
                       "selected": winner, "test": final_metrics, "test_predictions": normalize(prediction),
                       "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                                       "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform()},
                       "timings": {"execute_seconds": time.perf_counter()-start, "compile_seconds": None},
                       "frozen_partitions": frozen_partitions is not None})
    return result

def explain_result(result: dict) -> str:
    p = result["plan"]; parts = result["partitions"]
    lines = ["TraceSplit execution explanation", "", f"Dataset fingerprint: {result['dataset_sha256']}",
             f"Target: {p['target']} ({p['task']}).", f"Split strategy: {p['split']['strategy']}; seed: {p['split']['seed']}.",
             f"Rows: {len(parts['train'])} training, {len(parts['validation'])} validation, {len(parts['test'])} test.",
             "Preprocessing was fitted on the recorded training rows only.",
             f"Numeric predictors: {', '.join(result['schema']['numeric']) or '(none)' }.",
             f"Categorical predictors: {', '.join(result['schema']['categorical']) or '(none)' }."]
    for name, metrics in result["validation"].items(): lines.append(f"Candidate {name}: validation {p['metric']} = {metrics[p['metric']]:.8g}.")
    lines += [f"Selected candidate: {result['selected']}. Ties use declaration order.",
              "The selected model was assessed once in this run; it was not refitted on validation data.",
              "A separate run can reuse the same test set. This tool does not prevent that.",
              "These statements describe recorded operations, not causal feature explanations or a security attestation.", "", "Source-linked event record"]
    for e in result["events"]: lines.append(f"Line {e['line']}: {e['kind']} on {e['role']} ({len(e['rows'])} rows).")
    return "\n".join(lines) + "\n"

def save_result(result: dict, output: str | Path) -> Path:
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out/"evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (out/"explanation.txt").write_text(explain_result(result), encoding="utf-8")
    pd.DataFrame({"row_id": result["partitions"]["test"], "prediction": result["test_predictions"]}).to_csv(out/"predictions.csv", index=False)
    return out
