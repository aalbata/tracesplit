"""Injected-fault matrix for faulty execution backends.

Each fault changes the computation that produces the numerical outputs while
the recorded events keep the roles and row sets that the declared protocol
prescribes. Such a record echoes the protocol instead of the computation.
Every faulty execution is examined by the standard-library record checker
(with and without the CSV), by the standalone Python replay exported from the
record, and by the fixed-partition held-out feature perturbation. An honest
variant of the record, whose events describe the rows actually used, is
checked as well.
"""
from __future__ import annotations
import copy
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tracesplit import compile_source, run_plan
from tracesplit.engine import make_preprocessor, make_estimator, score
from tracesplit.evidence import digest, estimator_state, preprocessing_state, normalize
from tracesplit.exporter import compare_results, export_bundle
from tracesplit.verifier import verify

FAULTS = {
    "prepare_all_rows": {"prep": "all"},
    "prepare_train_validation": {"prep": "development"},
    "fit_train_validation": {"fit": "development"},
    "test_rows_in_training": {"fit": "train_test"},
    "select_by_test": {"select": "test"},
    "refit_before_assessment": {"refit": "development"},
    "undeclared_seed": {"seed_offset": 1},
    "partition_rule_ignored": {"partition": "random"},
    "target_proxy_column": {"data": "proxy"},
}
APPLIES = {"partition_rule_ignored": {"grouped", "temporal"}}


def rehash(e):
    previous = "0" * 64
    for i, event in enumerate(e["events"]):
        event["seq"] = i
        event["previous"] = previous
        event.pop("hash", None)
        event["hash"] = digest(event)
        previous = event["hash"]


def row_sets(parts, n):
    return {"train": list(parts["train"]), "validation": list(parts["validation"]), "test": list(parts["test"]),
            "development": list(parts["train"]) + list(parts["validation"]),
            "train_test": list(parts["train"]) + list(parts["test"]), "all": list(range(n))}


def execute(p, df, parts, *, prep="train", fit="train", select="validation", refit=None, seed_offset=0):
    """Run the supported computation with the requested deviations from the protocol."""
    rows = row_sets(parts, len(df))
    target, task, metric = p["target"], p["task"], p["metric"]
    preprocessor, _ = make_preprocessor(df, p, parts["train"])
    seed = p["split"]["seed"] + seed_offset
    out = {"validation": {}, "validation_predictions": {}, "model_fingerprints": {}}
    test_scores = {}
    models = {}
    specs = {m["name"]: m for m in p["models"]}
    with threadpool_limits(limits=1):
        preprocessor.fit(df.iloc[rows[prep]])
        out["preprocessing"] = preprocessing_state(preprocessor)
        x_fit = preprocessor.transform(df.iloc[rows[fit]])
        y_fit = df.iloc[rows[fit]][target]
        x_validation = preprocessor.transform(df.iloc[parts["validation"]])
        x_test = preprocessor.transform(df.iloc[parts["test"]])
        y_validation = df.iloc[parts["validation"]][target]
        y_test = df.iloc[parts["test"]][target]
        for m in p["models"]:
            model = make_estimator(m, task, seed)
            model.fit(x_fit, y_fit)
            models[m["name"]] = model
            out["model_fingerprints"][m["name"]] = digest(estimator_state(model))
            prediction = model.predict(x_validation)
            out["validation_predictions"][m["name"]] = normalize(prediction)
            out["validation"][m["name"]] = score(y_validation, prediction, task)
            test_scores[m["name"]] = score(y_test, model.predict(x_test), task)
        direction = -1 if metric in {"rmse", "mae"} else 1
        basis = out["validation"] if select == "validation" else test_scores
        winner = max([m["name"] for m in p["models"]], key=lambda name: direction * basis[name][metric])
        final = models[winner]
        out["refit_state"] = None
        if refit is not None:
            final = make_estimator(specs[winner], task, seed)
            final.fit(preprocessor.transform(df.iloc[rows[refit]]), df.iloc[rows[refit]][target])
            out["refit_state"] = digest(estimator_state(final))
        prediction = final.predict(x_test)
    out["selected"] = winner
    out["test_predictions"] = normalize(prediction)
    out["test"] = score(y_test, prediction, task)
    return out


def echo_record(base, out, parts=None):
    """Record whose events repeat the declared protocol while carrying the faulty outputs."""
    e = copy.deepcopy(base)
    if parts is not None:
        e["partitions"] = copy.deepcopy(parts)
    for key in ["preprocessing", "model_fingerprints", "validation", "validation_predictions", "selected", "test", "test_predictions"]:
        e[key] = copy.deepcopy(out[key])
    for event in e["events"]:
        kind, role = event["kind"], event["role"]
        event["rows"] = list(range(e["row_count"])) if role == "all" else list(e["partitions"][role])
        name = event["payload"].get("model")
        if kind == "prepare_fit":
            event["payload"]["state_sha256"] = digest(out["preprocessing"])
        elif kind == "fit":
            event["payload"]["state_sha256"] = out["model_fingerprints"][name]
        elif kind == "evaluate":
            event["payload"]["metrics"] = out["validation"][name]
            event["payload"]["predictions_sha256"] = digest(out["validation_predictions"][name])
        elif kind in {"select", "assess"} or (kind == "transform" and role == "test"):
            event["payload"]["model"] = out["selected"]
            if kind == "assess":
                event["payload"]["metrics"] = out["test"]
                event["payload"]["predictions_sha256"] = digest(out["test_predictions"])
    rehash(e)
    return e


def honest_record(echo, fault, out):
    """Record whose events describe the rows that the faulty computation actually used."""
    e = copy.deepcopy(echo)
    rows = row_sets(e["partitions"], e["row_count"])
    spec = FAULTS[fault]
    for event in e["events"]:
        if event["kind"] == "prepare_fit" and "prep" in spec:
            event["role"], event["rows"] = spec["prep"], rows[spec["prep"]]
        if event["kind"] == "fit" and "fit" in spec:
            event["role"], event["rows"] = spec["fit"], rows[spec["fit"]]
        if event["kind"] == "select" and spec.get("select") == "test":
            event["role"], event["rows"] = "test", rows["test"]
    if "refit" in spec:
        position = next(i for i, x in enumerate(e["events"]) if x["kind"] == "select") + 1
        e["events"].insert(position, {"seq": 0, "kind": "fit", "line": e["plan"]["lines"]["assess"], "role": spec["refit"],
                                      "rows": rows[spec["refit"]], "payload": {"model": out["selected"], "state_sha256": out["refit_state"]}, "previous": ""})
    if "seed_offset" in spec:
        e["plan"]["split"]["seed"] += spec["seed_offset"]
    if "partition" in spec:
        e["plan"]["drop"] = e["plan"]["drop"] + [e["plan"]["split"]["by"]]
        e["plan"]["split"]["strategy"] = spec["partition"]
        e["plan"]["split"]["by"] = None
    rehash(e)
    return e


def random_partition(parts, seed):
    sizes = {role: len(parts[role]) for role in ("train", "validation", "test")}
    n = sum(sizes.values())
    order = np.random.default_rng(seed + 1_000_003).permutation(n)
    a, b = sizes["train"], sizes["train"] + sizes["validation"]
    return {"train": [int(i) for i in order[:a]], "validation": [int(i) for i in order[a:b]], "test": [int(i) for i in order[b:]]}


def proxy_frame(df, p, seed):
    """Add a predictor that is a noisy copy of the target to the data itself."""
    rng = np.random.default_rng(seed + 7_919)
    frame = df.copy(deep=True)
    target = frame[p["target"]]
    if p["task"] == "regression":
        frame["proxy"] = target.astype(float) + rng.normal(scale=0.01 * float(target.std() or 1.0), size=len(frame))
    else:
        labels = {v: i for i, v in enumerate(sorted(target.unique()))}
        frame["proxy"] = target.map(labels).astype(float) + rng.normal(scale=0.05, size=len(frame))
    return frame


def changed_test_features(df, schema, parts):
    changed = df.copy(deep=True)
    ids = parts["test"]
    for col in schema["numeric"]:
        changed[col] = changed[col].astype(float)
        changed.loc[ids, col] = changed.loc[ids, col] * 11.0 + 17.0
    for col in schema["categorical"]:
        changed.loc[ids, col] = "__HOLDOUT_ONLY_CATEGORY__"
    return changed


def invariants(a, b):
    return {"preprocessing_unchanged": a["preprocessing"] == b["preprocessing"],
            "fitted_states_unchanged": a["model_fingerprints"] == b["model_fingerprints"],
            "validation_unchanged": a["validation"] == b["validation"],
            "selection_unchanged": a["selected"] == b["selected"]}


def replay_from_record(record, root, directory):
    exported = export_bundle(record, root, directory)
    spec = importlib.util.spec_from_file_location("replay_" + directory.name, exported / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run()


def evaluate_case(fault, result, df, root, atol, rtol, temp):
    """Apply one fault to one correct run and return the oracle outcomes."""
    p = result["plan"]
    spec = dict(FAULTS[fault])
    parts = result["partitions"]
    frame = df
    data_root = root
    if spec.get("data") == "proxy":
        frame = proxy_frame(df, p, p["split"]["seed"])
        data_root = Path(temp) / "proxy_root"
        data_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(data_root / "data.csv", index=False)
        source = p["source"].replace(p["dataset"], "data.csv")
        honest = run_plan(compile_source(source), data_root, frozen_partitions=parts)
        out = {k: honest[k] for k in ["preprocessing", "model_fingerprints", "validation", "validation_predictions", "selected", "test", "test_predictions"]}
        out["refit_state"] = None
        echo = copy.deepcopy(honest)
        echo["frozen_partitions"] = False
        frame = pd.read_csv(data_root / "data.csv", float_precision="round_trip")
        schema = honest["schema"]
    else:
        if "partition" in spec:
            parts = random_partition(result["partitions"], p["split"]["seed"])
        settings = {k: v for k, v in spec.items() if k in {"prep", "fit", "select", "refit", "seed_offset"}}
        out = execute(p, frame, parts, **settings)
        echo = echo_record(result, out, parts if "partition" in spec else None)
        schema = result["schema"]
    csv_path = data_root / (echo["plan"]["dataset"])
    honest = honest_record(echo, fault, out) if spec.get("data") != "proxy" else echo
    replay = replay_from_record(echo, data_root, Path(temp) / f"replay_{fault}")
    comparison = compare_results(echo, replay, atol, rtol)
    settings = {k: v for k, v in spec.items() if k in {"prep", "fit", "select", "refit", "seed_offset"}}
    perturbed_frame = changed_test_features(frame, schema, parts)
    if spec.get("data") == "proxy":
        perturbed_frame.to_csv(data_root / "perturbed.csv", index=False)
        perturbed = run_plan(compile_source(p["source"].replace(p["dataset"], "perturbed.csv")), data_root, frozen_partitions=parts)
    else:
        perturbed = execute(p, perturbed_frame, parts, **settings)
    correct_outputs = {k: result[k] for k in ["selected", "validation", "validation_predictions", "test", "test_predictions"]}
    faulty_outputs = {k: echo[k] for k in correct_outputs}
    same_data = spec.get("data") != "proxy"
    return {"outputs_differ_from_correct": (not compare_results(correct_outputs, faulty_outputs, atol, rtol)["ok"]) if same_data else None,
            "selected_differs_from_correct": (echo["selected"] != result["selected"]) if same_data else None,
            "honest_record_rejected_with_csv": not verify(honest, csv_path)["ok"],
            "echo_record_rejected_with_csv": not verify(echo, csv_path)["ok"],
            "echo_record_rejected_without_csv": not verify(echo)["ok"],
            "replay_detected": not comparison["ok"],
            "perturbation_detected": not all(invariants(echo, perturbed).values())}


def run_fault_matrix(out, config, root):
    """Write per-case and per-fault outcomes for every fault applied to every saved correct run."""
    rows = []
    for name in config["datasets"]:
        df = pd.read_csv(root / "datasets" / (name + ".csv"), float_precision="round_trip")
        for seed in config["seeds"]:
            key = f"{name}_seed_{seed}"
            result = __import__("json").loads((out / "runs" / key / "evidence.json").read_text())
            split = result["plan"]["split"]["strategy"]
            for fault in FAULTS:
                if fault in APPLIES and split not in APPLIES[fault]:
                    continue
                with tempfile.TemporaryDirectory(prefix="tracesplit-fault-") as temp:
                    outcome = evaluate_case(fault, result, df, root, config["prediction_atol"], config["prediction_rtol"], temp)
                rows.append({"case": key, "dataset": name, "seed": seed, "fault": fault, **outcome})
            print(f"Fault matrix {key}: {sum(1 for r in rows if r['case'] == key)} faults", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "fault_matrix.csv", index=False)
    columns = [c for c in table.columns if c not in {"case", "dataset", "seed", "fault"}]
    summary = table.groupby("fault", sort=False)[columns].sum(min_count=1)
    summary = summary.astype("Int64")
    summary.insert(0, "cases", table.groupby("fault", sort=False).size())
    summary = summary.loc[list(FAULTS)].reset_index()
    summary.to_csv(out / "fault_matrix_summary.csv", index=False)
    return summary
