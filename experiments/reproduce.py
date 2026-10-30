"""Run the complete constructed evaluation and preserve every case-level result.

This is a software-conformance study, not a human study or accuracy competition.
There are no network calls and no GPUs. `--quick` is a smaller smoke experiment,
not a substitute for the full manuscript results.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from tracesplit import compile_source, run_plan, run_source, LanguageError
from tracesplit.engine import make_preprocessor, make_estimator, score, save_result
from tracesplit.evidence import digest, estimator_state, preprocessing_state, normalize
from tracesplit.exporter import export_bundle, compare_results
from tracesplit.verifier import verify
from corpus import make_cases, write_corpus
from fault_matrix import run_fault_matrix

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(value), indent=2, allow_nan=False)+"\n")

def rehash(e):
    previous = "0"*64
    for i, event in enumerate(e["events"]):
        event["seq"] = i; event["previous"] = previous; event.pop("hash", None)
        event["hash"] = digest(event); previous = event["hash"]

def witness_mutants(original):
    def item(): return copy.deepcopy(original)
    mutations = {}
    for kind in ["prepare_fit", "fit", "evaluate", "assess"]:
        e = item(); event = next(x for x in e["events"] if x["kind"] == kind)
        wrong = "test" if event["role"] != "test" else "train"
        event["role"], event["rows"] = wrong, e["partitions"][wrong].copy()
        rehash(e); mutations[kind+"_scope"] = e
    e = item(); e["partitions"]["test"][0] = e["partitions"]["train"][0]; rehash(e); mutations["partition_overlap"] = e
    e = item(); e["partitions"]["test"].pop(); rehash(e); mutations["partition_omission"] = e
    e = item(); e["events"].append(copy.deepcopy(e["events"][-1])); rehash(e); mutations["second_assessment"] = e
    e = item(); wrong = next(m["name"] for m in e["plan"]["models"] if m["name"] != e["selected"])
    e["selected"] = wrong
    for event in e["events"][-3:]: event["payload"]["model"] = wrong
    rehash(e); mutations["wrong_selection"] = e
    e = item(); e["plan"]["source_sha256"] = "0"*64; rehash(e); mutations["source_fingerprint"] = e
    e = item(); e["events"][1]["line"] = 9999; rehash(e); mutations["source_location"] = e
    e = item(); metric = e["plan"]["metric"]; e["test"][metric] += 1.0
    e["events"][-1]["payload"]["metrics"] = copy.deepcopy(e["test"]); rehash(e); mutations["metric_recomputed_chain"] = e
    e = item(); e["dataset_sha256"] = "0"*64; rehash(e); mutations["dataset_fingerprint"] = e
    e = item(); name = e["plan"]["models"][0]["name"]; e["model_fingerprints"][name] = "0"*64; rehash(e); mutations["model_fingerprint"] = e
    e = item(); e["preprocessing"]["unexpected"] = True; rehash(e); mutations["preprocessing_fingerprint"] = e
    e = item(); e["events"][0]["hash"] = "0"*64; mutations["broken_chain"] = e
    return mutations

def leaky_backend(base, df):
    """Deliberately faulty negative control. NEVER called by the language engine."""
    p = base["plan"]; ids = base["partitions"]; e = copy.deepcopy(base)
    prep, _ = make_preprocessor(df, p, ids["train"])
    with threadpool_limits(limits=1):
        prep.fit(df)  # INJECTED FAULT: fitted preprocessing sees the whole dataset.
        e["preprocessing"] = preprocessing_state(prep)
        xt, xv, xh = [prep.transform(df.iloc[ids[role]]) for role in ["train", "validation", "test"]]
        models = {}; e["validation"] = {}; e["validation_predictions"] = {}; e["model_fingerprints"] = {}
        for m in p["models"]:
            model = make_estimator(m, p["task"], p["split"]["seed"])
            model.fit(xt, df.iloc[ids["train"]][p["target"]]); models[m["name"]] = model
            pred = model.predict(xv); e["validation_predictions"][m["name"]] = normalize(pred)
            e["validation"][m["name"]] = score(df.iloc[ids["validation"]][p["target"]], pred, p["task"])
            e["model_fingerprints"][m["name"]] = digest(estimator_state(model))
        direction = -1 if p["metric"] in {"rmse", "mae"} else 1
        e["selected"] = max([m["name"] for m in p["models"]], key=lambda name: direction*e["validation"][name][p["metric"]])
        pred = models[e["selected"]].predict(xh); e["test_predictions"] = normalize(pred)
        e["test"] = score(df.iloc[ids["test"]][p["target"]], pred, p["task"])
    for event in e["events"]:
        kind = event["kind"]; name = event["payload"].get("model")
        if kind == "prepare_fit":
            event["role"] = "all"; event["rows"] = list(range(len(df))); event["payload"]["state_sha256"] = digest(e["preprocessing"])
        elif kind == "fit": event["payload"]["state_sha256"] = e["model_fingerprints"][name]
        elif kind == "evaluate":
            event["payload"]["metrics"] = e["validation"][name]
            event["payload"]["predictions_sha256"] = digest(e["validation_predictions"][name])
        elif kind in {"select", "assess"} or (kind == "transform" and event["role"] == "test"):
            event["payload"]["model"] = e["selected"]
            if kind == "assess":
                event["payload"]["metrics"] = e["test"]
                event["payload"]["predictions_sha256"] = digest(e["test_predictions"])
    rehash(e)
    return e

def changed_test_features(df, result):
    changed = df.copy(deep=True); ids = result["partitions"]["test"]
    for col in result["schema"]["numeric"]:
        changed[col] = changed[col].astype(float)
        changed.loc[ids, col] = changed.loc[ids, col]*11.0 + 17.0
    for col in result["schema"]["categorical"]:
        changed.loc[ids, col] = "__HOLDOUT_ONLY_CATEGORY__"
    return changed

def invariants(a, b):
    checks = {"preprocessing_unchanged": a["preprocessing"] == b["preprocessing"],
              "fitted_states_unchanged": a["model_fingerprints"] == b["model_fingerprints"],
              "validation_unchanged": a["validation"] == b["validation"],
              "selection_unchanged": a["selected"] == b["selected"]}
    return {**checks, "all_pass": all(checks.values())}

def bootstrap_median_ci(values, count, seed):
    rng = np.random.default_rng(seed); x = np.asarray(values)
    estimates = np.median(x[rng.integers(0, len(x), size=(count, len(x)))], axis=1)
    return list(map(float, np.quantile(estimates, [.025, .975])))

def runtime_experiment(config, out, only_rows=None):
    runtime_rows = []; runtime_summary = []
    with tempfile.TemporaryDirectory(prefix="tracesplit-runtime-") as temp:
        temp = Path(temp)
        for n in ([only_rows] if only_rows else config["runtime_rows"]):
            X, y = make_classification(n_samples=n, n_features=12, n_informative=8, n_redundant=2, random_state=818)
            frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)]); frame["target"] = y; frame.to_csv(temp/"data.csv", index=False)
            source = '''read "data.csv"
predict "target" as classification
split stratified train 60% validation 20% test 20% seed 42
prepare numeric median scale standard categorical most_frequent on train
train linear logistic C=1 on train
evaluate linear on validation
select best by accuracy
assess best on test
'''
            plan = compile_source(source)
            for _ in range(config["runtime_warmups"]): run_source(source, temp); run_plan(plan, temp)
            print(f"Runtime block {n} rows", flush=True)
            for repetition in range(config["runtime_repetitions"]):
                times = {}
                order = ["language", "plan_api"] if repetition%2 == 0 else ["plan_api", "language"]
                for method in order:
                    t0 = time.perf_counter()
                    output = run_source(source, temp) if method == "language" else run_plan(plan, temp)
                    times[method] = (time.perf_counter()-t0)*1000
                if repetition % 5 == 0: print(f"  paired repetition {repetition}", flush=True)
                runtime_rows.append({"rows": n, "repetition": repetition, "order": "/".join(order),
                                     "language_ms": times["language"], "plan_api_ms": times["plan_api"], "difference_ms": times["language"]-times["plan_api"]})
            subset = [r for r in runtime_rows if r["rows"] == n]
            diffs = [r["difference_ms"] for r in subset]
            runtime_summary.append({"rows": n, "pairs": len(subset), "median_language_ms": statistics.median(r["language_ms"] for r in subset),
                                    "median_plan_api_ms": statistics.median(r["plan_api_ms"] for r in subset), "median_paired_difference_ms": statistics.median(diffs),
                                    "paired_difference_95pct_bootstrap_ci_ms": bootstrap_median_ci(diffs, config["bootstrap_resamples"], config["bootstrap_seed"]+n)})
            pd.DataFrame(subset).to_csv(out/f"runtime_pairs_{n}.csv", index=False)
            write_json(out/f"runtime_summary_{n}.json", runtime_summary[-1])
    paths = [out/f"runtime_pairs_{n}.csv" for n in config["runtime_rows"]]
    if not all(path.exists() for path in paths): return False
    pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(out/"runtime_pairs.csv", index=False)
    write_json(out/"runtime_summary.json", [json.loads((out/f"runtime_summary_{n}.json").read_text()) for n in config["runtime_rows"]])
    return True


def fault_summary(out):
    path = out/"fault_matrix_summary.csv"
    if not path.exists(): return None
    return [{k: (v if k == "fault" else (None if pd.isna(v) else int(v))) for k, v in row.items()} for row in pd.read_csv(path).to_dict(orient="records")]


def processor_name():
    """Read the processor model string from the operating system."""
    if Path("/proc/cpuinfo").exists():
        return next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--quick", action="store_true"); parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["all", "cases", "faults", "runtime"], default="all")
    parser.add_argument("--rows", type=int, choices=[200, 2000, 10000])
    args = parser.parse_args(argv); config = json.loads((ROOT/"experiments"/"config.json").read_text())
    out = args.output or ROOT/"results"/("quick" if args.quick else "full")
    out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    if args.quick:
        config["seeds"] = [42]; config["datasets"] = ["iris", "mixed", "diabetes"]
        config["runtime_rows"] = [200]; config["runtime_repetitions"] = 3; config["runtime_warmups"] = 1
    write_json(out/"config_used.json", config)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.rglob("*"))
              if path.is_file() and path.suffix in {".py", ".lark"} and "results" not in path.parts and "__pycache__" not in path.parts}
    write_json(out/f"source_snapshot_{args.stage}.json", hashes)
    cpu = processor_name()
    environment = {"python": sys.version, "platform": platform.platform(), "cpu": cpu, "logical_cpus_visible": os.cpu_count(),
                   "blas_threads_requested": 1, "clock_utc_reported_by_runtime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "source_snapshot_sha256": digest(hashes), "quick_mode": args.quick}
    write_json(out/f"environment_{args.stage}.json", environment)
    if args.stage in {"all", "cases"}:
        # 1. The constructed syntax and semantic conformance corpus.
        write_corpus(); rows = []
        for case in make_cases():
            accepted, error = True, ""
            try: compile_source(case["source"])
            except LanguageError as exc: accepted, error = False, exc.code
            rows.append({k: case[k] for k in ["id", "valid", "family"]} | {"accepted": accepted, "correct": accepted == case["valid"], "error": error})
        pd.DataFrame(rows).to_csv(out/"compiler_cases.csv", index=False)
        # 2. Separate Python replay, evidence mutants, and fixed-split perturbations.
        agreement = []; witness_rows = []; meta_rows = []; negative_rows = []; verify_times = []; numeric_rows = []
        for name in config["datasets"]:
            for seed in config["seeds"]:
                key = f"{name}_seed_{seed}"; source = (ROOT/"examples"/(name+".tsml")).read_text().replace("seed 42", f"seed {seed}")
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    result = run_source(source, ROOT)
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise RuntimeError(f"Convergence warning in {key}; do not silently publish this run.")
                directory = out/"runs"/key; save_result(result, directory)
                dataset_path = ROOT/"datasets"/(name+".csv")
                t0 = time.perf_counter(); report = verify(result, dataset_path); elapsed = time.perf_counter()-t0
                write_json(directory/"verification.json", report); verify_times.append({"case": key, "seconds": elapsed, "checks": report["checks"]})
                if not report["ok"]: raise RuntimeError(report)
                exported = export_bundle(result, ROOT, directory/"python_export")
                spec = importlib.util.spec_from_file_location("replay_"+key, exported/"experiment.py")
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                replay = module.run(); write_json(directory/"replay_result.json", replay)
                comparison = compare_results(result, replay, config["prediction_atol"], config["prediction_rtol"])
                agreement.append({"case": key, "dataset": name, "seed": seed, "candidates": len(result["plan"]["models"]), **comparison})
                metric = result["plan"]["metric"]
                numeric_rows.append({"dataset": name, "seed": seed, "metric": metric, "selected": result["selected"], "test_score": result["test"][metric]})
                for family, mutated in witness_mutants(result).items():
                    with_data = verify(mutated, dataset_path); without_data = verify(mutated)
                    witness_rows.append({"case": key, "family": family, "rejected_with_data": not with_data["ok"], "rejected_without_data": not without_data["ok"],
                                         "errors_with_data": " | ".join(with_data["errors"])})
                frame = pd.read_csv(dataset_path, float_precision="round_trip"); feature_change = changed_test_features(frame, result)
                label_change = frame.copy(deep=True); ids = result["partitions"]["test"]; target = result["plan"]["target"]
                if result["plan"]["task"] == "classification":
                    labels = sorted(frame[target].unique()); mapping = {v: labels[(i+1)%len(labels)] for i,v in enumerate(labels)}
                    label_change.loc[ids, target] = label_change.loc[ids, target].map(mapping)
                else:
                    label_change[target] = label_change[target].astype(float)
                    label_change.loc[ids, target] += 17.0
                with tempfile.TemporaryDirectory(prefix="tracesplit-counterfactual-") as temp:
                    temp = Path(temp)
                    for mode, altered in [("features", feature_change), ("labels", label_change)]:
                        altered.to_csv(temp/"data.csv", index=False)
                        text = source.replace(f"datasets/{name}.csv", "data.csv")
                        changed = run_plan(compile_source(text), temp, frozen_partitions=result["partitions"])
                        meta_rows.append({"case": key, "mutation": mode, **invariants(result, changed)})
                leaked = leaky_backend(result, frame)
                leaked_changed = leaky_backend(result, feature_change)
                honest_report = verify(leaked, dataset_path)
                forged = copy.deepcopy(leaked)
                prep_event = next(x for x in forged["events"] if x["kind"] == "prepare_fit")
                prep_event["role"], prep_event["rows"] = "train", forged["partitions"]["train"].copy(); rehash(forged)
                forged_report = verify(forged, dataset_path)
                numeric_difference = not compare_results(result, leaked, config["prediction_atol"], config["prediction_rtol"])["ok"]
                negative_rows.append({"case": key, "honest_leaky_trace_rejected": not honest_report["ok"],
                                      "fabricated_leaky_trace_accepted": forged_report["ok"],
                                      "replay_output_difference_detected": numeric_difference,
                                      "holdout_perturbation_detected": not invariants(leaked, leaked_changed)["all_pass"]})
                if seed == config["seeds"][0]:
                    write_json(directory/"negative_control_honest_leaky.json", leaked)
                    write_json(directory/"negative_control_fabricated_leaky.json", forged)
                print(f"Completed {key}: replay={comparison['ok']}, trace={report['ok']}", flush=True)
        pd.DataFrame(agreement).drop(columns=["errors"]).to_csv(out/"replay_agreement.csv", index=False)
        pd.DataFrame(witness_rows).to_csv(out/"witness_mutations.csv", index=False)
        pd.DataFrame(meta_rows).to_csv(out/"metamorphic_cases.csv", index=False)
        pd.DataFrame(negative_rows).to_csv(out/"negative_controls.csv", index=False)
        pd.DataFrame(verify_times).to_csv(out/"verification_times.csv", index=False)
        pd.DataFrame(numeric_rows).to_csv(out/"test_scores_descriptive.csv", index=False)
    if args.stage in {"all", "faults"}:
        # 3. Injected-fault matrix over every saved correct run.
        run_fault_matrix(out, config, ROOT)
    if args.stage == "faults":
        print("Fault matrix files are saved.", flush=True)
        return 0
    if args.stage == "cases":
        print("All numerical, witness, and counterfactual case files are saved.", flush=True)
        return 0
    rows = pd.read_csv(out/"compiler_cases.csv").to_dict(orient="records")
    agreement = pd.read_csv(out/"replay_agreement.csv").to_dict(orient="records")
    witness_rows = pd.read_csv(out/"witness_mutations.csv").to_dict(orient="records")
    meta_rows = pd.read_csv(out/"metamorphic_cases.csv").to_dict(orient="records")
    negative_rows = pd.read_csv(out/"negative_controls.csv").to_dict(orient="records")
    verify_times = pd.read_csv(out/"verification_times.csv").to_dict(orient="records")
    if not runtime_experiment(config, out, args.rows): return 0
    runtime_summary = json.loads((out/"runtime_summary.json").read_text())
    summary = {"compiler": {"valid": sum(r["valid"] for r in rows), "invalid": sum(not r["valid"] for r in rows), "correct": sum(r["correct"] for r in rows), "total": len(rows)},
               "replay": {"workflows": len(agreement), "candidate_fits_per_execution_path": sum(r["candidates"] for r in agreement), "passed": sum(r["ok"] for r in agreement),
                          "max_absolute_difference": max(r["max_absolute_difference"] for r in agreement)},
               "witness_mutations": {"families": len(set(r["family"] for r in witness_rows)), "total": len(witness_rows),
                                      "rejected_with_data": sum(r["rejected_with_data"] for r in witness_rows), "rejected_without_data": sum(r["rejected_without_data"] for r in witness_rows)},
               "metamorphic": {"total": len(meta_rows), "passed": sum(r["all_pass"] for r in meta_rows)},
               "negative_controls": {"total": len(negative_rows), **{k: sum(r[k] for r in negative_rows) for k in ["honest_leaky_trace_rejected", "fabricated_leaky_trace_accepted", "replay_output_difference_detected", "holdout_perturbation_detected"]}},
               "fault_matrix": fault_summary(out),
               "verification_median_ms": 1000*statistics.median(r["seconds"] for r in verify_times),
               "runtime": runtime_summary, "last_stage_elapsed_seconds": time.perf_counter()-started,
               "no_human_study": True, "no_external_tool_head_to_head": True, "quick_mode": args.quick}
    write_json(out/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    failures = summary["compiler"]["correct"] != summary["compiler"]["total"] or summary["replay"]["passed"] != len(agreement) or summary["metamorphic"]["passed"] != len(meta_rows)
    return 1 if failures else 0

if __name__ == "__main__": raise SystemExit(main())
