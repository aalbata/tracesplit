"""Standalone standard-library evidence checker.

Run directly with Python; neither TraceSplit nor an ML library is imported.
A valid record is internally consistent evidence, NOT proof that a machine
actually executed the recorded calls. Hashes are unkeyed, not signatures.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROLES = ("train", "validation", "test")

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def _time_value(x):
    if isinstance(x, (int, float)): return float(x)
    try: return float(x)
    except (ValueError, TypeError):
        parsed = dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()

def check_partition_semantics(parts, split, records):
    errors = []
    strategy, by = split["strategy"], split.get("by")
    if strategy == "grouped":
        values = [{str(records[i][by]) for i in parts[r]} for r in ROLES]
        if any(values[i] & values[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            errors.append("Group identifiers cross partition boundaries.")
    elif strategy == "temporal":
        try:
            values = [[_time_value(records[i][by]) for i in parts[r]] for r in ROLES]
            if not all(math.isfinite(x) for v in values for x in v): errors.append("Time values must be finite.")
            if not (max(values[0]) < min(values[1]) and max(values[1]) < min(values[2])):
                errors.append("Temporal partitions do not have strictly ordered boundaries.")
        except (ValueError, TypeError, KeyError): errors.append("Invalid temporal metadata.")
    return errors

def _equal_metrics(a, b):
    return set(a) == set(b) and all(isinstance(a[k], (int, float)) and not isinstance(a[k], bool)
                                  and math.isfinite(a[k]) and math.isclose(a[k], b[k], rel_tol=1e-10, abs_tol=1e-10) for k in a)

def independent_metrics(truth, pred, task):
    """Small arithmetic implementation, independent of sklearn.metrics."""
    if len(truth) != len(pred) or len(truth) < 2: raise ValueError("Prediction lengths do not match the partition.")
    if task == "regression":
        truth, pred = list(map(float, truth)), list(map(float, pred))
        if not all(map(math.isfinite, truth+pred)): raise ValueError("Non-finite targets or predictions.")
        n = len(truth); residual = sum((a-b)**2 for a, b in zip(truth, pred))
        mean = sum(truth)/n; denom = sum((a-mean)**2 for a in truth)
        r2 = 1-residual/denom if denom else (1.0 if residual == 0 else 0.0)
        return {"rmse": math.sqrt(residual/n), "mae": sum(abs(a-b) for a, b in zip(truth, pred))/n, "r2": r2}
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in pred)
    truth = [float(x) for x in truth] if numeric else [str(x) for x in truth]
    pred = [float(x) for x in pred] if numeric else [str(x) for x in pred]
    labels = set(truth) | set(pred)
    true_counts, pred_counts = Counter(truth), Counter(pred)
    correct = Counter(a for a, b in zip(truth, pred) if a == b)
    return {"accuracy": sum(correct.values())/len(truth),
            "balanced_accuracy": sum(correct[c]/true_counts[c] for c in true_counts)/len(true_counts),
            "macro_f1": sum(2*correct[c]/(true_counts[c]+pred_counts[c]) for c in labels)/len(labels)}

def verify(evidence, data_path=None):
    errors = []; checks = 0; bound = False
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition: errors.append(message)
    try:
        e = evidence; p = e["plan"]; parts = e["partitions"]; n = e["row_count"]
        check(e["format"] == "tracesplit-evidence-v1", "Unsupported evidence format.")
        check(type(n) is int and 15 <= n <= 100_000, "Invalid row count.")
        check(set(parts) == set(ROLES), "Missing or extra partition roles.")
        flat = [i for r in ROLES for i in parts[r]]
        valid_ids = all(type(i) is int for i in flat)
        check(valid_ids and len(flat) == n and sorted(flat) == list(range(n)), "Partitions must cover each row exactly once.")
        check(all(len(parts[r]) >= 2 for r in ROLES), "A partition contains fewer than two rows.")
        check(hashlib.sha256(p["source"].encode()).hexdigest() == p["source_sha256"], "Source fingerprint mismatch.")
        check(p["task"] in {"classification", "regression"}, "Invalid task.")
        check(p["split"]["strategy"] in {"random", "stratified", "grouped", "temporal"}, "Invalid splitting strategy.")
        fractions = p["split"]["fractions"]
        check(len(fractions) == 3 and all(0 < x < 1 for x in fractions) and math.isclose(sum(fractions), 1, abs_tol=1e-12), "Invalid split fractions.")
        check(p["split"]["strategy"] != "stratified" or p["task"] == "classification", "Regression cannot use stratification.")
        check(p["prepare"]["numeric"] in {"mean", "median", "most_frequent"} and p["prepare"]["scale"] in {"none", "standard"}
              and p["prepare"]["categorical"] in {"most_frequent", "constant"}, "Invalid preprocessing specification.")
        valid_metrics = {"accuracy", "balanced_accuracy", "macro_f1"} if p["task"] == "classification" else {"rmse", "mae", "r2"}
        check(p["metric"] in valid_metrics, "Invalid selection metric.")
        names = [m["name"] for m in p["models"]]
        check(1 <= len(names) <= 12 and len(set(names)) == len(names), "Candidate names must be unique and bounded.")
        allowed_algorithms = {"logistic", "forest", "tree"} if p["task"] == "classification" else {"ridge", "forest", "tree"}
        check(all(m["algorithm"] in allowed_algorithms for m in p["models"]), "An algorithm is incompatible with the task.")
        schema = e["schema"]; columns = schema["numeric"]+schema["categorical"]
        excluded = {p["target"], *p["drop"]}
        if p["split"].get("by"): excluded.add(p["split"]["by"])
        check(bool(columns) and len(columns) == len(set(columns)) and not (set(columns)&excluded), "Invalid predictor schema or excluded predictor present.")
        expected = [("split", p["lines"]["split"], "all", None),
                    ("prepare_fit", p["lines"]["prepare"], "train", None),
                    ("transform", p["lines"]["prepare"], "train", None)]
        trained, evaluated = set(), set()
        model_lines = {m["name"]: m["line"] for m in p["models"]}
        for action in p["actions"]:
            name, line = action["model"], action["line"]
            check(name in names, "Action uses an undeclared candidate.")
            if action["kind"] == "train":
                check(name not in trained and line == model_lines.get(name), "Repeated training or inconsistent model source line.")
                trained.add(name); expected.append(("fit", line, "train", name))
            elif action["kind"] == "evaluate":
                check(name in trained and name not in evaluated, "Evaluation precedes fitting or is repeated.")
                evaluated.add(name)
                expected += [("transform", line, "validation", name), ("evaluate", line, "validation", name)]
            else: check(False, "Unknown action.")
        check(trained == evaluated == set(names), "All candidates must be trained and validated exactly once.")
        check(set(e["validation"]) == set(names) and set(e["model_fingerprints"]) == set(names)
              and set(e["validation_predictions"]) == set(names), "Candidate result sets are inconsistent.")
        for name in names:
            check(set(e["validation"][name]) == valid_metrics and all(isinstance(x, (int, float)) and math.isfinite(x) for x in e["validation"][name].values()), "Invalid validation metrics.")
            check(len(e["validation_predictions"][name]) == len(parts["validation"]), "Validation prediction length mismatch.")
        sign = -1 if p["metric"] in {"rmse", "mae"} else 1
        winner = max(names, key=lambda name: sign*e["validation"][name][p["metric"]])
        check(e["selected"] == winner, "Selected candidate violates the metric or declaration-order tie rule.")
        expected += [("select", p["lines"]["select"], "validation", winner),
                     ("transform", p["lines"]["assess"], "test", winner), ("assess", p["lines"]["assess"], "test", winner)]
        events = e["events"]; check(len(events) == len(expected), "Event count does not match the workflow protocol.")
        previous = "0"*64
        source_lines = p["source"].splitlines()
        for idx, event in enumerate(events):
            reported_hash = event.get("hash")
            item = {k: v for k, v in event.items() if k != "hash"}
            check(event["seq"] == idx and event["previous"] == previous and canonical_hash(item) == reported_hash, "Broken event ordering or hash chain.")
            previous = reported_hash
            if idx >= len(expected): continue
            kind, line, role, model = expected[idx]
            check((event["kind"], event["line"], event["role"]) == (kind, line, role), "Event kind, source line, or role does not match the protocol.")
            check(type(line) is int and 1 <= line <= len(source_lines), "Source line outside the program.")
            rows = list(range(n)) if role == "all" else parts[role]
            check(event["rows"] == rows, "Recorded row use violates the required partition scope.")
            check(event["payload"].get("model") == model, "Event refers to the wrong candidate.")
            if kind == "prepare_fit":
                check(event["payload"].get("state_sha256") == canonical_hash(e["preprocessing"]), "Preprocessing fingerprint mismatch.")
                check(event["payload"].get("schema") == schema, "Preprocessing schema mismatch.")
            elif kind == "fit": check(event["payload"].get("state_sha256") == e["model_fingerprints"][model], "Model fingerprint mismatch.")
            elif kind == "evaluate":
                check(event["payload"].get("metrics") == e["validation"][model], "Validation record mismatch.")
                check(event["payload"].get("predictions_sha256") == canonical_hash(e["validation_predictions"][model]), "Validation prediction fingerprint mismatch.")
            elif kind == "select": check(event["payload"].get("metric") == p["metric"], "Selection metric mismatch.")
            elif kind == "assess":
                check(event["payload"].get("metrics") == e["test"], "Test metric record mismatch.")
                check(event["payload"].get("predictions_sha256") == canonical_hash(e["test_predictions"]), "Test prediction fingerprint mismatch.")
        check(set(e["test"]) == valid_metrics and len(e["test_predictions"]) == len(parts["test"]), "Invalid final result shape.")
        if data_path is not None:
            raw = Path(data_path).read_bytes()
            bound = hashlib.sha256(raw).hexdigest() == e["dataset_sha256"]
            check(bound, "Dataset fingerprint mismatch.")
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            check(len(rows) == n, "Input row count mismatch.")
            if valid_ids and sorted(flat) == list(range(len(rows))):
                errors.extend(check_partition_semantics(parts, p["split"], rows))
                check(set(columns) == set(rows[0])-excluded, "Predictor schema does not match the CSV columns.")
                for name in names:
                    truth = [rows[i][p["target"]] for i in parts["validation"]]
                    actual = independent_metrics(truth, e["validation_predictions"][name], p["task"])
                    check(_equal_metrics(e["validation"][name], actual), "Independent validation metric calculation disagrees.")
                truth = [rows[i][p["target"]] for i in parts["test"]]
                check(_equal_metrics(e["test"], independent_metrics(truth, e["test_predictions"], p["task"])), "Independent test metric calculation disagrees.")
    except (KeyError, TypeError, ValueError, IndexError, OSError, UnicodeError, OverflowError) as exc:
        errors.append(f"Malformed or unverifiable evidence: {type(exc).__name__}: {exc}")
    return {"ok": not errors, "errors": errors, "checks": checks, "dataset_bound": bound,
            "authenticates_execution": False,
            "scope": "Record conformance and, when data are supplied, file binding and score recomputation; not execution attestation."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path); parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    try: report = verify(json.loads(args.evidence.read_text()), args.data)
    except (OSError, ValueError) as exc: report = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1

if __name__ == "__main__": raise SystemExit(main())
