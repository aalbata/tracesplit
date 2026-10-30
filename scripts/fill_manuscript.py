"""Fill the manuscript and supplement templates with numbers from the result files.

Every number that the article reports is read from the saved results and
substituted for a placeholder of the form @@name@@. The script asserts that
each defined placeholder occurs in at least one template, that every
occurrence is replaced, and that no placeholder remains.
"""
from __future__ import annotations
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"@@([A-Za-z0-9_]+)@@")
FAULT_ORDER = ["prepare_all_rows", "prepare_train_validation", "fit_train_validation", "test_rows_in_training", "select_by_test",
               "refit_before_assessment", "undeclared_seed", "partition_rule_ignored", "target_proxy_column"]
FAULT_TEXT = {"prepare_all_rows": "Preparation fitted on all rows", "prepare_train_validation": "Preparation fitted on training and validation rows",
              "fit_train_validation": "Estimators fitted on training and validation rows", "test_rows_in_training": "Test rows added to training rows",
              "select_by_test": "Selection by test score", "refit_before_assessment": "Winner refitted on training and validation rows before assessment",
              "undeclared_seed": "Estimators fitted with an undeclared seed", "partition_rule_ignored": "Grouped or temporal rule ignored",
              "target_proxy_column": "Target proxy column in the data"}
FAMILY_TEXT = {"valid_workflow": "Valid workflow", "unsplit_fit": "Fit before any split", "preprocess_before_split": "Preprocessing before the split",
               "prepare_all": "Preparation fitted on all rows", "fit_validation": "Fit on validation rows", "fit_test": "Fit on test rows",
               "test_rows_in_training": "Test rows added to training rows", "evaluate_test": "Validation evaluation on test rows",
               "select_by_test": "Selection by test score", "assess_validation": "Final assessment on validation rows",
               "second_assessment": "Second assessment of the same test rows", "post_assessment_fit": "Development operation after assessment",
               "missing_assessment": "Workflow without final assessment"}
DATASET_TEXT = {"iris": "Iris", "wine": "Wine", "breast_cancer": "Breast cancer", "digits": "Digits", "diabetes": "Diabetes",
                "mixed": "Mixed fixture", "grouped": "Grouped fixture", "temporal": "Temporal fixture"}
TASK_TEXT = {"classification": "Class.", "regression": "Regr."}
SPLIT_TEXT = {"stratified": "Stratified", "random": "Random", "grouped": "Grouped", "temporal": "Temporal"}


def latex_escape(text):
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fmt_ms(value):
    return f"{value:.3f}"


def fmt_interval(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def collect(results, root):
    summary = json.loads((results / "summary.json").read_text())
    faults = pd.read_csv(results / "fault_matrix_summary.csv").set_index("fault")
    matrix = pd.read_csv(results / "fault_matrix.csv")
    comparison = pd.read_csv(results / "comparison" / "protocol_violations_comparison.csv").set_index("family")
    comparison_cases = pd.read_csv(results / "comparison" / "mlw_protocol_violations.csv")
    mlw_environment = json.loads((results / "comparison" / "mlw_protocol_violations.environment.json").read_text())
    witness = pd.read_csv(results / "witness_family_summary.csv")
    scores = pd.read_csv(results / "test_scores_descriptive.csv")
    manifest = json.loads((root / "datasets" / "manifest.json").read_text())
    environment = json.loads(next(results.glob("environment_*.json")).read_text())
    config = json.loads((results / "config_used.json").read_text())
    tests = ET.parse(root / "results" / "unit_tests.xml").getroot()
    suite = tests if tests.tag == "testsuite" else tests.find("testsuite")
    coverage = json.loads((root / "results" / "coverage.json").read_text())["totals"]
    runs = {}
    for path in sorted((results / "runs").glob("*/evidence.json")):
        runs[path.parent.name] = json.loads(path.read_text())["environment"]
    libraries = next(iter(runs.values()))
    v = {}
    v["compiler_valid"] = summary["compiler"]["valid"]
    v["compiler_invalid"] = summary["compiler"]["invalid"]
    v["compiler_total"] = summary["compiler"]["total"]
    v["compiler_correct"] = summary["compiler"]["correct"]
    v["invalid_families"] = len({f for f in pd.read_csv(results / "compiler_cases.csv").family if f != "valid_configuration"})
    v["replay_workflows"] = summary["replay"]["workflows"]
    v["replay_passed"] = summary["replay"]["passed"]
    v["candidate_fits"] = summary["replay"]["candidate_fits_per_execution_path"]
    v["replay_max_abs"] = "0" if summary["replay"]["max_absolute_difference"] == 0 else f"{summary['replay']['max_absolute_difference']:.1e}"
    v["witness_total"] = summary["witness_mutations"]["total"]
    v["witness_families"] = summary["witness_mutations"]["families"]
    v["witness_with"] = summary["witness_mutations"]["rejected_with_data"]
    v["witness_without"] = summary["witness_mutations"]["rejected_without_data"]
    v["witness_extra"] = v["witness_with"] - v["witness_without"]
    data_only = witness[(witness.with_csv == witness.cases) & (witness.without_csv == 0)]
    v["witness_data_only_families"] = len(data_only)
    v["meta_total"] = summary["metamorphic"]["total"]
    v["meta_passed"] = summary["metamorphic"]["passed"]
    v["datasets"] = len(config["datasets"])
    v["seeds"] = len(config["seeds"])
    v["seed_list"] = ", ".join(str(s) for s in config["seeds"])
    v["fault_mechanisms"] = len(faults)
    v["fault_cases"] = int(faults.cases.sum())
    same = faults.drop(index="target_proxy_column")
    v["same_data_cases"] = int(same.cases.sum())
    v["same_data_changed"] = int(same.outputs_differ_from_correct.sum())
    v["replay_total"] = int(faults.replay_detected.sum())
    v["perturbation_total"] = int(faults.perturbation_detected.sum())
    v["echo_csv_total"] = int(faults.echo_record_rejected_with_csv.sum())
    v["echo_nocsv_total"] = int(faults.echo_record_rejected_without_csv.sum())
    v["honest_total"] = int(faults.honest_record_rejected_with_csv.sum())
    v["honest_cases_with_scope_change"] = int(faults.loc[["prepare_all_rows", "prepare_train_validation", "fit_train_validation", "test_rows_in_training", "select_by_test", "refit_before_assessment"]].cases.sum())
    for fault in FAULT_ORDER:
        row = faults.loc[fault]
        v[f"{fault}_cases"] = int(row.cases)
        v[f"{fault}_changed"] = "n/a" if pd.isna(row.outputs_differ_from_correct) else int(row.outputs_differ_from_correct)
        v[f"{fault}_honest"] = int(row.honest_record_rejected_with_csv)
        v[f"{fault}_echo_csv"] = int(row.echo_record_rejected_with_csv)
        v[f"{fault}_echo_nocsv"] = int(row.echo_record_rejected_without_csv)
        v[f"{fault}_replay"] = int(row.replay_detected)
        v[f"{fault}_perturb"] = int(row.perturbation_detected)
    v["prepare_all_rows_missed"] = v["prepare_all_rows_cases"] - v["prepare_all_rows_replay"]
    misses = matrix[(matrix.fault == "prepare_all_rows") & (~matrix.replay_detected.astype(bool))]
    regression = {"diabetes", "temporal"}
    v["prepare_all_rows_missed_regression"] = int(misses.dataset.isin(regression).sum())
    v["prepare_all_rows_missed_classification"] = int((~misses.dataset.isin(regression)).sum())
    grouped_subset = matrix[(matrix.fault == "prepare_all_rows") & (matrix.dataset == "grouped")]
    v["prepare_all_rows_grouped_replay"] = int(grouped_subset.replay_detected.sum())
    v["prepare_all_rows_grouped_cases"] = len(grouped_subset)
    replay_only = faults.loc[["fit_train_validation", "prepare_train_validation", "refit_before_assessment", "undeclared_seed"]]
    v["replay_only_cases"] = int(replay_only.cases.sum())
    v["replay_only_detected"] = int(replay_only.replay_detected.sum())
    v["replay_only_perturbation"] = int(replay_only.perturbation_detected.sum())
    v["select_by_test_record_pct"] = f"{100 * v['select_by_test_echo_csv'] / v['select_by_test_cases']:.0f}"
    v["fault_matrix_rows"] = "\n".join(
        f"{FAULT_TEXT[f]} & {v[f + '_cases']} & {v[f + '_changed']} & {v[f + '_honest']} & {v[f + '_echo_csv']} & {v[f + '_echo_nocsv']} & {v[f + '_replay']} & {v[f + '_perturb']}\\\\"
        for f in FAULT_ORDER)
    per_dataset = matrix.groupby(["fault", "dataset"]).agg(cases=("case", "count"), replay=("replay_detected", "sum"), perturb=("perturbation_detected", "sum"),
                                                          record=("echo_record_rejected_with_csv", "sum")).reset_index()
    lines = []
    for fault in FAULT_ORDER:
        cells = []
        for d in DATASET_TEXT:
            part = per_dataset[(per_dataset.fault == fault) & (per_dataset.dataset == d)]
            if part.empty:
                cells.append("--")
            else:
                r = part.iloc[0]
                cells.append(f"{int(r.record)}/{int(r.replay)}/{int(r.perturb)}")
        lines.append(f"{FAULT_TEXT[fault]} & " + " & ".join(cells) + "\\\\")
    v["fault_matrix_by_input_rows"] = "\n".join(lines)
    order = ["valid_workflow", "unsplit_fit", "preprocess_before_split", "prepare_all", "fit_validation", "fit_test", "test_rows_in_training",
             "evaluate_test", "select_by_test", "assess_validation", "second_assessment", "post_assessment_fit", "missing_assessment"]
    v["comparison_families"] = len(order) - 1
    v["comparison_inputs"] = int(comparison.loc["valid_workflow"].inputs)
    v["comparison_valid_accepted_tracesplit"] = v["comparison_inputs"] - int(comparison.loc["valid_workflow"].tracesplit_rejected)
    v["comparison_valid_accepted_mlw"] = v["comparison_inputs"] - int(comparison.loc["valid_workflow"].mlw_rejected)
    violations = comparison.loc[order[1:]]
    v["comparison_expressible"] = int(violations.expressible_in_tracesplit.sum())
    v["comparison_inexpressible"] = int((~violations.expressible_in_tracesplit).sum())
    v["comparison_tracesplit_rejected_families"] = int(((violations.tracesplit_rejected == v["comparison_inputs"]) & violations.expressible_in_tracesplit).sum())
    v["comparison_mlw_rejected_families"] = int((violations.mlw_rejected == v["comparison_inputs"]).sum())
    v["comparison_mlw_accepted_families"] = int((violations.mlw_rejected == 0).sum())
    assert v["comparison_mlw_rejected_families"] + v["comparison_mlw_accepted_families"] == len(order) - 1, "mlw outcomes vary across inputs"
    v["comparison_mlw_accepted_list"] = ", ".join(FAMILY_TEXT[f].lower() for f in order[1:] if comparison.loc[f].mlw_rejected == 0)
    rows = []
    for f in order:
        r = comparison.loc[f]
        ts = "not expressible" if not r.expressible_in_tracesplit else f"{int(r.tracesplit_rejected)}/{int(r.inputs)}"
        rows.append(f"{FAMILY_TEXT[f]} & {ts} & {int(r.mlw_rejected)}/{int(r.inputs)}\\\\")
    v["comparison_rows"] = "\n".join(rows)
    rows = []
    for f in order:
        r = comparison.loc[f]
        codes = latex_escape(r.tracesplit_codes if isinstance(r.tracesplit_codes, str) else "")
        errors = latex_escape(r.mlw_errors if isinstance(r.mlw_errors, str) else "")
        rows.append(f"{FAMILY_TEXT[f]} & {codes or '--'} & {errors or '--'}\\\\")
    v["comparison_error_rows"] = "\n".join(rows)
    v["mlw_version"] = mlw_environment["mlw"]
    v["mlw_pandas"] = mlw_environment["pandas"]
    v["mlw_audit_entries"] = mlw_environment["audit_chain_entries_after_run"]
    v["runtime_rows"] = "\n".join(f"{r['rows']:,} & {fmt_ms(r['median_language_ms'])} & {fmt_ms(r['median_plan_api_ms'])} & {fmt_ms(r['median_paired_difference_ms'])} & {fmt_interval(r['paired_difference_95pct_bootstrap_ci_ms'])}\\\\"
                                  for r in summary["runtime"])
    v["runtime_pairs"] = summary["runtime"][0]["pairs"]
    v["runtime_warmups"] = config["runtime_warmups"]
    v["runtime_sizes"] = ", ".join(f"{r['rows']:,}" for r in summary["runtime"])
    v["runtime_medians"] = ", ".join(fmt_ms(r["median_paired_difference_ms"]) for r in summary["runtime"])
    v["runtime_intervals_with_zero"] = sum(1 for r in summary["runtime"] if r["paired_difference_95pct_bootstrap_ci_ms"][0] <= 0 <= r["paired_difference_95pct_bootstrap_ci_ms"][1])
    v["runtime_largest_abs_median"] = fmt_ms(max(abs(r["median_paired_difference_ms"]) for r in summary["runtime"]))
    v["bootstrap_resamples"] = f"{config['bootstrap_resamples']:,}"
    v["verification_median_ms"] = f"{summary['verification_median_ms']:.1f}"
    v["tests_total"] = int(suite.attrib["tests"]) - int(suite.attrib.get("skipped", 0))
    v["tests_failed"] = int(suite.attrib["failures"]) + int(suite.attrib["errors"])
    v["coverage_pct"] = f"{coverage['percent_covered']:.0f}"
    v["coverage_statements"] = coverage["num_statements"]
    v["python_version"] = environment["python"].split()[0]
    v["processor"] = environment["cpu"].replace("(R)", "").replace("(TM)", "")
    v["logical_cpus"] = environment["logical_cpus_visible"]
    v["numpy_version"] = libraries["numpy"]
    v["pandas_version"] = libraries["pandas"]
    v["scipy_version"] = libraries["scipy"]
    v["sklearn_version"] = libraries["scikit_learn"]
    v["lark_version"] = __import__("importlib.metadata").metadata.version("lark")
    v["dataset_rows"] = "\n".join(f"{DATASET_TEXT[d['name']]} & {d['rows']} & {d['predictors']} & {TASK_TEXT[d['task']]} & {SPLIT_TEXT[d['split']]}\\\\" for d in manifest)
    v["total_rows"] = sum(d["rows"] for d in manifest)
    v["score_rows"] = "\n".join(f"{latex_escape(r.dataset)} & {r.seed} & {r.metric} & {r.selected} & {r.test_score:.8f}\\\\" for r in scores.itertuples())
    v["witness_rows"] = "\n".join(f"{latex_escape(r.family)} & {r.cases} & {r.with_csv} & {r.without_csv}\\\\" for r in witness.itertuples())
    for key in ["datasets", "seeds", "fault_mechanisms", "witness_data_only_families", "comparison_mlw_rejected_families",
                "comparison_mlw_accepted_families", "comparison_inexpressible", "comparison_expressible", "comparison_families"]:
        v[key + "_word"] = word(v[key])
    v["runtime_intervals_phrase"] = "all three" if v["runtime_intervals_with_zero"] == 3 else f"{word(v['runtime_intervals_with_zero'])} of the three"
    guard_claims(v, faults, matrix, comparison, order)
    return v


def word(n):
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    return names[n] if 0 <= n < len(names) else str(n)


def guard_claims(v, faults, matrix, comparison, order):
    """Assert the qualitative statements that the article makes about the result files."""
    scope_faults = ["prepare_all_rows", "prepare_train_validation", "fit_train_validation", "test_rows_in_training", "select_by_test", "refit_before_assessment"]
    assert all(faults.loc[f].honest_record_rejected_with_csv == faults.loc[f].cases for f in scope_faults), "honest scope records must all be rejected"
    assert faults.loc["undeclared_seed"].honest_record_rejected_with_csv == 0, "honest undeclared-seed records must be accepted"
    assert faults.loc["partition_rule_ignored"].honest_record_rejected_with_csv == 0, "honest partition-rule records must be accepted"
    backend = ["prepare_all_rows", "prepare_train_validation", "fit_train_validation", "test_rows_in_training", "refit_before_assessment", "undeclared_seed"]
    assert all(faults.loc[f].echo_record_rejected_with_csv == 0 and faults.loc[f].echo_record_rejected_without_csv == 0 for f in backend), "echo records of backend faults must pass"
    assert faults.loc["select_by_test"].echo_record_rejected_with_csv == faults.loc["select_by_test"].echo_record_rejected_without_csv == faults.loc["select_by_test"].outputs_differ_from_correct, "selection echo rejections must equal the runs with a changed winner"
    assert faults.loc["partition_rule_ignored"].echo_record_rejected_with_csv == faults.loc["partition_rule_ignored"].cases, "partition echo records must be rejected with the CSV"
    assert faults.loc["partition_rule_ignored"].echo_record_rejected_without_csv == 0 and faults.loc["partition_rule_ignored"].replay_detected == 0, "partition fault must pass without CSV and under replay"
    assert faults.loc["target_proxy_column"][["honest_record_rejected_with_csv", "echo_record_rejected_with_csv", "echo_record_rejected_without_csv", "replay_detected", "perturbation_detected"]].sum() == 0, "data-level fault must pass every oracle"
    same = faults.drop(index=["target_proxy_column", "partition_rule_ignored"])
    assert (same.replay_detected == same.outputs_differ_from_correct).all(), "replay detection must coincide with changed outputs on unchanged data"
    assert faults.loc["prepare_all_rows"].perturbation_detected == faults.loc["prepare_all_rows"].cases, "perturbation must detect preparation on all rows"
    assert faults.loc["test_rows_in_training"].perturbation_detected == faults.loc["test_rows_in_training"].cases, "perturbation must detect test rows in training"
    confined = ["prepare_train_validation", "fit_train_validation", "refit_before_assessment", "undeclared_seed", "partition_rule_ignored", "target_proxy_column"]
    assert all(faults.loc[f].perturbation_detected == 0 for f in confined), "perturbation must miss faults confined to development rows"
    assert v["prepare_all_rows_missed_regression"] == 0, "replay misses on the preparation fault must all be classification runs"
    regression = matrix[(matrix.fault == "prepare_all_rows") & (matrix.dataset.isin(["diabetes", "temporal"]))]
    assert regression.replay_detected.all(), "replay must detect the preparation fault on every regression run"
    assert v["prepare_all_rows_grouped_replay"] == 0, "replay must miss the preparation fault on the grouped fixture"
    rejected = {f for f in order[1:] if comparison.loc[f].mlw_rejected == comparison.loc[f].inputs}
    assert rejected == {"unsplit_fit", "fit_test", "test_rows_in_training", "evaluate_test", "select_by_test", "assess_validation", "second_assessment"}, rejected
    assert all(comparison.loc[f].tracesplit_rejected == comparison.loc[f].inputs for f in order[1:]), "the checker must reject every expressible family"
    assert comparison.loc["valid_workflow"].tracesplit_rejected == 0 and comparison.loc["valid_workflow"].mlw_rejected == 0, "both tools must accept the valid workflow"
    assert v["tests_failed"] == 0, "the test suite must pass"


def fill(template, values, used):
    def repl(match):
        key = match.group(1)
        if key not in values:
            raise KeyError(f"No value for placeholder {key}")
        used.add(key)
        return str(values[key])
    text = PLACEHOLDER.sub(repl, template)
    assert "@@" not in text, "a placeholder remained unfilled"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = parser.parse_args(argv)
    values = collect(args.results.resolve(), ROOT)
    used = set()
    for name in ["Manuscript", "Supplementary_Methods"]:
        template = (args.paper / f"{name}.in.tex").read_text(encoding="utf-8")
        (args.paper / f"{name}.tex").write_text(fill(template, values, used), encoding="utf-8")
    unused = sorted(set(values) - used)
    (args.paper / "numbers.json").write_text(json.dumps({k: v for k, v in values.items() if not k.endswith("_rows")}, indent=2) + "\n", encoding="utf-8")
    print(f"filled {len(used)} distinct placeholders; computed but unused: {unused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
