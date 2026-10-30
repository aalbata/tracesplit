"""Write the results tables of the README from the saved result files.

The block between the results markers in README.md is replaced; the rest of
the file is left unchanged. The script asserts that exactly one such block
exists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"
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


def tables(results):
    summary = json.loads((results / "summary.json").read_text())
    faults = pd.read_csv(results / "fault_matrix_summary.csv")
    comparison = pd.read_csv(results / "comparison" / "protocol_violations_comparison.csv").set_index("family")
    lines = ["| Check | Outcome |", "|---|---|",
             f"| Valid source programs accepted | {summary['compiler']['valid']}/{summary['compiler']['valid']} |",
             f"| Invalid source programs rejected | {summary['compiler']['invalid']}/{summary['compiler']['invalid']} |",
             f"| Standalone numerical replays agreeing | {summary['replay']['passed']}/{summary['replay']['workflows']} |",
             f"| Altered records rejected, CSV supplied | {summary['witness_mutations']['rejected_with_data']}/{summary['witness_mutations']['total']} |",
             f"| Altered records rejected, no CSV | {summary['witness_mutations']['rejected_without_data']}/{summary['witness_mutations']['total']} |",
             f"| Correct-engine counterfactual invariants preserved | {summary['metamorphic']['passed']}/{summary['metamorphic']['total']} |", ""]
    lines += ["| Injected fault | Cases | Changed | Honest | Echo | Echo without CSV | Replay | Perturbation |", "|---|---|---|---|---|---|---|---|"]
    for row in faults.itertuples():
        changed = "n/a" if pd.isna(row.outputs_differ_from_correct) else int(row.outputs_differ_from_correct)
        lines.append(f"| {FAULT_TEXT[row.fault]} | {row.cases} | {changed} | {int(row.honest_record_rejected_with_csv)} | {int(row.echo_record_rejected_with_csv)} | "
                     f"{int(row.echo_record_rejected_without_csv)} | {int(row.replay_detected)} | {int(row.perturbation_detected)} |")
    lines += ["", "| Protocol-violation family | TraceSplit | mlw |", "|---|---|---|"]
    for family, text in FAMILY_TEXT.items():
        r = comparison.loc[family]
        ts = "not expressible" if not r.expressible_in_tracesplit else f"{int(r.tracesplit_rejected)}/{int(r.inputs)}"
        lines.append(f"| {text} | {ts} | {int(r.mlw_rejected)}/{int(r.inputs)} |")
    lines += ["", "| Rows | Language (ms) | Plan API (ms) | Paired difference (ms) | 95% interval (ms) |", "|---|---|---|---|---|"]
    for r in summary["runtime"]:
        ci = r["paired_difference_95pct_bootstrap_ci_ms"]
        lines.append(f"| {r['rows']:,} | {r['median_language_ms']:.3f} | {r['median_plan_api_ms']:.3f} | {r['median_paired_difference_ms']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args(argv)
    text = args.readme.read_text(encoding="utf-8")
    assert text.count(START) == 1 and text.count(END) == 1, "the README must contain exactly one results block"
    head, rest = text.split(START)
    _, tail = rest.split(END)
    filled = head + START + "\n" + tables(args.results.resolve()) + "\n" + END + tail
    assert "@@" not in filled, "a placeholder remained unfilled"
    args.readme.write_text(filled, encoding="utf-8")
    print("README results block written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
