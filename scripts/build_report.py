"""Derive the reported summaries and figures from the recorded result files."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
COLUMN_WIDTH = 3.45
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.5, "lines.linewidth": 0.8, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "pdf.fonttype": 42, "savefig.dpi": 300})
SHADE = LinearSegmentedColormap.from_list("shade", ["#f7f7f7", "#2b4c6f"])
FAULT_LABELS = {"prepare_all_rows": "Preparation, all rows", "prepare_train_validation": "Preparation, train + validation",
                "fit_train_validation": "Estimators, train + validation", "test_rows_in_training": "Test rows in training",
                "select_by_test": "Selection by test score", "refit_before_assessment": "Refit before assessment",
                "undeclared_seed": "Undeclared seed", "partition_rule_ignored": "Partition rule ignored",
                "target_proxy_column": "Target proxy column"}
ORACLE_COLUMNS = [("honest_record_rejected_with_csv", "Honest\nrecord"), ("echo_record_rejected_with_csv", "Echo\nrecord,\nCSV"),
                  ("echo_record_rejected_without_csv", "Echo\nrecord,\nno CSV"), ("replay_detected", "Replay"),
                  ("perturbation_detected", "Pertur-\nbation")]
DATASET_LABELS = {"iris": "Iris", "wine": "Wine", "breast_cancer": "Breast c.", "digits": "Digits", "diabetes": "Diabetes",
                  "mixed": "Mixed", "grouped": "Grouped", "temporal": "Temporal"}


def summarize(out):
    comp = pd.read_csv(out / "compiler_cases.csv")
    replay = pd.read_csv(out / "replay_agreement.csv")
    witness = pd.read_csv(out / "witness_mutations.csv")
    meta = pd.read_csv(out / "metamorphic_cases.csv")
    neg = pd.read_csv(out / "negative_controls.csv")
    verify = pd.read_csv(out / "verification_times.csv")
    faults = pd.read_csv(out / "fault_matrix_summary.csv")
    runtime = json.loads((out / "runtime_summary.json").read_text())
    summary = {
        "compiler": {"valid": int(comp.valid.sum()), "invalid": int((~comp.valid).sum()), "correct": int(comp.correct.sum()), "total": len(comp)},
        "replay": {"workflows": len(replay), "candidate_fits_per_execution_path": int(replay.candidates.sum()), "passed": int(replay.ok.sum()),
                   "max_absolute_difference": float(replay.max_absolute_difference.max())},
        "witness_mutations": {"families": int(witness.family.nunique()), "total": len(witness),
                              "rejected_with_data": int(witness.rejected_with_data.sum()), "rejected_without_data": int(witness.rejected_without_data.sum())},
        "metamorphic": {"total": len(meta), "passed": int(meta.all_pass.sum())},
        "negative_controls": {"total": len(neg), **{k: int(neg[k].sum()) for k in
                              ["honest_leaky_trace_rejected", "fabricated_leaky_trace_accepted", "replay_output_difference_detected", "holdout_perturbation_detected"]}},
        "fault_matrix": [{k: (v if k == "fault" else (None if pd.isna(v) else int(v))) for k, v in row.items()} for row in faults.to_dict(orient="records")],
        "verification_median_ms": 1000 * float(verify.seconds.median()), "runtime": runtime,
        "no_human_study": True, "no_external_tool_head_to_head": True}
    return summary


def fault_figure(out, fig):
    table = pd.read_csv(out / "fault_matrix.csv")
    summary = pd.read_csv(out / "fault_matrix_summary.csv")
    faults = [f for f in FAULT_LABELS if f in set(summary.fault)]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.3), gridspec_kw={"height_ratios": [2.6, 1.3]})
    grid = []
    for fault in faults:
        row = summary.set_index("fault").loc[fault]
        grid.append([row[c] / row["cases"] for c, _ in ORACLE_COLUMNS])
    top.imshow(grid, cmap=SHADE, vmin=0, vmax=1, aspect="auto")
    for i, fault in enumerate(faults):
        row = summary.set_index("fault").loc[fault]
        for j, (c, _) in enumerate(ORACLE_COLUMNS):
            share = row[c] / row["cases"]
            top.text(j, i, f"{int(row[c])}/{int(row['cases'])}", ha="center", va="center", fontsize=6.5,
                     color="white" if share > 0.55 else "#1a1a1a")
    top.set_xticks(range(len(ORACLE_COLUMNS)), [label for _, label in ORACLE_COLUMNS])
    top.set_yticks(range(len(faults)), [FAULT_LABELS[f] for f in faults])
    top.tick_params(length=0)
    for spine in top.spines.values():
        spine.set_visible(False)
    top.set_title("(a) Faulty executions detected, by oracle", loc="left", fontsize=7)
    subset = table[table.fault == "prepare_all_rows"]
    datasets = [d for d in DATASET_LABELS if d in set(subset.dataset)]
    replay = [int(subset[subset.dataset == d].replay_detected.sum()) for d in datasets]
    perturb = [int(subset[subset.dataset == d].perturbation_detected.sum()) for d in datasets]
    counts = [int((subset.dataset == d).sum()) for d in datasets]
    x = range(len(datasets)); width = 0.38
    bottom.bar([i - width / 2 for i in x], replay, width, color="#2b4c6f", label="Replay")
    bottom.bar([i + width / 2 for i in x], perturb, width, color="#9fb3c8", label="Perturbation")
    bottom.set_xticks(list(x), [DATASET_LABELS[d] for d in datasets], rotation=30, ha="right")
    bottom.set_ylim(0, max(counts) * 1.55)
    bottom.set_yticks(range(0, max(counts) + 1))
    bottom.set_ylabel("Runs detected")
    bottom.legend(frameon=False, ncol=2, loc="upper center")
    bottom.set_title("(b) Preparation on all rows, by input", loc="left", fontsize=7)
    for spine in ("top", "right"):
        bottom.spines[spine].set_visible(False)
    figure.tight_layout(h_pad=1.2)
    figure.savefig(fig / "fault_matrix.pdf")
    figure.savefig(fig / "fault_matrix.png")
    plt.close(figure)


def runtime_figure(out, fig):
    pairs = pd.read_csv(out / "runtime_pairs.csv")
    sizes = sorted(pairs.rows.unique())
    values = [pairs.loc[pairs.rows == n, "difference_ms"].to_numpy() for n in sizes]
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
    ax.boxplot(values, tick_labels=[f"{n:,}" for n in sizes], widths=0.5, showmeans=False,
               boxprops={"linewidth": 0.7, "color": "#2b4c6f"}, whiskerprops={"linewidth": 0.7, "color": "#2b4c6f"},
               capprops={"linewidth": 0.7, "color": "#2b4c6f"}, medianprops={"linewidth": 1.0, "color": "#c0392b"},
               flierprops={"marker": "o", "markersize": 2.5, "markeredgewidth": 0.5, "markeredgecolor": "#2b4c6f"})
    ax.axhline(0, linestyle="--", linewidth=0.6, color="#555555")
    ax.set_xlabel("Input rows")
    ax.set_ylabel("Language minus plan API (ms)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(fig / "runtime_differences.pdf")
    figure.savefig(fig / "runtime_differences.png")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/full")
    args = parser.parse_args()
    out = args.results.resolve()
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    summary = summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    witness = pd.read_csv(out / "witness_mutations.csv")
    witness.groupby("family", sort=True).agg(cases=("case", "count"), with_csv=("rejected_with_data", "sum"),
                                               without_csv=("rejected_without_data", "sum")).reset_index().to_csv(out / "witness_family_summary.csv", index=False)
    scores = pd.read_csv(out / "test_scores_descriptive.csv")
    scores.groupby(["dataset", "metric"]).test_score.agg(["count", "mean", "std", "min", "max"]).reset_index().to_csv(out / "descriptive_scores_summary.csv", index=False)
    fault_figure(out, fig)
    runtime_figure(out, fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
