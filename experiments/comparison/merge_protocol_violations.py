"""Combine the TraceSplit and mlw protocol-violation outcomes into one table.

Both input files hold one row per data set and family. The output counts,
for each family, on how many of the eight inputs each tool rejected the
sequence, and records whether the family is expressible in the language.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ORDER = ["valid_workflow", "unsplit_fit", "preprocess_before_split", "prepare_all", "fit_validation", "fit_test",
         "test_rows_in_training", "evaluate_test", "select_by_test", "assess_validation", "second_assessment",
         "post_assessment_fit", "missing_assessment"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracesplit", type=Path, required=True)
    parser.add_argument("--mlw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    a = pd.read_csv(args.tracesplit)
    b = pd.read_csv(args.mlw)
    merged = a.merge(b, on=["dataset", "family"], how="outer", validate="one_to_one")
    assert merged[["tracesplit_accepted", "mlw_accepted"]].isna().sum().sum() == 0, "every family must have outcomes from both tools"
    rows = []
    for family in ORDER:
        part = merged[merged.family == family]
        rows.append({"family": family, "inputs": len(part), "expressible_in_tracesplit": bool(part.expressible.all()),
                     "tracesplit_rejected": int((~part.tracesplit_accepted).sum()), "mlw_rejected": int((~part.mlw_accepted).sum()),
                     "tracesplit_codes": "/".join(sorted(set(part.tracesplit_error.fillna("").astype(str)) - {""})),
                     "mlw_errors": "/".join(sorted(set(part.mlw_error.fillna("").astype(str)) - {""}))})
    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
