# TraceSplit

TraceSplit is a small, checked language for tabular machine-learning experiments. It separates training, validation, and test roles at the source; records source-linked execution evidence; exports a standalone Python replay; and tests a run against held-out perturbations under frozen partitions. The repository holds the implementation, the input data, the experiment scripts, the saved results, and the article sources.

## Install

The recorded experiments used CPython 3.12.10 on Windows 11. The commands below assume the current directory is this folder. An initial installation needs internet access to obtain dependencies.

```sh
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python -m pip install --no-build-isolation --no-deps -e .
```

On Linux or macOS, activate the environment with `. .venv/bin/activate`. Dependency ranges in `pyproject.toml` describe the intended compatibility range; `requirements-tested.txt` identifies the measured environment.

## Run a complete example

```sh
python -m tracesplit check examples/iris.tsml
python -m tracesplit run examples/iris.tsml --root . --out results/demo --export
python -m tracesplit verify results/demo/evidence.json --data datasets/iris.csv
python results/demo/python_export/experiment.py --output results/demo/replay.json
python -m tracesplit compare results/demo/evidence.json results/demo/replay.json
```

An automated version is available:

```sh
python scripts/smoke_test.py
```

## Open the local application

```sh
python -m tracesplit serve --root . --port 8765
```

Open `http://127.0.0.1:8765` in a browser. Keep the terminal open and stop the server with Ctrl+C. The application offers eight editable examples, results, source-linked explanations, the evidence record, and standalone Python downloads. It is a local, single-user tool; do not expose its port publicly.

## Inspect the evidence without machine-learning libraries

```sh
python -S src/tracesplit/verifier.py results/demo/evidence.json --data datasets/iris.csv
```

This checker uses only the Python standard library. It checks the declared protocol, partition use, fingerprints, recorded metrics, and event consistency. A record that is internally consistent passes even when the computation it describes did not take place; the injected-fault study in `results/full/fault_matrix.csv` measures this boundary.

## Reproduce the results

Fast path from the shipped results (no experiments are rerun):

```sh
python scripts/build_report.py --results results/full
python scripts/build_paper.py --results results/full
```

The first command regenerates `summary.json`, the derived tables, and the figures from the saved CSV files. The second fills every number in the article and supplement from the same files and typesets both PDFs (requires pdflatex and bibtex).

Full path (all experiments are rerun; about fifteen minutes on the recorded machine):

```sh
python -m pytest --junitxml=results/unit_tests.xml --cov=tracesplit --cov-report=json:results/coverage.json
python experiments/reproduce.py --stage all --output results/reproduced
python scripts/build_report.py --results results/reproduced
```

The stages can also be run separately with `--stage cases`, `--stage faults`, and `--stage runtime --rows 200|2000|10000`; the last runtime block combines the saved blocks. Timings vary with the computer and load; numerical agreement and structural outcomes are the reproducibility targets. A reduced installation check is `python experiments/reproduce.py --quick --output results/quick`.

The matched comparison with the mlw package runs in a separate environment because mlw pins pandas below version 3:

```sh
python -m venv .mlw
.mlw\Scripts\python -m pip install -r experiments\comparison\requirements-mlw.txt
.mlw\Scripts\python experiments\comparison\protocol_violations_mlw.py --datasets datasets --output results\reproduced\comparison\mlw_protocol_violations.csv
python experiments\comparison\protocol_violations_tracesplit.py --output results\reproduced\comparison\tracesplit_protocol_violations.csv
python experiments\comparison\merge_protocol_violations.py --tracesplit results\reproduced\comparison\tracesplit_protocol_violations.csv --mlw results\reproduced\comparison\mlw_protocol_violations.csv --output results\reproduced\comparison\protocol_violations_comparison.csv
```

## Results

The table below is generated from `results/full/summary.json`, `results/full/fault_matrix_summary.csv`, and `results/full/comparison/protocol_violations_comparison.csv` by `scripts/fill_readme.py`.

<!-- results:start -->
| Check | Outcome |
|---|---|
| Valid source programs accepted | 48/48 |
| Invalid source programs rejected | 192/192 |
| Standalone numerical replays agreeing | 40/40 |
| Altered records rejected, CSV supplied | 600/600 |
| Altered records rejected, no CSV | 520/600 |
| Correct-engine counterfactual invariants preserved | 80/80 |

| Injected fault | Cases | Changed | Honest | Echo | Echo without CSV | Replay | Perturbation |
|---|---|---|---|---|---|---|---|
| Preparation fitted on all rows | 40 | 26 | 40 | 0 | 0 | 26 | 40 |
| Preparation fitted on training and validation rows | 40 | 23 | 40 | 0 | 0 | 23 | 0 |
| Estimators fitted on training and validation rows | 40 | 39 | 40 | 0 | 0 | 39 | 0 |
| Test rows added to training rows | 40 | 40 | 40 | 0 | 0 | 40 | 40 |
| Selection by test score | 40 | 10 | 40 | 10 | 10 | 10 | 19 |
| Winner refitted on training and validation rows before assessment | 40 | 27 | 40 | 0 | 0 | 27 | 0 |
| Estimators fitted with an undeclared seed | 40 | 33 | 0 | 0 | 0 | 33 | 0 |
| Grouped or temporal rule ignored | 10 | 10 | 0 | 10 | 0 | 0 | 0 |
| Target proxy column in the data | 40 | n/a | 0 | 0 | 0 | 0 | 0 |

| Protocol-violation family | TraceSplit | mlw |
|---|---|---|
| Valid workflow | 0/8 | 0/8 |
| Fit before any split | 8/8 | 8/8 |
| Preprocessing before the split | 8/8 | 0/8 |
| Preparation fitted on all rows | 8/8 | 0/8 |
| Fit on validation rows | 8/8 | 0/8 |
| Fit on test rows | 8/8 | 8/8 |
| Test rows added to training rows | not expressible | 8/8 |
| Validation evaluation on test rows | 8/8 | 8/8 |
| Selection by test score | not expressible | 8/8 |
| Final assessment on validation rows | 8/8 | 8/8 |
| Second assessment of the same test rows | 8/8 | 8/8 |
| Development operation after assessment | 8/8 | 0/8 |
| Workflow without final assessment | 8/8 | 0/8 |

| Rows | Language (ms) | Plan API (ms) | Paired difference (ms) | 95% interval (ms) |
|---|---|---|---|---|
| 200 | 36.312 | 35.180 | 0.339 | [-0.884, 2.742] |
| 2,000 | 64.732 | 64.213 | 1.075 | [-0.170, 2.928] |
| 10,000 | 163.944 | 167.311 | 0.076 | [-3.207, 2.030] |
<!-- results:end -->

Column definitions: Cases is the number of executions of a mechanism; Changed is the number whose compared outputs differ from the correct run; Honest is the number of honest records rejected with the CSV; Echo and Echo without CSV are the numbers of protocol-echoing records rejected with and without the CSV; Replay is the number of replay disagreements; Perturbation is the number of development-state changes under the held-out feature perturbation. In the comparison table, each cell is the number of inputs, out of eight, on which the tool rejected the family.

## What this release contains

`src/tracesplit` contains the grammar, checker, engine, evidence validator, exporter, command-line interface, and local application. `examples` contains the eight complete programs. `datasets` contains the exact CSV inputs and provenance descriptions. `experiments` contains the configuration, the evaluation code, the injected-fault matrix, and the comparison cases. `results/full` contains saved records, replay outputs, row partitions, counterfactual outcomes, fault-matrix outcomes, comparison outcomes, and raw timing pairs. `paper` contains the article and supplement templates, the bibliography, and the build output. `docs` explains the language, the reviewer workflow, and the evidence map.

## Scope

Only supervised classification and regression on bounded, local CSV inputs are supported. There is no arbitrary Python, neural-network training, cross-validation, hyperparameter search, or general loop or function system. Input parsing and split construction can examine the complete input; the supported invariant concerns fitted development state under fixed partitions and schema. The terminal test rule applies to one run. A safe split cannot identify target proxies or repair a poorly collected dataset.

## License and attribution

The TraceSplit code and documentation are released under the MIT license in `LICENSE`. Third-party software and datasets retain their own terms; see `THIRD_PARTY_NOTICES.md`.
