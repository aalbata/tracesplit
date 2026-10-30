# Reviewer guide

## What can be tested

TraceSplit is an executable research prototype for checked tabular machine-learning workflows. The package contains the complete implementation, eight examples, exact input data, a standalone evidence checker, standalone Python exports, tests, raw experiment outputs, the injected-fault and comparison scripts, and the scripts that rebuild the reported numbers, tables, and figures.

The central distinction is between a consistent record and the computation it describes. The package includes faulty runs whose records echo the declared protocol and pass the checker. A passing record check is evidence of consistency, not of correct execution.

## First installation

Use this folder as the working directory. The measured environment was CPython 3.12.10 on Windows 11 with the versions in `requirements-tested.txt`.

```sh
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python -m pip install --no-build-isolation --no-deps -e .
python scripts/smoke_test.py
```

The smoke script checks source compilation, workflow execution, standard-library verification, exported Python execution, and output comparison. A successful run prints JSON with `ok: true` and zero return codes.

## Inspect one run manually

```sh
python -m tracesplit run examples/iris.tsml --root . --out results/demo --export
python -S src/tracesplit/verifier.py results/demo/evidence.json --data datasets/iris.csv
python results/demo/python_export/experiment.py --output results/demo/replay.json
python -m tracesplit compare results/demo/evidence.json results/demo/replay.json
```

The first command saves the evidence record, a textual explanation, held-out predictions, and a standalone Python replay. The second command succeeds without importing site packages. The final comparison reports an exact match in the recorded environment; the declared tolerance is 1e-10.

## Open the application

```sh
python -m tracesplit serve --root . --port 8765
```

Open `http://127.0.0.1:8765`. Select an example, read or edit the program, and press Run workflow. The tabs expose numerical results, the recorded explanation, the evidence record, and the exported Python. Download links return the actual run files. Keep the terminal open and stop it with Ctrl+C.

## Check an invalid program

Copy `examples/iris.tsml` and replace `on train` in the preparation statement with `on all`. The checker reports `E_FIT_SCOPE`. Changing `evaluate linear on validation` to `evaluate linear on test` reports `E_EVAL_SCOPE`. Adding a training statement after assessment reports `E_TERMINAL`. The full constructed corpus is in `experiments/corpus`; its 48 valid and 192 invalid cases are also used by the automated tests.

## Inspect the central negative result

```sh
python -S src/tracesplit/verifier.py results/full/runs/iris_seed_0/negative_control_honest_leaky.json --data datasets/iris.csv
python -S src/tracesplit/verifier.py results/full/runs/iris_seed_0/negative_control_fabricated_leaky.json --data datasets/iris.csv
```

The honest record fails because preprocessing used all rows. The second record carries the same faulty numbers but declares the training-only scope, and it passes. The per-execution outcomes of all nine injected-fault mechanisms are in `results/full/fault_matrix.csv`; the per-mechanism counts are in `results/full/fault_matrix_summary.csv`.

## Rebuild all reported results

```sh
python -m pytest --junitxml=results/unit_tests.xml --cov=tracesplit --cov-report=json:results/coverage.json
python experiments/reproduce.py --stage all --output results/reproduced
python scripts/build_report.py --results results/reproduced
```

Preserve the original `results/full` directory and put new runs in another output directory. Expected structural outcomes are 240 correct source decisions, 40 replay matches, 600 altered records rejected with the CSV and 520 without it, 80 preserved correct-engine counterfactuals, and the fault-matrix and comparison counts reported in `results/full/summary.json`. Timing values are machine dependent.

## Interpretation and scope

The conditional invariant fixes partitions, schema, seeds, supported operations, and accepted input status. Loading and splitting may inspect the full CSV. The test-assessment restriction applies to one run. The language cannot detect target proxies or duplicate entities, and its hashes are unkeyed consistency checks. The replay shares numerical libraries with the engine.
