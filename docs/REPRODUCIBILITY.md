# Reproducibility and evidence map

## Fixed configuration

`experiments/config.json` specifies eight datasets, five seeds (0, 1, 7, 42, 2026), three candidates per workflow, prediction tolerances of 1e-10, three runtime sizes, 25 measured pairs after three warm-up pairs, and a seeded 10,000-resample percentile bootstrap. The input programs and dataset hashes are supplied. Five datasets are bundled scikit-learn examples; three are generated fixtures.

`requirements-tested.txt` and `results/full/environment_all.json` identify the numerical environment. The frozen standalone exports record their own partitions and schema. Use the bundled CSV files for exact fixed-input checks.

## Claims to files

| Claim | Evidence |
|---|---|
| 48 accepted and 192 rejected constructed programs | `results/full/compiler_cases.csv`, `experiments/corpus/` |
| 40 exact replay matches | `results/full/replay_agreement.csv`, each run's `python_export/` and `replay_result.json` |
| 600 altered records rejected with CSV; 520 without CSV | `results/full/witness_mutations.csv`, `results/full/witness_family_summary.csv` |
| 80 fixed-partition perturbations preserve development state | `results/full/metamorphic_cases.csv` |
| Injected-fault matrix, nine mechanisms, five observations each | `results/full/fault_matrix.csv`, `results/full/fault_matrix_summary.csv` |
| Matched protocol-violation comparison with mlw | `results/full/comparison/protocol_violations_comparison.csv` and the two per-input files beside it |
| Paired runtime observations | `results/full/runtime_pairs.csv`, per-size CSVs, `runtime_summary.json` |
| Test and coverage results | `results/unit_tests.xml`, `results/coverage.json`, `results/unit_tests.log` |
| Every number in the article and supplement | `paper/numbers.json`, written by `scripts/fill_manuscript.py` |

## Replay versus independent validation

The replay script imports the same numerical backend libraries but not TraceSplit, its parser, or its engine. It freezes the observed partitions and column schema and constructs the preprocessing and estimators separately. Equality on the tested inputs is fixed-input differential evidence, not proof of translation correctness on all inputs. A replay exported from a record reproduces the record's partitions; it does not check them against the declared strategy.

The checker in `src/tracesplit/verifier.py` uses no machine-learning library. With `--data`, it verifies the dataset hash, the grouped or temporal constraints, and the predictor schema, and it recomputes scalar metrics from saved predictions and labels. It cannot verify that recorded predictions were produced by training. Hashes are unkeyed.

The counterfactual cases perturb held-out features or labels while freezing row assignments. The tested state consists of preprocessing statistics, model fingerprints, validation scores, and the selected candidate. It excludes final test predictions, which may change.

## Injected faults

`experiments/fault_matrix.py` applies nine mechanisms to every saved correct run. For each faulty execution it writes whether the compared outputs differ from the correct run, whether the honest record and the protocol-echoing record are rejected (with and without the CSV), whether a replay exported from the echo record disagrees with it, and whether the held-out feature perturbation changes the faulty development state. The partition-rule mechanism applies only to the grouped and temporal workflows. The target-proxy mechanism changes the data rather than the backend, so its output comparison with the correct run is undefined.

## Re-running and comparison

Run `python scripts/smoke_test.py` first, then the commands in the README. Compare outcome counts, split identities, predictions, selected candidates, and metrics. Do not require identical run times or environment strings. `source_snapshot_all.json` records source hashes at execution time.
