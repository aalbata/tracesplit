# Language reference

TraceSplit 0.1.0 accepts one complete supervised-learning workflow per program. Keywords and model names are case-sensitive. Each statement occupies one line. Blank lines and comments beginning with `#` are allowed. Strings use double quotes with JSON-style escapes. There are no user-defined functions, loops, Python imports, or external execution statements.

## Program order

A complete program reads one dataset, declares one target and task, optionally drops columns, creates a single split, prepares features, trains and evaluates one or more candidates, selects a candidate, and assesses it on the test partition. `explain` may follow assessment once. It is an explicit request for the completed explanation; an explanation file is retained with every run's audit outputs regardless of whether this marker appears.

```text
read "datasets/iris.csv"
predict "target" as classification
split stratified train 60% validation 20% test 20% seed 42
prepare numeric median scale standard categorical most_frequent on train
train linear logistic C=1 on train
evaluate linear on validation
train forest forest trees=40 depth=6 on train
evaluate forest on validation
select best by accuracy
assess best on test
explain
```

The workflow checker rejects any later command except the single optional explanation. This rule is local to one execution. Running a modified program again can reuse the same test data.

## Data and splits

Paths resolve inside the supplied workspace, including after symlink resolution. Data must be a UTF-8 CSV with unique column names, a nonmissing target, finite numeric entries, 15 to 100,000 rows, and at most 200 columns. The file limit is 50 MB. At least one predictor must remain. Every partition must contain at least two rows; classification training must contain at least two classes. Invalid combinations fail rather than silently change strategy.

`classification` and `regression` are supported. The target is always excluded from predictors. Optional `drop "column" "other"` statements exclude additional columns before splitting. A grouped or temporal metadata column is excluded automatically and cannot also be the target or an explicit dropped column.

Split fractions must be positive and sum to exactly 100 in decimal arithmetic. Seeds are integers from 0 to 2^32-1. `random` uses seeded shuffled partitions; `stratified` preserves class composition through two scikit-learn stratified splits and is classification-only. Small classes can make a requested stratified split infeasible. `grouped by "subject"` splits distinct group identifiers and allocates whole groups, so percentages describe group counts rather than exact row counts. `temporal by "time"` sorts numeric or parseable date-time keys and requires strict temporal boundaries; a tie crossing a boundary is rejected.

Row identifiers are zero-based positions after CSV parsing, not semantic person identifiers. Duplicate records or near-duplicates are not automatically recognized. The input parser and split construction can inspect the full dataset; this is not an implementation in which all test bytes remain unread until the final command.

## Preprocessing and models

Numeric imputation supports `median`, `mean`, and `most_frequent`. Scaling is `standard` or `none`. Categorical imputation is `most_frequent` or `constant`, followed by one-hot encoding. The constant token is `__MISSING__`. Unseen categories are ignored by the encoder and map to an all-zero category vector. Preprocessing parameters are learned only from training rows. Empty training columns are retained. Dense expansion is bounded by 2,000 output columns and ten million cells.

Candidate names must be distinct and cannot be `best`. There are at most 12 candidates. Supported classification estimators are `logistic`, `forest`, and `tree`. Supported regression estimators are `ridge`, `forest`, and `tree`. Logistic regression accepts `C`; ridge accepts `alpha`; forests accept `trees` and `depth`; trees accept `depth`. Positive regularization values are at most one million. Trees are integers from 1 to 200 and explicit depth is an integer from 1 to 30. Unspecified forest size is 40; unspecified depth is unlimited. Logistic regression uses LBFGS and at most 2,000 iterations. Ridge uses Cholesky. Forests and numerical library threads are limited to one within execution.

Every fitted candidate is evaluated exactly once on validation. Selection is permitted only after all declared candidates have validation scores. Classification metrics are `accuracy`, `balanced_accuracy`, and `macro_f1`. Regression metrics are `rmse`, `mae`, and `r2`. RMSE and MAE are minimized; other metrics are maximized. Ties choose the earliest candidate in declaration order. Only the selected candidate is assessed on test. It is not refitted on training plus validation.

## Evidence and export

`evidence.json` stores source text and its hash, normalized plan, input fingerprint, row partitions, schema, source-linked events, preprocessing state, model fingerprints, predictions, metrics, and environment information. `explanation.txt` summarizes those recorded operations; it is not a causal explanation of a model prediction. `predictions.csv` stores held-out row identifiers and predictions.

Each event links to the preceding event through an unkeyed hash. The standalone checker checks consistency, not a digital signature or trusted execution attestation. With the original CSV it also recomputes metric values from recorded predictions. A consistently fabricated record can pass; the research package includes such a case.

The Python export freezes the observed partitions and schema and rebuilds the numerical workflow using public scientific Python libraries. It does not import TraceSplit. Replay agreement is a fixed-input test, not a proof that every possible export is correct. The export intentionally exposes detailed implementation rather than promising shorter Python code.

## Complete grammar

The grammar below is the exact Lark source distributed with this release.

```lark
start: (_NL | statement _NL)*
?statement: read | predict | drop | split | prepare | train | evaluate | select | assess | explain
read: "read" STRING
predict: "predict" STRING "as" NAME
drop: "drop" STRING+
split: "split" NAME ["by" STRING] "train" PERCENT "validation" PERCENT "test" PERCENT "seed" INT
prepare: "prepare" "numeric" NAME "scale" NAME "categorical" NAME "on" NAME
train: "train" NAME NAME parameter* "on" NAME
parameter: NAME "=" NUMBER
evaluate: "evaluate" NAME "on" NAME
select: "select" "best" "by" NAME
assess: "assess" "best" "on" NAME
explain: "explain"
PERCENT: /[0-9]+(\.[0-9]+)?%/
NUMBER: /[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/
%import common.CNAME -> NAME
%import common.INT
%import common.ESCAPED_STRING -> STRING
%import common.NEWLINE -> _NL
%import common.WS_INLINE
%ignore WS_INLINE
%ignore /#[^\n]*/
```
