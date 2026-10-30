"""Build the exact small benchmark suite from local sklearn data and fixed generators."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import datasets

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT/"datasets"; out.mkdir(exist_ok=True)
    specs = []
    def save(name, frame, task, split, origin, description=None):
        frame.to_csv(out/f"{name}.csv", index=False)
        item = {"name": name, "rows": len(frame), "predictors": len(frame.columns)-1-(1 if split in {"grouped", "temporal"} else 0),
                "task": task, "split": split, "origin": origin, "sha256": hashlib.sha256((out/f"{name}.csv").read_bytes()).hexdigest()}
        if description: (out/f"{name}_source.txt").write_text(description)
        specs.append(item)
    for name, loader, task in [("iris", datasets.load_iris, "classification"), ("wine", datasets.load_wine, "classification"),
                                ("breast_cancer", datasets.load_breast_cancer, "classification"), ("digits", datasets.load_digits, "classification"),
                                ("diabetes", lambda: datasets.load_diabetes(scaled=False), "regression")]:
        bundle = loader(); frame = pd.DataFrame(bundle.data, columns=[f"x{i}" for i in range(bundle.data.shape[1])]); frame["target"] = bundle.target
        save(name, frame, task, "stratified" if task == "classification" else "random", "scikit-learn bundled real dataset", bundle.DESCR)
    rng = np.random.default_rng(271828)
    X, y = datasets.make_classification(n_samples=800, n_features=12, n_informative=7, n_redundant=2, random_state=314159)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    frame["category"] = np.array(["north", "south", "east"])[rng.integers(0, 3, len(frame))]
    frame["band"] = np.where(frame["x0"] > 0, "high", "low")
    for col in ["x0", "x3", "category"]: frame.loc[rng.random(len(frame)) < .08, col] = np.nan
    frame["target"] = y
    save("mixed", frame, "classification", "stratified", "generated stress fixture with missing values and categories")
    group = np.repeat(np.arange(60), 10)
    X = rng.normal(size=(600, 8)); effects = rng.normal(size=60)
    y = (X[:, 0]+.4*X[:, 1]+effects[group]+rng.normal(scale=.5, size=600) > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]); frame["subject"] = group; frame["target"] = y
    save("grouped", frame, "classification", "grouped", "generated repeated-subject stress fixture")
    times = np.arange(600); X = rng.normal(size=(600, 8)); y = 3*X[:, 0]-2*X[:, 1]+.003*times+rng.normal(scale=.3, size=600)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]); frame["time"] = times; frame["target"] = y
    save("temporal", frame, "regression", "temporal", "generated chronological stress fixture")
    (out/"manifest.json").write_text(json.dumps(specs, indent=2)+"\n")
    for item in specs:
        name, task, strat = item["name"], item["task"], item["split"]
        by = ' by "subject"' if strat == "grouped" else ' by "time"' if strat == "temporal" else ''
        first = "logistic C=1" if task == "classification" else "ridge alpha=1"
        metric = "accuracy" if task == "classification" else "rmse"
        source = f'''read "datasets/{name}.csv"
predict "target" as {task}
split {strat}{by} train 60% validation 20% test 20% seed 42
prepare numeric median scale standard categorical most_frequent on train
train linear {first} on train
evaluate linear on validation
train forest forest trees=40 depth=6 on train
evaluate forest on validation
train tree tree depth=5 on train
evaluate tree on validation
select best by {metric}
assess best on test
explain
'''
        (ROOT/"examples"/f"{name}.tsml").write_text(source)
    print(json.dumps(specs, indent=2))

if __name__ == "__main__": main()
