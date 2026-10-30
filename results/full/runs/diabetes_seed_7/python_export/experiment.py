"""Standalone Python replay. Only public scientific Python libraries are used.

The split is frozen to separate execution equivalence from split generation.
The source dataset is fingerprint-checked. Editing this file removes any
claim that it is the unmodified export tested in the research artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

SPEC = {'dataset_sha256': 'c114d448f30bd4157fe94b415bcb11494293a74e3cb21fa2ce0df970085d523b', 'partitions': {'train': [20, 55, 342, 393, 347, 34, 181, 150, 406, 385, 246, 263, 97, 350, 154, 122, 113, 240, 292, 243, 108, 376, 94, 228, 125, 277, 423, 91, 419, 372, 253, 368, 380, 362, 24, 336, 307, 437, 205, 318, 126, 360, 78, 245, 178, 58, 382, 328, 223, 324, 404, 61, 308, 93, 378, 264, 69, 36, 40, 160, 313, 11, 300, 144, 303, 434, 171, 127, 319, 316, 184, 46, 9, 367, 394, 176, 236, 231, 210, 128, 143, 241, 7, 254, 412, 295, 8, 156, 27, 64, 255, 321, 153, 48, 208, 1, 269, 265, 288, 294, 95, 427, 281, 116, 45, 161, 396, 117, 77, 346, 63, 230, 186, 297, 199, 301, 121, 409, 381, 158, 85, 225, 262, 355, 76, 344, 330, 87, 441, 182, 105, 49, 5, 17, 234, 352, 106, 134, 358, 226, 214, 190, 390, 320, 282, 29, 169, 180, 100, 197, 209, 135, 414, 379, 198, 166, 341, 136, 285, 335, 131, 39, 215, 43, 54, 417, 299, 401, 304, 35, 315, 309, 276, 124, 142, 110, 323, 332, 386, 107, 425, 12, 272, 47, 203, 377, 119, 21, 83, 130, 329, 229, 433, 286, 123, 33, 415, 4, 19, 219, 438, 356, 149, 224, 247, 421, 151, 317, 291, 233, 10, 279, 71, 252, 92, 349, 374, 192, 257, 339, 430, 195, 343, 137, 260, 38, 201, 411, 290, 189, 251, 112, 183, 152, 312, 325, 191, 44, 275, 6, 250, 311, 75, 0, 391, 383, 68, 167, 392, 218, 42, 366, 345, 72, 23, 398, 185, 348, 103, 407, 211, 67, 25, 196, 175], 'validation': [359, 187, 403, 256, 322, 188, 101, 333, 82, 287, 284, 416, 222, 293, 114, 179, 172, 90, 115, 193, 2, 164, 204, 327, 439, 266, 334, 206, 60, 129, 132, 238, 436, 371, 200, 310, 81, 424, 395, 249, 168, 159, 258, 365, 426, 306, 30, 402, 242, 155, 305, 369, 420, 109, 400, 133, 22, 53, 353, 145, 73, 80, 98, 338, 232, 14, 202, 244, 237, 387, 331, 351, 217, 280, 314, 259, 52, 96, 32, 3, 302, 364, 57, 268, 428, 408, 289, 18], 'test': [361, 357, 59, 141, 146, 138, 435, 388, 74, 147, 28, 212, 88, 273, 397, 173, 66, 111, 429, 70, 174, 157, 56, 16, 227, 340, 41, 220, 140, 51, 221, 422, 440, 375, 120, 79, 26, 207, 165, 413, 62, 239, 15, 213, 31, 65, 298, 418, 389, 86, 50, 274, 354, 296, 170, 102, 278, 267, 410, 194, 84, 37, 431, 216, 337, 261, 162, 139, 163, 118, 363, 373, 89, 104, 271, 148, 248, 270, 370, 13, 405, 399, 283, 326, 432, 177, 384, 235, 99]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/diabetes.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'rmse', 'models': [{'algorithm': 'ridge', 'line': 5, 'name': 'linear', 'params': {'alpha': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/diabetes.csv"\npredict "target" as regression\nsplit random train 60% validation 20% test 20% seed 7\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear ridge alpha=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by rmse\nassess best on test\nexplain\n', 'source_sha256': 'd92dba1d839b9ca392b00f01732ab23a173288680e7c96ec8e11709853befb54', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 7, 'strategy': 'random'}, 'target': 'target', 'task': 'regression', 'version': 1}}

def run(data_path=None):
    base = Path(__file__).resolve().parent
    path = Path(data_path) if data_path is not None else base / "data.csv"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC["dataset_sha256"]:
        raise ValueError("The CSV fingerprint differs from the exported dataset.")
    df = pd.read_csv(path, float_precision="round_trip")
    p, ids = SPEC["plan"], SPEC["partitions"]
    features = SPEC["schema"]
    numeric, categorical = features["numeric"], features["categorical"]
    blocks = []
    if numeric:
        steps = [SimpleImputer(strategy=p["prepare"]["numeric"], keep_empty_features=True)]
        if p["prepare"]["scale"] == "standard": steps.append(StandardScaler())
        blocks.append(("numeric", make_pipeline(*steps), numeric))
    if categorical:
        blocks.append(("categorical", make_pipeline(SimpleImputer(strategy=p["prepare"]["categorical"], fill_value="__MISSING__", keep_empty_features=True),
                                                        OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical))
    transformer = ColumnTransformer(blocks, remainder="drop")
    y = df[p["target"]]; train, valid, test = (ids[k] for k in ("train", "validation", "test"))
    def metrics(actual, pred):
        if p["task"] == "classification":
            return {"accuracy": float(accuracy_score(actual, pred)), "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
                    "macro_f1": float(f1_score(actual, pred, average="macro", zero_division=0))}
        return {"rmse": float(root_mean_squared_error(actual, pred)), "mae": float(mean_absolute_error(actual, pred)), "r2": float(r2_score(actual, pred))}
    models, scores, predictions = {}, {}, {}
    with threadpool_limits(limits=1):
        Xtrain = transformer.fit_transform(df.iloc[train])
        Xvalid = transformer.transform(df.iloc[valid])
        for spec in p["models"]:
            name, kind, par = spec["name"], spec["algorithm"], spec["params"]
            seed = p["split"]["seed"]
            if kind == "logistic": estimator = LogisticRegression(C=par.get("C", 1.0), solver="lbfgs", max_iter=2000, random_state=seed)
            elif kind == "ridge": estimator = Ridge(alpha=par.get("alpha", 1.0), solver="cholesky")
            elif kind == "forest":
                cls = RandomForestClassifier if p["task"] == "classification" else RandomForestRegressor
                estimator = cls(n_estimators=par.get("trees", 40), max_depth=par.get("depth", None), random_state=seed, n_jobs=1)
            else:
                cls = DecisionTreeClassifier if p["task"] == "classification" else DecisionTreeRegressor
                estimator = cls(max_depth=par.get("depth", None), random_state=seed)
            estimator.fit(Xtrain, y.iloc[train]); models[name] = estimator
            pred = estimator.predict(Xvalid); predictions[name] = pred.tolist()
            scores[name] = metrics(y.iloc[valid], pred)
        ordered = [item["name"] for item in p["models"]]
        values = [scores[name][p["metric"]] for name in ordered]
        position = int(np.argmin(values)) if p["metric"] in {"rmse", "mae"} else int(np.argmax(values))
        selected = ordered[position]
        test_pred = models[selected].predict(transformer.transform(df.iloc[test]))
        result = {"selected": selected, "validation": scores, "validation_predictions": predictions,
                  "test": metrics(y.iloc[test], test_pred), "test_predictions": test_pred.tolist()}
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("replay_result.json"))
    args = parser.parse_args()
    result = run(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"selected": result["selected"], "test": result["test"]}, indent=2))
