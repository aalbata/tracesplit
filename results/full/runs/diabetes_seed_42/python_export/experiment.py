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

SPEC = {'dataset_sha256': 'c114d448f30bd4157fe94b415bcb11494293a74e3cb21fa2ce0df970085d523b', 'partitions': {'train': [249, 250, 59, 111, 204, 410, 6, 397, 305, 158, 150, 177, 184, 10, 417, 103, 81, 38, 277, 167, 414, 421, 198, 261, 208, 376, 194, 89, 301, 163, 247, 147, 2, 92, 353, 69, 347, 123, 340, 96, 143, 335, 317, 265, 97, 272, 68, 23, 37, 144, 122, 253, 202, 182, 67, 234, 362, 262, 310, 384, 361, 248, 125, 146, 338, 266, 223, 228, 86, 371, 219, 129, 428, 368, 289, 334, 183, 312, 314, 373, 342, 112, 179, 271, 299, 290, 425, 318, 233, 274, 164, 136, 197, 258, 232, 115, 120, 224, 356, 297, 127, 365, 107, 392, 133, 322, 44, 406, 65, 283, 85, 242, 186, 159, 12, 35, 28, 170, 142, 352, 302, 221, 95, 51, 240, 430, 336, 178, 41, 420, 206, 282, 254, 379, 4, 256, 396, 100, 226, 377, 213, 171, 98, 292, 215, 61, 47, 32, 267, 200, 394, 27, 383, 230, 260, 288, 162, 375, 138, 62, 135, 128, 8, 326, 64, 300, 14, 156, 40, 405, 357, 216, 279, 382, 337, 236, 207, 212, 295, 403, 251, 431, 404, 303, 350, 269, 201, 161, 43, 217, 190, 309, 259, 105, 53, 1, 426, 49, 80, 205, 34, 263, 366, 91, 339, 52, 345, 264, 241, 13, 315, 88, 387, 273, 166, 328, 434, 134, 306, 424, 319, 243, 54, 363, 50, 174, 189, 438, 187, 169, 58, 48, 344, 235, 252, 21, 313, 160, 276, 191, 385, 293, 413, 343, 257, 308, 149, 130, 151, 359, 99, 372, 87, 330, 214, 121, 441, 20, 188, 71, 106, 270, 348, 435, 102], 'validation': [285, 255, 370, 349, 57, 124, 331, 193, 33, 84, 346, 378, 56, 42, 333, 439, 108, 389, 244, 323, 408, 3, 5, 386, 418, 74, 321, 246, 395, 422, 22, 440, 311, 77, 354, 402, 185, 118, 278, 137, 433, 180, 222, 17, 281, 113, 286, 119, 401, 268, 117, 294, 412, 325, 220, 101, 15, 153, 411, 139, 19, 358, 24, 181, 0, 381, 26, 157, 192, 419, 211, 11, 109, 7, 145, 173, 237, 18, 391, 296, 63, 416, 9, 203, 45, 329, 245, 332], 'test': [116, 82, 327, 407, 388, 172, 398, 36, 380, 351, 30, 165, 227, 60, 427, 16, 432, 93, 148, 316, 76, 409, 320, 175, 275, 195, 355, 291, 66, 104, 132, 298, 141, 79, 126, 284, 83, 114, 94, 304, 307, 239, 90, 374, 168, 72, 437, 110, 152, 131, 155, 196, 341, 210, 238, 154, 176, 75, 31, 55, 324, 369, 25, 400, 78, 367, 209, 393, 73, 39, 360, 229, 423, 429, 231, 280, 287, 225, 29, 218, 199, 390, 436, 70, 415, 140, 399, 364, 46]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/diabetes.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'rmse', 'models': [{'algorithm': 'ridge', 'line': 5, 'name': 'linear', 'params': {'alpha': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/diabetes.csv"\npredict "target" as regression\nsplit random train 60% validation 20% test 20% seed 42\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear ridge alpha=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by rmse\nassess best on test\nexplain\n', 'source_sha256': '34e2a82da8eb561a7c699042e5f9bfd85b66b7444674a3495ac433dfdff8d409', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 42, 'strategy': 'random'}, 'target': 'target', 'task': 'regression', 'version': 1}}

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
