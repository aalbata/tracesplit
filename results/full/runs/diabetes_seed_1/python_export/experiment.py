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

SPEC = {'dataset_sha256': 'c114d448f30bd4157fe94b415bcb11494293a74e3cb21fa2ce0df970085d523b', 'partitions': {'train': [111, 51, 112, 9, 287, 16, 229, 0, 305, 105, 236, 426, 261, 372, 70, 244, 38, 150, 342, 163, 167, 247, 223, 412, 415, 145, 238, 303, 401, 42, 160, 354, 355, 306, 245, 434, 225, 361, 312, 418, 323, 368, 147, 250, 100, 34, 188, 110, 175, 53, 227, 135, 411, 154, 19, 142, 272, 158, 300, 44, 298, 268, 113, 296, 211, 108, 378, 169, 79, 416, 388, 84, 8, 32, 99, 382, 360, 350, 222, 28, 322, 55, 419, 341, 48, 362, 309, 33, 35, 63, 270, 234, 432, 45, 177, 219, 343, 21, 248, 199, 137, 24, 184, 205, 134, 116, 212, 397, 376, 291, 231, 217, 56, 345, 344, 407, 181, 97, 114, 278, 118, 329, 170, 379, 54, 176, 194, 198, 182, 358, 103, 220, 130, 406, 60, 94, 193, 140, 148, 202, 152, 423, 10, 269, 96, 210, 240, 57, 301, 334, 36, 394, 20, 392, 75, 200, 77, 263, 435, 149, 386, 2, 52, 262, 253, 259, 333, 321, 183, 151, 417, 302, 190, 74, 348, 243, 427, 87, 239, 143, 366, 266, 288, 136, 166, 83, 428, 155, 356, 279, 126, 195, 265, 405, 104, 153, 282, 226, 367, 25, 196, 64, 15, 297, 371, 109, 336, 308, 26, 76, 43, 280, 3, 404, 49, 387, 430, 30, 121, 115, 410, 216, 264, 209, 316, 384, 1, 313, 22, 317, 319, 7, 436, 141, 86, 352, 241, 215, 68, 50, 398, 413, 156, 252, 395, 402, 357, 254, 276, 178, 281, 390, 237, 71, 129, 144, 335, 133, 203, 393, 255, 72, 396, 235, 37], 'validation': [365, 102, 377, 206, 46, 374, 271, 92, 326, 349, 440, 78, 289, 124, 311, 40, 275, 138, 314, 233, 5, 185, 230, 4, 363, 122, 89, 127, 125, 286, 400, 168, 409, 157, 251, 310, 18, 29, 187, 93, 257, 101, 292, 331, 299, 128, 31, 91, 273, 320, 88, 364, 23, 162, 120, 284, 347, 340, 425, 294, 373, 422, 315, 439, 346, 173, 213, 338, 186, 62, 27, 204, 337, 82, 69, 260, 197, 207, 307, 242, 39, 339, 380, 332, 414, 171, 403, 249], 'test': [85, 95, 424, 433, 119, 59, 370, 159, 228, 117, 285, 359, 139, 431, 180, 324, 330, 421, 61, 267, 107, 58, 274, 385, 174, 132, 221, 17, 375, 420, 12, 441, 161, 437, 81, 80, 232, 131, 41, 146, 399, 90, 218, 65, 318, 164, 383, 214, 258, 391, 189, 208, 179, 192, 438, 224, 304, 13, 325, 6, 327, 429, 73, 295, 369, 389, 11, 408, 381, 47, 123, 172, 14, 201, 106, 283, 98, 293, 353, 191, 290, 277, 246, 328, 351, 165, 67, 66, 256]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/diabetes.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'rmse', 'models': [{'algorithm': 'ridge', 'line': 5, 'name': 'linear', 'params': {'alpha': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/diabetes.csv"\npredict "target" as regression\nsplit random train 60% validation 20% test 20% seed 1\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear ridge alpha=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by rmse\nassess best on test\nexplain\n', 'source_sha256': '841af124f5acc08d124d384019ce50184986e1f4f8fe6a791b56f1301aed70e8', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 1, 'strategy': 'random'}, 'target': 'target', 'task': 'regression', 'version': 1}}

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
