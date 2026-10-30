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

SPEC = {'dataset_sha256': 'c114d448f30bd4157fe94b415bcb11494293a74e3cb21fa2ce0df970085d523b', 'partitions': {'train': [348, 196, 392, 315, 233, 210, 162, 88, 405, 235, 356, 62, 184, 383, 115, 439, 402, 410, 336, 46, 94, 122, 309, 371, 138, 284, 53, 113, 199, 165, 140, 255, 180, 4, 414, 198, 280, 65, 55, 406, 169, 308, 91, 425, 376, 341, 114, 254, 372, 26, 178, 78, 109, 302, 163, 438, 293, 289, 364, 217, 428, 386, 119, 8, 303, 149, 312, 204, 52, 420, 54, 34, 265, 274, 110, 211, 227, 275, 349, 283, 150, 142, 290, 0, 310, 174, 176, 378, 222, 240, 155, 299, 331, 185, 237, 146, 398, 104, 9, 338, 335, 166, 147, 323, 13, 285, 120, 305, 223, 286, 238, 243, 36, 228, 68, 232, 267, 281, 118, 129, 82, 179, 375, 440, 413, 45, 408, 297, 241, 434, 192, 224, 153, 347, 187, 93, 123, 167, 59, 48, 231, 202, 219, 51, 117, 423, 70, 319, 20, 209, 266, 63, 419, 97, 362, 171, 272, 328, 108, 32, 124, 182, 256, 98, 387, 92, 38, 132, 95, 35, 264, 339, 85, 214, 269, 221, 229, 33, 277, 195, 215, 407, 270, 395, 69, 89, 2, 19, 67, 188, 418, 307, 252, 334, 43, 320, 66, 161, 212, 381, 164, 317, 276, 194, 391, 14, 251, 139, 75, 79, 22, 58, 101, 365, 288, 421, 73, 259, 311, 159, 216, 291, 226, 49, 152, 137, 292, 3, 245, 296, 429, 326, 351, 361, 148, 47, 435, 201, 250, 17, 316, 112, 324, 25, 294, 304, 417, 404, 111, 135, 96, 263, 249, 27, 205, 130, 28, 412, 116, 29, 333, 77, 282, 390, 257], 'validation': [191, 40, 314, 206, 355, 246, 121, 190, 278, 76, 239, 271, 105, 298, 236, 71, 183, 384, 61, 154, 86, 329, 370, 74, 295, 416, 141, 332, 208, 84, 352, 377, 87, 399, 56, 18, 107, 313, 431, 394, 218, 41, 136, 5, 350, 126, 160, 433, 1, 60, 337, 24, 411, 234, 247, 197, 430, 396, 133, 80, 379, 403, 21, 37, 50, 262, 400, 44, 6, 23, 360, 172, 106, 424, 409, 426, 99, 72, 16, 437, 81, 31, 318, 200, 441, 367, 10, 382], 'test': [102, 306, 363, 354, 213, 368, 12, 151, 369, 415, 143, 39, 422, 300, 273, 385, 157, 373, 125, 258, 134, 321, 186, 90, 144, 253, 279, 242, 168, 131, 57, 7, 287, 244, 158, 330, 358, 389, 325, 11, 327, 359, 207, 346, 343, 225, 342, 42, 301, 103, 220, 374, 268, 177, 393, 353, 156, 436, 432, 30, 83, 203, 193, 145, 64, 100, 261, 401, 427, 260, 345, 181, 380, 230, 340, 175, 248, 366, 397, 357, 15, 344, 127, 173, 388, 128, 170, 189, 322]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/diabetes.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'rmse', 'models': [{'algorithm': 'ridge', 'line': 5, 'name': 'linear', 'params': {'alpha': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/diabetes.csv"\npredict "target" as regression\nsplit random train 60% validation 20% test 20% seed 2026\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear ridge alpha=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by rmse\nassess best on test\nexplain\n', 'source_sha256': '225376075eeb1de5a5ffda53f8a490ff138afe1a6d51bac3cffc13063a5eae73', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 2026, 'strategy': 'random'}, 'target': 'target', 'task': 'regression', 'version': 1}}

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
