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

SPEC = {'dataset_sha256': 'c114d448f30bd4157fe94b415bcb11494293a74e3cb21fa2ce0df970085d523b', 'partitions': {'train': [191, 116, 168, 367, 120, 218, 112, 303, 136, 308, 221, 126, 196, 332, 424, 133, 33, 419, 162, 34, 44, 97, 85, 61, 167, 199, 173, 229, 236, 433, 376, 250, 108, 407, 46, 357, 372, 263, 73, 137, 422, 246, 253, 29, 181, 374, 301, 51, 365, 391, 217, 432, 27, 2, 259, 156, 255, 417, 346, 138, 212, 104, 310, 306, 311, 215, 348, 189, 214, 35, 230, 204, 234, 293, 239, 210, 67, 24, 295, 216, 223, 129, 111, 166, 349, 40, 317, 79, 431, 350, 13, 364, 436, 228, 161, 83, 439, 416, 110, 149, 152, 16, 299, 109, 312, 139, 237, 260, 318, 355, 248, 342, 19, 269, 226, 340, 3, 309, 125, 280, 286, 77, 184, 354, 420, 275, 294, 384, 182, 395, 80, 307, 258, 11, 331, 86, 266, 36, 423, 58, 41, 270, 50, 209, 352, 351, 368, 123, 222, 62, 379, 130, 187, 23, 43, 0, 201, 377, 98, 304, 429, 178, 256, 94, 409, 95, 390, 169, 69, 305, 48, 341, 207, 279, 227, 148, 143, 334, 180, 356, 131, 425, 440, 262, 324, 203, 84, 121, 383, 406, 91, 82, 267, 119, 358, 291, 57, 321, 257, 394, 42, 105, 408, 273, 392, 38, 53, 385, 128, 290, 28, 183, 370, 163, 151, 244, 202, 31, 32, 127, 185, 413, 288, 398, 402, 147, 285, 410, 177, 99, 338, 335, 197, 243, 115, 404, 265, 72, 333, 25, 165, 337, 426, 174, 428, 39, 193, 314, 396, 88, 70, 87, 292, 242, 277, 211, 9, 359, 195, 251, 323, 192, 117, 47, 172], 'validation': [441, 71, 157, 213, 171, 415, 411, 247, 76, 21, 122, 343, 190, 438, 435, 274, 366, 12, 186, 65, 381, 421, 298, 254, 18, 268, 362, 45, 145, 164, 144, 153, 371, 360, 60, 118, 176, 102, 249, 75, 393, 339, 7, 150, 1, 434, 232, 326, 106, 328, 287, 319, 14, 283, 93, 369, 297, 373, 378, 241, 245, 206, 8, 170, 276, 300, 272, 114, 347, 155, 100, 96, 17, 315, 302, 282, 63, 52, 90, 49, 179, 37, 205, 200, 141, 175, 54, 235], 'test': [159, 92, 132, 15, 135, 64, 399, 380, 325, 26, 363, 89, 403, 264, 146, 188, 344, 10, 124, 6, 160, 238, 400, 313, 56, 20, 316, 134, 353, 345, 261, 320, 414, 296, 198, 330, 412, 389, 4, 103, 289, 375, 154, 5, 281, 208, 224, 427, 387, 194, 361, 418, 158, 231, 388, 22, 219, 430, 59, 140, 336, 401, 113, 252, 327, 405, 101, 107, 271, 142, 386, 225, 397, 78, 68, 233, 30, 66, 437, 81, 240, 278, 220, 284, 322, 74, 382, 55, 329]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/diabetes.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'rmse', 'models': [{'algorithm': 'ridge', 'line': 5, 'name': 'linear', 'params': {'alpha': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/diabetes.csv"\npredict "target" as regression\nsplit random train 60% validation 20% test 20% seed 0\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear ridge alpha=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by rmse\nassess best on test\nexplain\n', 'source_sha256': 'b590044a0f458c112957c8c41580b48c6fe39a7a11eb0a27fcb751e6bee0328c', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 0, 'strategy': 'random'}, 'target': 'target', 'task': 'regression', 'version': 1}}

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
