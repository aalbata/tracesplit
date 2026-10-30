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

SPEC = {'dataset_sha256': '6b649815869a967a273c7f01e9f1c0af6b11f854e99c1a2e854b97d437b4e8a8', 'partitions': {'train': [173, 537, 116, 516, 512, 566, 388, 454, 110, 6, 461, 61, 438, 176, 473, 546, 409, 511, 161, 238, 127, 11, 283, 49, 50, 156, 342, 133, 185, 74, 313, 462, 459, 88, 123, 139, 83, 177, 541, 237, 385, 305, 219, 231, 475, 346, 348, 169, 565, 416, 526, 258, 337, 374, 292, 547, 380, 136, 432, 398, 107, 324, 113, 550, 400, 5, 253, 378, 233, 560, 57, 442, 296, 303, 66, 281, 441, 164, 121, 212, 21, 284, 34, 106, 567, 530, 180, 288, 149, 299, 386, 73, 218, 254, 555, 38, 437, 389, 12, 510, 426, 545, 175, 312, 525, 379, 351, 71, 41, 167, 10, 338, 33, 250, 422, 490, 401, 86, 138, 130, 168, 151, 484, 487, 307, 496, 358, 103, 417, 376, 323, 277, 536, 464, 357, 562, 276, 520, 289, 100, 205, 488, 58, 314, 89, 241, 543, 344, 436, 321, 430, 341, 518, 229, 244, 16, 527, 70, 120, 226, 115, 256, 306, 391, 301, 95, 440, 532, 54, 472, 456, 434, 194, 63, 18, 56, 300, 260, 498, 369, 297, 220, 360, 447, 0, 521, 242, 500, 381, 43, 146, 235, 393, 251, 263, 533, 154, 481, 372, 225, 503, 216, 48, 404, 77, 273, 102, 514, 193, 309, 478, 293, 366, 112, 340, 27, 353, 165, 94, 255, 328, 90, 486, 439, 326, 245, 203, 384, 265, 336, 397, 109, 450, 568, 513, 325, 424, 275, 67, 331, 211, 189, 446, 564, 282, 131, 217, 204, 243, 345, 252, 182, 208, 269, 308, 506, 232, 361, 140, 9, 26, 181, 82, 196, 51, 350, 423, 75, 517, 371, 316, 201, 489, 497, 467, 172, 394, 458, 435, 501, 200, 383, 411, 228, 190, 155, 141, 445, 451, 157, 399, 286, 144, 364, 557, 79, 8, 287, 92, 302, 355, 290, 143, 320, 421, 483, 407, 291, 375, 559, 236, 68, 554, 408, 468, 538, 259, 239, 188, 198, 522, 248, 47, 471, 15, 122, 356, 544, 178, 97, 45, 556, 230, 174, 519, 20, 4, 395, 420, 515, 124], 'validation': [294, 531, 91, 433, 135, 22, 170, 171, 85, 222, 529, 548, 126, 455, 78, 36, 84, 428, 425, 69, 191, 7, 528, 30, 363, 221, 261, 491, 392, 42, 64, 499, 377, 479, 267, 298, 270, 76, 59, 186, 268, 2, 55, 413, 158, 262, 280, 347, 274, 23, 335, 227, 25, 553, 246, 162, 469, 62, 209, 415, 29, 332, 431, 443, 552, 334, 87, 93, 44, 266, 159, 40, 249, 329, 477, 452, 295, 37, 448, 387, 349, 119, 46, 223, 150, 13, 427, 311, 539, 524, 339, 132, 195, 390, 272, 542, 523, 480, 410, 405, 460, 343, 551, 125, 561, 549, 264, 370, 504, 108, 466, 495, 142, 240], 'test': [508, 304, 319, 279, 327, 153, 118, 285, 507, 470, 101, 137, 457, 502, 207, 60, 322, 134, 482, 362, 318, 505, 3, 368, 215, 152, 463, 17, 1, 111, 96, 32, 192, 81, 494, 563, 14, 206, 333, 247, 179, 183, 257, 105, 403, 72, 365, 163, 418, 476, 382, 31, 160, 465, 128, 202, 317, 412, 19, 210, 493, 509, 199, 187, 65, 558, 39, 52, 485, 315, 449, 24, 330, 354, 234, 444, 197, 184, 414, 359, 429, 28, 214, 98, 224, 492, 117, 367, 453, 534, 147, 104, 213, 80, 540, 310, 396, 148, 474, 373, 114, 271, 419, 402, 166, 129, 535, 35, 278, 352, 145, 53, 99, 406]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/breast_cancer.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/breast_cancer.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 2026\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '344a0aa3c16eea02fad5ba18c572fb28553a65a2a7b29c6bacc0a89d5de0cfa1', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 2026, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
