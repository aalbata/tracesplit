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

SPEC = {'dataset_sha256': '6b649815869a967a273c7f01e9f1c0af6b11f854e99c1a2e854b97d437b4e8a8', 'partitions': {'train': [525, 39, 58, 342, 229, 40, 172, 213, 489, 413, 559, 480, 560, 552, 360, 183, 187, 362, 453, 380, 0, 287, 341, 262, 543, 294, 163, 332, 382, 195, 339, 435, 181, 485, 364, 220, 391, 551, 441, 250, 351, 234, 286, 18, 94, 336, 524, 311, 238, 426, 82, 550, 425, 400, 563, 491, 327, 226, 119, 115, 154, 148, 331, 15, 60, 473, 21, 302, 521, 2, 190, 303, 240, 555, 404, 47, 25, 207, 107, 197, 418, 281, 189, 442, 46, 209, 376, 78, 357, 101, 443, 486, 135, 191, 304, 389, 63, 374, 215, 106, 384, 280, 239, 323, 19, 30, 11, 55, 558, 241, 222, 109, 401, 174, 216, 297, 547, 97, 502, 503, 6, 72, 221, 356, 493, 248, 34, 259, 252, 373, 469, 333, 45, 112, 379, 69, 270, 449, 225, 392, 520, 402, 276, 290, 186, 36, 512, 100, 88, 436, 279, 4, 157, 59, 165, 538, 151, 42, 414, 396, 454, 170, 133, 114, 31, 313, 89, 117, 253, 35, 129, 27, 293, 533, 233, 475, 463, 134, 386, 120, 317, 136, 274, 531, 50, 162, 565, 22, 76, 465, 91, 562, 236, 557, 515, 28, 105, 277, 193, 282, 29, 264, 312, 188, 514, 218, 210, 10, 530, 474, 430, 451, 196, 350, 299, 249, 160, 330, 393, 68, 126, 118, 124, 90, 539, 243, 166, 420, 462, 61, 505, 145, 394, 208, 219, 272, 378, 201, 421, 5, 322, 152, 132, 245, 255, 347, 446, 346, 448, 153, 271, 214, 169, 517, 52, 427, 479, 500, 32, 456, 227, 146, 545, 161, 164, 273, 471, 438, 429, 224, 142, 409, 131, 211, 185, 173, 130, 254, 546, 371, 329, 155, 92, 33, 459, 108, 125, 457, 458, 144, 179, 365, 522, 566, 285, 57, 71, 306, 143, 159, 12, 424, 437, 377, 309, 416, 315, 410, 16, 87, 111, 147, 289, 405, 419, 338, 49, 56, 175, 554, 263, 9, 283, 314, 178, 544, 481, 455, 510, 476, 335, 478, 319, 368, 388, 499, 487, 184, 367, 139, 230], 'validation': [84, 507, 434, 95, 407, 445, 307, 506, 516, 403, 496, 470, 399, 96, 536, 298, 513, 102, 343, 328, 541, 206, 137, 326, 98, 467, 518, 267, 291, 65, 511, 85, 296, 535, 519, 268, 318, 369, 490, 83, 235, 532, 116, 408, 247, 385, 556, 158, 395, 370, 527, 150, 352, 494, 363, 14, 468, 354, 509, 325, 251, 223, 349, 423, 70, 320, 334, 156, 372, 121, 321, 232, 344, 242, 534, 26, 217, 202, 422, 104, 492, 498, 433, 464, 237, 66, 292, 7, 383, 359, 41, 203, 501, 194, 180, 228, 361, 324, 93, 198, 64, 48, 261, 24, 432, 549, 466, 316, 537, 257, 483, 477, 497, 182], 'test': [340, 1, 113, 73, 452, 38, 149, 358, 205, 86, 199, 176, 568, 482, 204, 529, 375, 450, 428, 284, 138, 140, 246, 540, 495, 567, 275, 8, 381, 231, 192, 43, 564, 397, 444, 244, 412, 23, 447, 81, 398, 54, 353, 3, 415, 553, 74, 122, 80, 77, 103, 305, 528, 288, 256, 269, 387, 508, 75, 390, 266, 411, 523, 561, 128, 265, 310, 348, 439, 484, 542, 548, 200, 67, 212, 488, 99, 177, 258, 308, 167, 440, 417, 355, 53, 526, 260, 79, 366, 345, 127, 461, 168, 110, 337, 17, 300, 44, 472, 171, 141, 20, 13, 406, 295, 460, 62, 51, 504, 278, 301, 123, 37, 431]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/breast_cancer.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/breast_cancer.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 42\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': 'e7364d5d5e790a2b4b3326afb5764adfbc311f0f2129cc0bece7d4f2c0b84aa1', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 42, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
