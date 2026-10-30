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

SPEC = {'dataset_sha256': '6b649815869a967a273c7f01e9f1c0af6b11f854e99c1a2e854b97d437b4e8a8', 'partitions': {'train': [407, 48, 532, 468, 243, 190, 126, 491, 257, 566, 66, 489, 546, 142, 425, 559, 526, 375, 123, 358, 324, 19, 225, 36, 370, 108, 44, 347, 549, 378, 333, 322, 351, 436, 392, 84, 162, 506, 449, 67, 305, 427, 16, 350, 275, 528, 181, 459, 5, 524, 232, 276, 443, 330, 133, 233, 229, 444, 466, 72, 544, 282, 182, 269, 475, 452, 137, 310, 473, 148, 568, 556, 470, 24, 18, 105, 47, 297, 238, 421, 89, 171, 56, 114, 286, 445, 516, 4, 365, 291, 522, 189, 536, 227, 379, 0, 140, 496, 118, 90, 64, 254, 453, 51, 220, 136, 146, 198, 151, 104, 414, 293, 39, 93, 352, 184, 298, 340, 547, 134, 98, 224, 400, 552, 319, 296, 212, 159, 418, 476, 485, 250, 55, 328, 308, 307, 419, 163, 488, 535, 155, 499, 542, 42, 223, 217, 187, 309, 204, 336, 541, 461, 205, 82, 38, 294, 76, 500, 244, 271, 558, 385, 172, 388, 46, 486, 14, 49, 120, 451, 13, 497, 543, 245, 68, 438, 62, 353, 503, 467, 115, 239, 555, 150, 176, 265, 390, 540, 313, 230, 518, 478, 348, 40, 218, 551, 119, 389, 525, 393, 495, 331, 417, 208, 563, 364, 316, 454, 430, 334, 377, 97, 507, 337, 199, 345, 433, 52, 194, 397, 251, 202, 165, 169, 214, 168, 58, 368, 216, 80, 17, 362, 538, 455, 6, 562, 287, 539, 369, 2, 139, 85, 567, 193, 387, 228, 160, 356, 323, 200, 471, 152, 422, 141, 107, 43, 553, 278, 502, 395, 423, 465, 357, 426, 81, 31, 267, 446, 424, 63, 521, 344, 317, 71, 439, 511, 54, 69, 78, 481, 203, 50, 360, 32, 96, 565, 361, 241, 109, 472, 396, 30, 129, 110, 487, 158, 383, 248, 447, 329, 302, 354, 498, 483, 211, 268, 429, 398, 22, 125, 406, 448, 240, 94, 12, 292, 303, 75, 440, 149, 175, 177, 537, 37, 34, 520, 207, 11, 550, 86, 77, 508, 144, 259, 273, 557, 242, 479, 509, 113, 161], 'validation': [95, 363, 157, 529, 380, 376, 458, 306, 260, 457, 289, 382, 386, 420, 261, 253, 246, 262, 215, 517, 325, 408, 221, 179, 560, 101, 209, 102, 320, 45, 28, 441, 534, 116, 374, 249, 231, 88, 180, 196, 339, 9, 301, 255, 145, 533, 318, 26, 74, 437, 272, 153, 167, 53, 493, 431, 183, 394, 384, 135, 480, 462, 280, 83, 23, 335, 60, 29, 403, 25, 33, 371, 456, 355, 124, 174, 514, 235, 173, 170, 73, 399, 274, 117, 482, 263, 41, 548, 59, 226, 515, 409, 1, 70, 311, 147, 130, 219, 270, 474, 442, 21, 186, 312, 484, 258, 332, 295, 143, 213, 112, 164, 122, 561], 'test': [460, 166, 191, 266, 342, 10, 111, 290, 91, 252, 27, 381, 373, 127, 128, 435, 341, 279, 65, 314, 247, 7, 410, 477, 99, 201, 8, 132, 510, 92, 79, 519, 367, 554, 210, 405, 359, 284, 277, 57, 185, 154, 300, 490, 469, 530, 494, 285, 527, 428, 523, 512, 192, 432, 413, 61, 338, 402, 564, 304, 391, 281, 501, 531, 3, 256, 100, 106, 237, 450, 288, 138, 156, 504, 103, 513, 327, 434, 349, 401, 87, 264, 283, 299, 505, 35, 195, 372, 545, 131, 411, 234, 463, 415, 464, 492, 326, 346, 121, 206, 404, 236, 321, 15, 343, 20, 178, 188, 315, 412, 366, 197, 222, 416]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/breast_cancer.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/breast_cancer.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 1\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '99a3eb8d14d2a8b6bfc388a135ab063fbe1c174ed6254f422af98b432d8ab3dd', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 1, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
