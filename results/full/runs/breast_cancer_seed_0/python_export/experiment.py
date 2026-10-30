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

SPEC = {'dataset_sha256': '6b649815869a967a273c7f01e9f1c0af6b11f854e99c1a2e854b97d437b4e8a8', 'partitions': {'train': [187, 337, 396, 95, 23, 185, 3, 298, 235, 195, 545, 525, 113, 445, 49, 10, 136, 340, 518, 148, 127, 285, 311, 314, 553, 163, 343, 242, 250, 371, 165, 554, 18, 510, 262, 350, 399, 556, 318, 536, 363, 205, 345, 444, 73, 184, 123, 94, 256, 55, 463, 533, 213, 432, 380, 442, 326, 181, 59, 367, 271, 563, 389, 527, 332, 173, 342, 386, 160, 87, 5, 339, 422, 54, 522, 333, 426, 529, 558, 384, 263, 305, 69, 282, 294, 15, 130, 7, 564, 398, 461, 2, 483, 288, 321, 404, 493, 58, 225, 115, 29, 178, 383, 50, 251, 105, 509, 208, 471, 253, 27, 8, 313, 560, 20, 472, 137, 121, 297, 469, 264, 477, 142, 494, 307, 567, 216, 412, 129, 466, 449, 36, 166, 52, 283, 267, 277, 244, 300, 261, 330, 329, 405, 203, 416, 540, 496, 538, 447, 214, 37, 114, 252, 539, 458, 204, 227, 44, 212, 16, 546, 217, 281, 226, 526, 415, 403, 207, 144, 65, 197, 369, 201, 456, 200, 443, 299, 186, 176, 41, 172, 71, 211, 147, 427, 224, 516, 424, 110, 515, 357, 289, 248, 74, 465, 310, 402, 93, 209, 79, 376, 395, 474, 303, 562, 265, 239, 467, 159, 547, 241, 542, 451, 479, 551, 555, 379, 92, 491, 490, 246, 30, 4, 508, 86, 530, 190, 99, 275, 486, 421, 428, 230, 151, 78, 126, 33, 489, 420, 215, 484, 223, 544, 468, 270, 171, 156, 245, 344, 378, 568, 220, 61, 164, 500, 272, 96, 372, 323, 120, 487, 210, 131, 523, 104, 232, 70, 77, 56, 279, 429, 346, 341, 22, 13, 193, 499, 319, 45, 14, 473, 0, 351, 397, 348, 25, 549, 505, 561, 287, 407, 63, 91, 355, 103, 455, 543, 12, 247, 411, 374, 118, 269, 419, 531, 524, 138, 109, 309, 67, 180, 72, 352, 548, 301, 182, 206, 154, 512, 231, 107, 507, 322, 175, 353, 417, 381, 452, 257, 335, 149, 53, 57, 51, 521, 334, 409, 82, 382, 266, 453], 'validation': [221, 503, 158, 88, 89, 17, 97, 406, 169, 157, 145, 492, 394, 475, 364, 306, 179, 19, 528, 514, 139, 418, 9, 460, 361, 498, 349, 433, 218, 238, 66, 39, 438, 370, 125, 459, 43, 249, 141, 192, 501, 76, 168, 375, 476, 189, 237, 26, 296, 112, 478, 408, 312, 462, 111, 170, 481, 1, 519, 161, 194, 153, 291, 83, 280, 290, 40, 106, 324, 437, 506, 124, 388, 259, 362, 255, 495, 60, 274, 358, 75, 317, 366, 457, 38, 229, 511, 485, 24, 128, 48, 430, 101, 260, 133, 34, 268, 119, 46, 228, 413, 98, 273, 470, 233, 258, 434, 11, 64, 243, 439, 254, 520, 320], 'test': [122, 191, 550, 356, 325, 565, 199, 80, 336, 146, 435, 441, 308, 338, 390, 440, 423, 284, 84, 90, 35, 454, 162, 513, 566, 401, 316, 315, 28, 387, 143, 177, 328, 152, 450, 150, 108, 116, 488, 134, 497, 360, 174, 81, 85, 42, 62, 155, 47, 302, 132, 117, 219, 140, 392, 359, 331, 354, 286, 240, 410, 425, 202, 517, 6, 222, 464, 414, 31, 480, 482, 436, 393, 535, 368, 541, 446, 552, 234, 327, 167, 448, 537, 198, 188, 400, 559, 135, 196, 373, 391, 365, 431, 502, 278, 183, 32, 385, 534, 276, 295, 21, 304, 557, 377, 532, 292, 236, 102, 293, 504, 68, 100, 347]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/breast_cancer.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/breast_cancer.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 0\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '0935dbc1012ed52b92fd393f944365599b70541bb266747d1b8e4910dc0f6a96', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 0, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
