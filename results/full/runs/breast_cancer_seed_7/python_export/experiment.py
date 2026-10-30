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

SPEC = {'dataset_sha256': '6b649815869a967a273c7f01e9f1c0af6b11f854e99c1a2e854b97d437b4e8a8', 'partitions': {'train': [178, 227, 73, 127, 85, 349, 509, 494, 367, 34, 272, 448, 229, 235, 25, 310, 559, 490, 361, 408, 525, 452, 511, 502, 530, 312, 344, 154, 387, 288, 398, 99, 128, 411, 293, 204, 438, 269, 94, 500, 379, 403, 223, 410, 539, 306, 303, 290, 322, 501, 105, 395, 186, 177, 234, 279, 556, 206, 319, 461, 479, 474, 164, 110, 168, 197, 240, 172, 222, 207, 376, 469, 405, 92, 132, 100, 357, 35, 548, 13, 346, 292, 510, 8, 392, 130, 31, 70, 291, 307, 218, 17, 507, 5, 350, 146, 215, 60, 554, 95, 193, 406, 551, 138, 19, 211, 20, 296, 456, 116, 275, 449, 529, 209, 351, 173, 11, 483, 487, 444, 55, 135, 30, 219, 432, 255, 427, 53, 59, 140, 484, 202, 422, 242, 527, 473, 550, 347, 282, 14, 343, 142, 390, 533, 133, 313, 285, 528, 377, 45, 176, 520, 445, 480, 558, 382, 84, 274, 141, 259, 88, 40, 76, 565, 90, 526, 107, 185, 391, 508, 239, 238, 9, 174, 256, 471, 433, 93, 153, 430, 365, 534, 538, 368, 512, 184, 478, 124, 326, 161, 467, 345, 516, 232, 325, 136, 333, 459, 369, 61, 210, 56, 535, 21, 162, 226, 299, 65, 400, 397, 148, 495, 71, 470, 49, 287, 524, 69, 44, 29, 309, 159, 353, 52, 389, 147, 407, 537, 77, 496, 317, 190, 302, 409, 339, 300, 145, 358, 50, 562, 261, 271, 32, 252, 119, 27, 236, 481, 101, 383, 179, 224, 506, 421, 258, 384, 2, 41, 315, 465, 354, 3, 81, 286, 504, 72, 7, 158, 314, 231, 426, 304, 123, 57, 260, 37, 455, 270, 437, 522, 104, 79, 125, 237, 67, 543, 267, 439, 549, 225, 505, 169, 402, 194, 187, 468, 243, 265, 43, 497, 106, 394, 386, 323, 213, 241, 342, 182, 96, 134, 253, 385, 113, 378, 87, 15, 139, 567, 198, 370, 18, 201, 419, 458, 488, 477, 114, 247, 475, 83, 420, 429, 208, 301, 435, 316, 46, 203, 297, 440, 281], 'validation': [328, 493, 566, 22, 418, 66, 335, 51, 555, 230, 446, 115, 334, 466, 91, 111, 28, 360, 36, 381, 331, 200, 425, 183, 266, 181, 492, 192, 320, 129, 149, 417, 4, 268, 294, 150, 388, 372, 280, 74, 143, 499, 160, 441, 373, 460, 324, 26, 217, 414, 380, 120, 189, 62, 457, 413, 486, 205, 156, 171, 396, 289, 68, 363, 196, 399, 6, 464, 263, 436, 195, 273, 257, 228, 86, 462, 152, 16, 541, 308, 108, 244, 118, 170, 463, 233, 332, 547, 42, 337, 329, 191, 58, 340, 546, 513, 415, 476, 498, 283, 515, 75, 103, 321, 338, 404, 519, 97, 155, 416, 428, 412, 540, 453], 'test': [560, 482, 485, 295, 451, 298, 553, 454, 48, 401, 112, 254, 1, 424, 24, 165, 246, 126, 144, 434, 330, 167, 352, 82, 536, 472, 552, 518, 431, 563, 450, 423, 39, 180, 131, 63, 248, 442, 489, 523, 212, 276, 33, 214, 393, 278, 514, 245, 327, 122, 10, 544, 531, 121, 151, 362, 98, 264, 262, 220, 175, 355, 364, 0, 199, 277, 521, 447, 561, 284, 517, 503, 188, 375, 64, 348, 166, 356, 38, 12, 250, 557, 89, 311, 221, 341, 117, 78, 305, 23, 371, 366, 443, 102, 532, 359, 545, 564, 249, 374, 137, 318, 163, 491, 80, 336, 568, 47, 54, 109, 216, 157, 251, 542]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29'], 'categorical': []}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/breast_cancer.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/breast_cancer.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 7\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': 'a7a38fa8abf446cfed93873ad7c71db91e135ee8c21220d93b9085fcb9d68ef0', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 7, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
