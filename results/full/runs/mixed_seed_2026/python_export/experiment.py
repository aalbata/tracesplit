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

SPEC = {'dataset_sha256': '5fa5d2b91c9d860ca2a881a251d99ff98b7501c93179795f3c148e7bddc43e17', 'partitions': {'train': [449, 179, 773, 527, 409, 798, 254, 796, 701, 664, 383, 394, 586, 464, 749, 437, 609, 474, 421, 248, 652, 643, 505, 509, 520, 226, 790, 190, 85, 355, 663, 214, 365, 583, 14, 402, 794, 203, 22, 362, 791, 41, 46, 559, 360, 786, 446, 209, 623, 89, 488, 132, 87, 634, 496, 599, 314, 671, 108, 415, 723, 557, 553, 319, 333, 503, 705, 141, 237, 502, 35, 245, 670, 712, 753, 29, 787, 477, 463, 238, 587, 495, 393, 262, 440, 159, 544, 186, 52, 784, 659, 180, 730, 337, 727, 227, 555, 340, 148, 728, 481, 710, 657, 779, 291, 533, 366, 354, 387, 374, 507, 757, 83, 704, 61, 334, 782, 582, 760, 338, 754, 578, 293, 605, 747, 755, 94, 476, 601, 305, 80, 300, 654, 199, 443, 117, 735, 202, 665, 572, 795, 225, 719, 311, 196, 593, 774, 328, 302, 626, 369, 88, 616, 417, 770, 660, 350, 628, 66, 217, 487, 100, 767, 554, 414, 683, 204, 539, 128, 133, 21, 761, 377, 155, 28, 249, 174, 416, 412, 682, 301, 716, 617, 231, 528, 619, 607, 758, 322, 42, 577, 359, 380, 462, 697, 2, 139, 638, 12, 641, 18, 252, 763, 717, 10, 768, 764, 19, 635, 486, 493, 675, 397, 63, 478, 348, 115, 45, 259, 738, 331, 713, 9, 492, 585, 385, 373, 459, 316, 241, 687, 304, 691, 775, 0, 736, 471, 688, 126, 598, 97, 131, 363, 494, 399, 172, 216, 468, 92, 176, 317, 335, 3, 86, 271, 631, 358, 138, 77, 733, 636, 715, 382, 674, 312, 734, 411, 410, 649, 118, 109, 709, 637, 545, 284, 269, 518, 208, 506, 750, 524, 575, 306, 590, 401, 130, 292, 169, 777, 235, 119, 212, 686, 695, 197, 388, 219, 642, 357, 376, 422, 185, 268, 743, 256, 298, 668, 294, 522, 756, 542, 630, 16, 182, 286, 315, 323, 90, 51, 516, 112, 143, 264, 244, 404, 482, 161, 762, 426, 158, 27, 67, 678, 403, 424, 205, 608, 562, 321, 207, 170, 748, 134, 726, 597, 613, 371, 188, 329, 543, 81, 232, 711, 529, 110, 125, 689, 452, 746, 565, 556, 603, 13, 111, 361, 47, 778, 266, 253, 84, 595, 639, 224, 568, 283, 151, 287, 672, 96, 163, 389, 330, 120, 621, 273, 676, 395, 25, 566, 797, 667, 222, 662, 484, 123, 221, 242, 563, 339, 684, 72, 344, 721, 23, 653, 113, 425, 472, 596, 438, 450, 370, 400, 479, 656, 288, 31, 184, 647, 512, 629, 215, 307, 611, 326, 114, 702, 614, 720, 771, 345, 788, 285, 274, 33, 74, 536, 24, 32, 431, 744, 580, 320, 349, 289, 454, 65, 508, 523, 38, 535, 280, 106, 149, 40, 612, 696, 669, 600, 406, 166, 297, 724, 398, 789, 432, 4, 456, 30, 183, 793, 680, 36, 263, 58, 570, 164, 470, 552, 745], 'validation': [661, 103, 375, 515, 201, 277, 175, 469, 624, 651, 50, 243, 772, 473, 532, 740, 480, 737, 220, 195, 497, 766, 265, 549, 633, 490, 78, 725, 741, 588, 391, 160, 594, 560, 26, 324, 722, 116, 43, 234, 739, 82, 673, 228, 57, 258, 799, 276, 498, 17, 121, 783, 20, 11, 519, 420, 550, 433, 465, 378, 44, 444, 466, 167, 390, 752, 347, 356, 645, 419, 625, 602, 418, 685, 351, 341, 392, 439, 187, 384, 436, 457, 367, 193, 448, 368, 213, 513, 364, 589, 511, 278, 5, 576, 352, 455, 708, 427, 229, 218, 461, 408, 230, 295, 574, 501, 627, 655, 211, 405, 39, 558, 310, 780, 445, 129, 178, 251, 526, 648, 475, 210, 690, 618, 140, 413, 136, 200, 500, 152, 769, 640, 428, 442, 548, 153, 372, 703, 233, 1, 700, 658, 606, 584, 60, 145, 261, 223, 53, 147, 694, 6, 69, 71, 458, 343, 198, 434, 342, 257], 'test': [191, 386, 453, 441, 296, 154, 517, 171, 483, 308, 91, 540, 525, 430, 236, 59, 8, 98, 677, 290, 504, 34, 564, 679, 379, 62, 267, 101, 79, 731, 751, 173, 144, 537, 707, 567, 765, 49, 165, 571, 7, 281, 95, 718, 137, 632, 124, 181, 732, 776, 569, 407, 781, 692, 168, 561, 93, 192, 681, 581, 546, 275, 381, 157, 615, 309, 714, 250, 156, 102, 336, 76, 706, 189, 541, 729, 107, 240, 547, 491, 48, 299, 142, 514, 270, 135, 592, 346, 534, 272, 73, 579, 37, 423, 239, 699, 499, 162, 644, 521, 785, 206, 742, 610, 551, 396, 282, 604, 279, 104, 75, 759, 327, 127, 68, 146, 792, 429, 70, 64, 620, 591, 447, 303, 105, 122, 650, 460, 489, 325, 435, 485, 177, 573, 15, 99, 150, 55, 531, 666, 318, 451, 538, 54, 353, 56, 255, 247, 313, 510, 194, 246, 260, 530, 332, 646, 693, 467, 698, 622]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'categorical': ['category', 'band']}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/mixed.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/mixed.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 2026\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '3ca302d4f6dff27dfdfaa79272424134bb3a17b6e9e386e3c65a7f99bc13622a', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 2026, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
