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

SPEC = {'dataset_sha256': '5fa5d2b91c9d860ca2a881a251d99ff98b7501c93179795f3c148e7bddc43e17', 'partitions': {'train': [548, 276, 59, 21, 53, 126, 223, 138, 679, 279, 393, 600, 345, 399, 7, 483, 189, 176, 789, 163, 225, 772, 665, 56, 141, 507, 132, 347, 156, 2, 275, 667, 660, 577, 364, 166, 103, 74, 651, 430, 167, 535, 6, 518, 486, 63, 331, 484, 775, 391, 450, 517, 477, 116, 131, 245, 194, 405, 653, 593, 745, 557, 659, 178, 346, 733, 257, 445, 9, 24, 769, 512, 779, 416, 572, 280, 480, 124, 255, 560, 537, 766, 523, 319, 559, 57, 284, 485, 296, 718, 671, 317, 482, 256, 281, 84, 18, 744, 552, 513, 527, 332, 239, 709, 250, 336, 14, 750, 692, 374, 549, 758, 684, 378, 411, 585, 67, 695, 787, 106, 182, 130, 164, 570, 338, 195, 213, 634, 52, 532, 584, 161, 117, 298, 353, 753, 41, 421, 40, 743, 361, 601, 687, 531, 603, 123, 681, 657, 268, 152, 435, 311, 1, 314, 96, 181, 69, 683, 793, 147, 735, 227, 108, 784, 425, 473, 362, 274, 45, 739, 404, 729, 700, 388, 120, 717, 186, 150, 716, 631, 562, 490, 627, 742, 121, 666, 390, 611, 252, 222, 668, 630, 51, 792, 88, 457, 764, 699, 510, 162, 621, 652, 200, 175, 118, 588, 304, 586, 171, 706, 727, 691, 746, 780, 599, 703, 233, 372, 786, 606, 712, 387, 693, 318, 79, 125, 575, 545, 17, 496, 324, 357, 431, 635, 326, 4, 278, 242, 645, 301, 456, 366, 309, 350, 773, 133, 495, 506, 385, 363, 525, 589, 154, 419, 46, 65, 705, 395, 534, 254, 493, 536, 655, 328, 509, 476, 487, 260, 198, 135, 638, 379, 796, 292, 394, 478, 25, 605, 71, 297, 757, 737, 423, 609, 38, 55, 649, 452, 459, 210, 642, 358, 571, 696, 251, 111, 406, 191, 797, 437, 731, 492, 72, 294, 788, 654, 582, 424, 42, 505, 481, 663, 221, 204, 196, 305, 579, 444, 107, 602, 13, 95, 226, 594, 458, 556, 295, 263, 127, 10, 287, 261, 102, 354, 408, 711, 313, 547, 503, 265, 713, 538, 370, 790, 722, 658, 352, 316, 740, 439, 614, 741, 288, 201, 471, 184, 81, 441, 555, 381, 310, 85, 323, 20, 26, 664, 752, 114, 188, 224, 155, 109, 262, 99, 367, 122, 587, 215, 688, 253, 707, 794, 508, 426, 286, 608, 438, 646, 564, 569, 626, 206, 417, 521, 183, 169, 113, 139, 607, 618, 273, 721, 68, 542, 443, 349, 489, 661, 624, 576, 595, 35, 632, 413, 282, 403, 565, 420, 180, 277, 137, 725, 146, 466, 60, 397, 448, 620, 720, 341, 597, 680, 389, 685, 234, 440, 774, 738, 248, 619, 530, 596, 30, 472, 465, 306, 462, 723, 464, 546, 771, 100, 293, 37, 197, 541, 598, 526, 479, 190, 31, 415, 218, 140, 778, 470, 359, 747, 414, 335, 291, 29, 715, 214, 734, 686, 376, 87, 39, 330], 'validation': [266, 561, 22, 583, 230, 19, 170, 522, 264, 322, 93, 500, 23, 101, 16, 43, 672, 327, 340, 710, 762, 356, 383, 344, 247, 339, 75, 760, 433, 728, 343, 633, 333, 567, 625, 799, 776, 446, 12, 402, 11, 329, 748, 592, 682, 398, 129, 783, 791, 308, 89, 491, 428, 629, 369, 211, 145, 48, 670, 551, 355, 173, 656, 229, 76, 719, 228, 98, 689, 70, 461, 637, 105, 677, 365, 540, 32, 377, 554, 453, 749, 187, 299, 524, 761, 86, 172, 604, 112, 142, 128, 236, 134, 515, 474, 217, 650, 158, 28, 36, 241, 447, 451, 615, 153, 119, 50, 736, 785, 58, 307, 174, 573, 488, 115, 144, 590, 690, 300, 436, 754, 220, 285, 469, 209, 610, 702, 694, 730, 270, 179, 237, 244, 303, 449, 249, 94, 544, 15, 568, 97, 553, 148, 543, 591, 647, 110, 539, 697, 205, 269, 238, 337, 511, 767, 216, 321, 240, 64, 533], 'test': [27, 104, 641, 497, 80, 504, 246, 92, 371, 520, 243, 467, 698, 62, 384, 580, 258, 732, 289, 460, 563, 412, 617, 192, 499, 396, 763, 612, 455, 382, 648, 669, 49, 78, 83, 44, 54, 34, 528, 5, 427, 61, 232, 82, 463, 410, 442, 143, 639, 759, 90, 272, 259, 159, 755, 782, 207, 714, 475, 581, 290, 578, 185, 400, 613, 325, 33, 203, 8, 409, 193, 271, 208, 502, 498, 644, 235, 798, 675, 418, 795, 724, 375, 777, 422, 315, 73, 701, 622, 429, 320, 177, 219, 674, 407, 231, 342, 351, 673, 165, 0, 368, 616, 47, 678, 160, 312, 77, 704, 267, 643, 434, 640, 157, 360, 558, 3, 454, 662, 136, 202, 373, 348, 529, 636, 380, 283, 151, 212, 574, 566, 726, 623, 756, 628, 302, 676, 401, 91, 334, 781, 501, 751, 519, 149, 708, 550, 514, 768, 66, 770, 432, 516, 468, 199, 392, 765, 386, 494, 168]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'categorical': ['category', 'band']}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/mixed.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/mixed.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 7\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': 'f84e0d45a741cddffb2d2db8060b434c92f7fa014b4a87e9db6de366e49459c4', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 7, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
