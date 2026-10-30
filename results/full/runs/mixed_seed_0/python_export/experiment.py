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

SPEC = {'dataset_sha256': '5fa5d2b91c9d860ca2a881a251d99ff98b7501c93179795f3c148e7bddc43e17', 'partitions': {'train': [126, 117, 726, 148, 579, 414, 275, 346, 451, 18, 590, 769, 12, 708, 246, 614, 89, 678, 559, 311, 286, 605, 168, 542, 469, 377, 562, 636, 341, 755, 752, 738, 744, 99, 142, 631, 161, 112, 461, 782, 187, 281, 119, 291, 612, 599, 404, 430, 693, 539, 507, 702, 238, 740, 490, 23, 799, 13, 59, 301, 292, 719, 594, 352, 299, 263, 72, 411, 22, 737, 660, 283, 33, 9, 558, 644, 53, 508, 589, 582, 732, 758, 557, 103, 289, 543, 6, 394, 640, 158, 463, 667, 773, 182, 493, 57, 724, 718, 188, 635, 433, 48, 477, 154, 754, 261, 351, 565, 39, 793, 787, 450, 221, 694, 309, 475, 145, 613, 320, 114, 391, 257, 210, 691, 330, 783, 108, 282, 15, 242, 63, 459, 166, 615, 357, 480, 288, 86, 663, 440, 31, 531, 739, 435, 220, 547, 572, 337, 779, 675, 143, 326, 546, 156, 200, 479, 375, 522, 17, 198, 583, 399, 609, 290, 651, 16, 772, 334, 396, 306, 725, 97, 501, 596, 561, 776, 139, 228, 70, 177, 76, 549, 528, 51, 14, 424, 465, 676, 280, 729, 643, 293, 436, 100, 468, 68, 768, 622, 586, 294, 165, 641, 65, 457, 588, 503, 585, 91, 784, 361, 116, 87, 711, 616, 552, 67, 27, 47, 42, 381, 566, 224, 71, 581, 568, 671, 144, 690, 736, 494, 66, 788, 52, 431, 84, 680, 24, 373, 672, 81, 786, 336, 1, 312, 628, 240, 603, 123, 383, 356, 592, 710, 239, 556, 279, 208, 43, 390, 403, 580, 454, 606, 307, 223, 327, 212, 129, 94, 169, 203, 124, 56, 2, 152, 344, 366, 655, 324, 662, 155, 134, 305, 639, 722, 353, 398, 206, 113, 255, 617, 421, 504, 598, 29, 610, 58, 520, 545, 515, 731, 453, 49, 482, 104, 474, 321, 342, 577, 287, 489, 650, 167, 733, 217, 619, 249, 470, 785, 128, 497, 181, 314, 150, 764, 345, 601, 34, 343, 536, 525, 569, 429, 138, 385, 604, 304, 360, 140, 21, 78, 794, 28, 364, 201, 328, 315, 105, 526, 265, 245, 462, 131, 674, 73, 756, 267, 272, 618, 444, 40, 524, 677, 111, 395, 380, 192, 4, 437, 115, 684, 432, 699, 218, 681, 775, 751, 273, 534, 748, 313, 194, 647, 405, 189, 45, 77, 172, 382, 162, 204, 98, 443, 795, 197, 720, 199, 130, 25, 285, 654, 488, 483, 466, 209, 467, 476, 473, 402, 316, 350, 657, 509, 60, 253, 19, 92, 498, 426, 423, 791, 620, 548, 237, 317, 136, 689, 310, 544, 584, 514, 434, 516, 757, 231, 555, 420, 763, 110, 247, 264, 727, 506, 269, 665, 153, 401, 460, 487, 518, 789, 95, 634, 55, 535, 149, 225, 236, 707, 481, 735, 656, 107, 428, 685, 560, 205, 322, 529, 780, 36, 386, 37, 513, 593, 262, 749, 521, 502, 540, 792], 'validation': [193, 767, 624, 574, 696, 90, 478, 180, 519, 118, 750, 349, 495, 278, 704, 0, 408, 658, 216, 491, 258, 778, 621, 46, 83, 335, 571, 649, 697, 688, 362, 354, 570, 379, 196, 359, 400, 332, 8, 427, 323, 340, 670, 632, 96, 464, 456, 669, 774, 88, 705, 575, 541, 762, 215, 202, 38, 765, 438, 252, 409, 151, 230, 415, 666, 419, 106, 271, 173, 235, 365, 303, 576, 234, 418, 623, 251, 274, 527, 284, 664, 388, 638, 392, 184, 627, 44, 445, 32, 80, 406, 761, 295, 371, 723, 232, 243, 26, 486, 637, 692, 393, 69, 141, 276, 796, 537, 712, 157, 417, 3, 211, 517, 701, 591, 160, 318, 452, 185, 659, 512, 446, 412, 550, 260, 698, 683, 120, 171, 597, 448, 629, 370, 695, 270, 125, 376, 510, 170, 389, 766, 363, 229, 716, 646, 747, 600, 358, 442, 642, 679, 777, 256, 661, 75, 441, 668, 554, 339, 259], 'test': [770, 298, 325, 741, 458, 630, 715, 673, 471, 186, 595, 226, 625, 686, 607, 333, 422, 207, 717, 781, 746, 492, 176, 331, 190, 179, 85, 551, 771, 175, 653, 499, 496, 30, 687, 132, 254, 367, 734, 122, 372, 564, 79, 538, 93, 7, 109, 730, 700, 797, 135, 387, 505, 511, 227, 578, 308, 626, 214, 121, 244, 164, 5, 248, 523, 178, 645, 567, 648, 455, 587, 347, 195, 573, 302, 137, 54, 652, 277, 213, 101, 300, 416, 219, 703, 374, 62, 413, 439, 319, 378, 447, 728, 759, 191, 530, 485, 74, 533, 532, 296, 500, 355, 753, 147, 745, 709, 64, 713, 297, 159, 163, 61, 82, 41, 11, 348, 233, 338, 743, 397, 222, 798, 611, 174, 133, 682, 553, 407, 790, 608, 241, 102, 721, 410, 425, 10, 369, 384, 602, 146, 472, 50, 633, 449, 183, 484, 760, 329, 706, 266, 127, 20, 35, 742, 714, 250, 563, 368, 268]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'categorical': ['category', 'band']}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/mixed.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/mixed.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 0\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '17f01e83f9a31146b31f27142c8a950ab8a0a0304ad7ee07067cabbc8a006d86', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 0, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
