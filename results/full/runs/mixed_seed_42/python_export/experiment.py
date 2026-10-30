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

SPEC = {'dataset_sha256': '5fa5d2b91c9d860ca2a881a251d99ff98b7501c93179795f3c148e7bddc43e17', 'partitions': {'train': [385, 369, 530, 325, 459, 335, 663, 669, 760, 222, 287, 524, 343, 774, 523, 763, 216, 67, 361, 266, 81, 456, 160, 309, 592, 623, 332, 411, 124, 32, 598, 778, 64, 357, 415, 126, 11, 417, 666, 495, 422, 392, 387, 165, 401, 200, 161, 239, 597, 599, 46, 192, 630, 371, 710, 298, 354, 596, 128, 31, 434, 150, 572, 293, 169, 487, 514, 436, 660, 464, 252, 217, 754, 465, 338, 90, 481, 171, 131, 275, 206, 389, 156, 396, 545, 284, 496, 167, 295, 566, 30, 718, 384, 633, 305, 52, 311, 19, 7, 262, 728, 542, 688, 474, 41, 667, 441, 59, 71, 595, 745, 642, 640, 615, 539, 147, 713, 272, 72, 741, 552, 687, 528, 468, 51, 515, 555, 651, 223, 653, 626, 460, 420, 162, 244, 399, 297, 752, 438, 21, 451, 690, 654, 278, 187, 393, 447, 764, 249, 519, 10, 734, 749, 115, 333, 24, 742, 280, 221, 556, 376, 748, 157, 168, 755, 285, 194, 558, 265, 671, 274, 40, 469, 581, 45, 395, 580, 650, 26, 218, 60, 224, 38, 770, 386, 334, 585, 312, 609, 780, 153, 589, 121, 765, 212, 445, 391, 337, 567, 729, 174, 639, 271, 632, 79, 723, 631, 771, 98, 48, 726, 504, 74, 351, 672, 307, 489, 269, 600, 144, 268, 318, 344, 290, 336, 518, 205, 143, 628, 2, 758, 602, 308, 235, 601, 427, 397, 644, 203, 779, 65, 148, 668, 257, 627, 18, 276, 507, 543, 49, 574, 506, 786, 751, 181, 63, 145, 685, 140, 583, 532, 453, 210, 292, 662, 488, 606, 701, 364, 193, 154, 303, 133, 466, 493, 681, 86, 793, 15, 622, 127, 36, 634, 591, 498, 490, 331, 616, 146, 527, 233, 253, 480, 753, 782, 432, 475, 675, 277, 182, 787, 315, 610, 117, 773, 310, 66, 197, 110, 44, 738, 243, 419, 559, 178, 579, 440, 563, 362, 28, 783, 569, 103, 594, 503, 228, 125, 282, 794, 425, 412, 461, 95, 661, 676, 341, 715, 421, 777, 152, 702, 281, 724, 409, 6, 546, 402, 695, 743, 534, 176, 494, 113, 408, 709, 684, 390, 458, 135, 791, 679, 562, 704, 323, 693, 536, 99, 220, 114, 398, 3, 647, 360, 136, 226, 373, 322, 410, 317, 607, 294, 261, 717, 400, 92, 716, 353, 119, 720, 450, 76, 656, 426, 761, 646, 339, 301, 405, 286, 230, 137, 454, 151, 788, 448, 692, 47, 82, 61, 340, 227, 576, 367, 34, 134, 795, 549, 776, 366, 732, 54, 9, 83, 757, 652, 319, 296, 790, 548, 22, 733, 618, 551, 746, 772, 120, 766, 473, 722, 478, 512, 418, 238, 229, 42, 467, 457, 664, 189, 356, 104, 100, 215, 246, 696, 797, 234, 27, 769, 87, 665, 1, 368, 329, 620, 91, 190, 270, 680, 247, 365, 522, 242, 23, 605, 259, 789, 575, 691, 158], 'validation': [577, 69, 740, 188, 102, 186, 799, 775, 570, 501, 645, 509, 105, 568, 58, 241, 731, 686, 414, 586, 424, 711, 712, 122, 84, 578, 471, 472, 116, 29, 698, 328, 347, 35, 587, 584, 525, 302, 17, 225, 406, 166, 517, 621, 736, 55, 730, 299, 207, 719, 237, 123, 179, 407, 267, 13, 625, 624, 537, 452, 744, 89, 739, 449, 416, 183, 177, 737, 435, 433, 111, 33, 462, 617, 346, 747, 231, 785, 430, 349, 232, 781, 85, 358, 694, 470, 380, 767, 689, 370, 682, 25, 636, 670, 56, 505, 582, 413, 593, 254, 213, 374, 796, 142, 655, 604, 43, 382, 97, 727, 199, 191, 673, 5, 557, 350, 714, 141, 511, 513, 255, 619, 550, 708, 700, 8, 88, 721, 12, 735, 108, 643, 508, 93, 159, 638, 251, 492, 279, 324, 326, 175, 14, 429, 236, 101, 273, 4, 256, 612, 375, 677, 109, 544, 132, 520, 77, 792, 394, 68], 'test': [264, 57, 316, 404, 291, 16, 560, 547, 260, 383, 94, 381, 263, 649, 611, 70, 163, 697, 614, 798, 538, 209, 762, 500, 388, 750, 106, 379, 497, 531, 759, 526, 201, 516, 590, 50, 184, 314, 240, 304, 118, 444, 588, 288, 185, 678, 706, 553, 659, 455, 437, 502, 342, 564, 423, 180, 657, 39, 96, 683, 443, 78, 300, 208, 521, 756, 363, 648, 196, 705, 699, 345, 155, 164, 482, 348, 149, 554, 637, 313, 306, 463, 327, 499, 635, 561, 377, 107, 330, 565, 629, 75, 170, 483, 674, 541, 173, 80, 214, 535, 355, 510, 529, 573, 211, 403, 641, 485, 658, 431, 491, 428, 129, 540, 245, 202, 283, 486, 603, 73, 703, 439, 321, 442, 477, 250, 378, 53, 359, 608, 20, 37, 320, 62, 352, 139, 204, 768, 130, 112, 172, 289, 533, 479, 219, 613, 372, 784, 484, 476, 258, 725, 446, 138, 0, 248, 198, 571, 707, 195]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'categorical': ['category', 'band']}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/mixed.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/mixed.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 42\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '5bbd2d0b78df4bed3157d99ca716501b7138f098b28cec034d8860344c8c28b1', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 42, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
