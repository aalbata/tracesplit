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

SPEC = {'dataset_sha256': '5fa5d2b91c9d860ca2a881a251d99ff98b7501c93179795f3c148e7bddc43e17', 'partitions': {'train': [255, 277, 262, 661, 587, 174, 213, 273, 539, 321, 412, 468, 515, 183, 603, 643, 143, 117, 351, 647, 644, 222, 122, 736, 471, 218, 631, 163, 480, 243, 705, 74, 773, 698, 782, 283, 137, 689, 599, 794, 533, 445, 217, 346, 208, 659, 532, 98, 516, 687, 116, 535, 495, 756, 510, 499, 494, 100, 577, 235, 534, 245, 464, 380, 40, 71, 453, 569, 341, 552, 61, 26, 492, 748, 734, 82, 135, 394, 717, 456, 443, 285, 335, 80, 367, 144, 791, 640, 612, 463, 379, 760, 399, 620, 601, 714, 766, 404, 737, 763, 207, 787, 377, 195, 638, 648, 692, 582, 196, 286, 348, 221, 162, 512, 784, 786, 374, 417, 757, 686, 171, 560, 536, 518, 407, 288, 142, 312, 390, 733, 239, 588, 645, 184, 543, 91, 375, 438, 765, 324, 210, 110, 626, 607, 189, 86, 581, 726, 364, 164, 747, 166, 225, 38, 400, 328, 343, 610, 48, 246, 92, 600, 327, 428, 202, 731, 520, 186, 490, 371, 487, 70, 95, 682, 83, 467, 46, 36, 699, 451, 424, 491, 127, 544, 474, 297, 775, 89, 25, 20, 735, 591, 436, 139, 329, 797, 732, 551, 498, 613, 94, 176, 681, 725, 140, 486, 373, 604, 230, 634, 411, 125, 472, 497, 357, 53, 54, 422, 432, 621, 755, 457, 37, 509, 502, 85, 99, 656, 575, 354, 454, 768, 310, 298, 563, 561, 168, 292, 522, 493, 261, 488, 564, 531, 209, 507, 529, 256, 545, 672, 727, 668, 156, 541, 214, 191, 338, 444, 307, 406, 745, 617, 88, 347, 102, 718, 13, 326, 69, 688, 318, 319, 60, 23, 130, 660, 147, 781, 376, 31, 549, 386, 242, 642, 614, 381, 152, 388, 232, 305, 500, 334, 333, 738, 511, 30, 39, 743, 301, 711, 68, 562, 0, 358, 33, 426, 514, 294, 434, 355, 66, 181, 473, 57, 254, 115, 231, 700, 233, 77, 416, 754, 50, 2, 90, 690, 615, 360, 477, 528, 409, 414, 566, 132, 10, 752, 251, 133, 695, 517, 764, 630, 22, 437, 118, 107, 618, 113, 362, 271, 353, 508, 722, 653, 759, 772, 123, 606, 295, 204, 397, 198, 227, 331, 462, 101, 574, 703, 129, 719, 387, 447, 597, 5, 796, 761, 75, 720, 646, 268, 778, 1, 12, 730, 570, 704, 568, 452, 238, 776, 654, 658, 51, 415, 670, 237, 352, 547, 623, 641, 739, 76, 32, 790, 691, 104, 674, 655, 304, 21, 596, 383, 316, 55, 267, 461, 274, 177, 697, 220, 589, 777, 496, 484, 479, 678, 244, 675, 542, 78, 131, 173, 165, 180, 750, 18, 554, 276, 780, 250, 446, 559, 540, 278, 624, 187, 584, 538, 344, 182, 799, 141, 702, 124, 785, 322, 330, 593, 3, 448, 365, 111, 259, 136, 657, 384, 161, 489, 203, 435, 466, 279, 229, 199, 592, 169, 216, 427, 572, 252], 'validation': [119, 578, 771, 311, 372, 42, 521, 378, 361, 160, 556, 676, 35, 712, 81, 684, 236, 158, 710, 795, 431, 56, 114, 219, 272, 441, 685, 481, 583, 93, 741, 302, 413, 112, 138, 266, 537, 206, 7, 715, 683, 253, 234, 513, 393, 504, 629, 440, 369, 303, 667, 188, 350, 571, 429, 608, 580, 128, 662, 627, 320, 58, 585, 789, 402, 249, 332, 293, 696, 157, 185, 6, 649, 650, 315, 526, 43, 16, 744, 708, 248, 389, 63, 398, 87, 275, 728, 155, 410, 401, 241, 366, 519, 707, 108, 483, 723, 59, 224, 192, 73, 458, 317, 8, 713, 323, 523, 193, 194, 506, 349, 553, 342, 762, 103, 263, 635, 625, 721, 527, 501, 530, 450, 257, 148, 175, 557, 212, 67, 405, 178, 679, 290, 475, 746, 201, 567, 565, 260, 281, 299, 200, 223, 45, 576, 525, 79, 359, 636, 433, 215, 308, 637, 121, 758, 300, 798, 24, 62, 605], 'test': [469, 151, 368, 639, 425, 337, 226, 84, 476, 72, 729, 671, 27, 396, 693, 611, 701, 150, 296, 146, 423, 669, 291, 485, 28, 29, 64, 211, 47, 449, 770, 363, 677, 149, 769, 753, 420, 460, 430, 478, 154, 792, 619, 767, 749, 17, 179, 419, 694, 740, 421, 167, 408, 228, 96, 793, 470, 4, 482, 459, 309, 306, 455, 392, 555, 258, 325, 664, 550, 442, 548, 673, 289, 628, 15, 197, 126, 579, 602, 145, 280, 680, 622, 651, 265, 172, 340, 97, 105, 345, 106, 287, 598, 41, 586, 395, 190, 524, 609, 742, 546, 240, 590, 120, 247, 339, 49, 313, 382, 665, 632, 503, 558, 153, 716, 439, 205, 391, 109, 34, 652, 356, 170, 594, 783, 751, 19, 159, 595, 616, 666, 269, 633, 44, 52, 418, 65, 134, 282, 336, 314, 505, 573, 403, 724, 284, 385, 9, 370, 663, 270, 709, 788, 465, 779, 14, 706, 11, 264, 774]}, 'schema': {'numeric': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11'], 'categorical': ['category', 'band']}, 'plan': {'actions': [{'kind': 'train', 'line': 5, 'model': 'linear'}, {'kind': 'evaluate', 'line': 6, 'model': 'linear'}, {'kind': 'train', 'line': 7, 'model': 'forest'}, {'kind': 'evaluate', 'line': 8, 'model': 'forest'}, {'kind': 'train', 'line': 9, 'model': 'tree'}, {'kind': 'evaluate', 'line': 10, 'model': 'tree'}], 'dataset': 'datasets/mixed.csv', 'drop': [], 'explain': True, 'lines': {'assess': 12, 'evaluate': 6, 'explain': 13, 'predict': 2, 'prepare': 4, 'read': 1, 'select': 11, 'split': 3, 'train': 5}, 'metric': 'accuracy', 'models': [{'algorithm': 'logistic', 'line': 5, 'name': 'linear', 'params': {'C': 1.0}}, {'algorithm': 'forest', 'line': 7, 'name': 'forest', 'params': {'depth': 6, 'trees': 40}}, {'algorithm': 'tree', 'line': 9, 'name': 'tree', 'params': {'depth': 5}}], 'prepare': {'categorical': 'most_frequent', 'numeric': 'median', 'scale': 'standard'}, 'source': 'read "datasets/mixed.csv"\npredict "target" as classification\nsplit stratified train 60% validation 20% test 20% seed 1\nprepare numeric median scale standard categorical most_frequent on train\ntrain linear logistic C=1 on train\nevaluate linear on validation\ntrain forest forest trees=40 depth=6 on train\nevaluate forest on validation\ntrain tree tree depth=5 on train\nevaluate tree on validation\nselect best by accuracy\nassess best on test\nexplain\n', 'source_sha256': '4c3e010eafbb0af3fa6522e23a805efd109a6a0962d2c7f7f86a8176d58a7019', 'split': {'by': None, 'fractions': [0.6, 0.2, 0.2], 'seed': 1, 'strategy': 'stratified'}, 'target': 'target', 'task': 'classification', 'version': 1}}

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
