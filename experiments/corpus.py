"""Constructed conformance corpus, not a sampled corpus of real developer bugs."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def make_cases():
    cases = []
    for path in sorted((ROOT/"examples").glob("*.tsml")):
        source = path.read_text(); stem = path.stem
        for imputer in ["median", "mean", "most_frequent"]:
            for scale in ["standard", "none"]:
                changed = source.replace("numeric median", f"numeric {imputer}").replace("scale standard", f"scale {scale}")
                cases.append({"id": f"valid_{stem}_{imputer}_{scale}", "valid": True, "family": "valid_configuration", "source": changed})
        lines = source.splitlines(keepends=True)
        selection = next(line for line in lines if line.startswith("select "))
        wrong_algorithm = "ridge" if "as classification" in source else "logistic"
        correct_algorithm = "logistic" if "as classification" in source else "ridge"
        metric = "rmse" if "as classification" in source else "accuracy"
        mutants = {
            "unknown_task": source.replace("as classification", "as unknown").replace("as regression", "as unknown"),
            "preprocess_before_split": "".join(lines[:2]+[lines[3], lines[2]]+lines[4:]),
            "repeated_split": "".join(lines[:3]+[lines[2]]+lines[3:]),
            "repeated_prepare": "".join(lines[:4]+[lines[3]]+lines[4:]),
            "fit_validation": source.replace("C=1 on train", "C=1 on validation").replace("alpha=1 on train", "alpha=1 on validation"),
            "prepare_all": source.replace("categorical most_frequent on train", "categorical most_frequent on all"),
            "evaluate_test": source.replace("evaluate linear on validation", "evaluate linear on test"),
            "assess_validation": source.replace("assess best on test", "assess best on validation"),
            "select_too_early": source.replace("evaluate linear on validation\n", selection, 1),
            "undefined_candidate": source.replace("evaluate linear", "evaluate undefined"),
            "duplicate_candidate": source.replace("train tree tree", "train linear tree"),
            "post_assessment_fit": source.replace("explain\n", "train late tree on train\n"),
            "bad_percentages": source.replace("train 60%", "train 61%"),
            "bad_seed": source.replace("seed 42", "seed 4294967296"),
            "unsupported_algorithm": source.replace(f"linear {correct_algorithm}", "linear neural_network"),
            "wrong_task_algorithm": source.replace(f"linear {correct_algorithm}", f"linear {wrong_algorithm}"),
            "unknown_parameter": source.replace("trees=40", "jobs=40"),
            "bad_parameter_range": source.replace("trees=40", "trees=0"),
            "duplicate_parameter": source.replace("trees=40", "trees=40 trees=50"),
            "missing_assessment": source.replace("assess best on test\n", "").replace("explain\n", ""),
            "wrong_metric": source.replace(selection, f"select best by {metric}\n"),
            "drop_target": "".join(lines[:2]+['drop "target"\n']+lines[2:]),
            "second_assessment": source.replace("explain\n", "assess best on test\n"),
            "arbitrary_python": source + "import os\n",
        }
        for family, text in mutants.items():
            if text == source: raise AssertionError(f"Ineffective mutation: {family}")
            cases.append({"id": f"invalid_{stem}_{family}", "valid": False, "family": family, "source": text})
    return cases

def write_corpus():
    out = ROOT/"experiments"/"corpus"; out.mkdir(exist_ok=True)
    manifest = []
    for case in make_cases():
        path = out/(case["id"]+".tsml"); path.write_text(case["source"])
        manifest.append({k:v for k,v in case.items() if k != "source"})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest

if __name__ == "__main__":
    manifest = write_corpus(); print(f"{sum(c['valid'] for c in manifest)} valid, {sum(not c['valid'] for c in manifest)} invalid")
