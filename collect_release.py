"""
collect_release.py - gather the per-sample test predictions and reports of the reported runs into results/,
for the Zenodo archive (Editor point 2).

For every run directory outputs/<exp>/ whose name matches one of the given prefixes, it writes
    results/<exp>/test_predictions.csv   sample_id, y_true, y_pred, true_class, pred_class, prob_<class> ...
    results/<exp>/test_report.json       all metrics and every setting of the run
and results/index.csv with one line per run (dataset, seed, test weighted F1, macro F1, accuracy, AUC).
It also copies the split files it finds (BRACS membership, ISIC lesion split) into splits/.

Usage
  python collect_release.py --prefixes sgt_bracs sgt_isic sgt_camelyon16 cgt_bracs cgt_isic ctl_
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, ROOT  # noqa: E402


def export_run(run_dir: Path, dst: Path):
    rep = run_dir / "report" / "test_report.json"
    npz = run_dir / "report" / "test_predictions.npz"
    if not rep.exists() or not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    r = json.load(open(rep))
    classes = ([str(c) for c in z["classes"]] if "classes" in z.files
               else CLASS_NAMES.get(r.get("dataset") or r.get("config", {}).get("dataset", ""), []))
    y_pred = z["y_pred"] if "y_pred" in z.files else z["probs"].argmax(1)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "y_true", "y_pred", "true_class", "pred_class"] + [f"prob_{c}" for c in classes])
        for sid, yt, yp, pr in zip(z["sample_id"], z["y_true"], y_pred, z["probs"]):
            w.writerow([sid, int(yt), int(yp), classes[int(yt)], classes[int(yp)]] + [f"{v:.6f}" for v in pr])
    shutil.copy2(rep, dst / "test_report.json")
    t = r["test"]
    return {"run": run_dir.name, "dataset": r.get("dataset") or r.get("config", {}).get("dataset"),
            "seed": r.get("seed", r.get("config", {}).get("seed")), "n_test": t.get("n"),
            "weighted_f1": t["weighted_f1"], "macro_f1": t["macro_f1"], "accuracy": t["accuracy"],
            "auc_macro_ovr": t["auc_macro_ovr"], "n_params": r.get("n_params")}


def main(a):
    outputs, results = Path(a.outputs), Path(a.results)
    rows = []
    for d in sorted(p for p in outputs.iterdir() if p.is_dir()):
        if any(d.name.startswith(px) for px in a.prefixes):
            row = export_run(d, results / d.name)
            print(f"{'exported' if row else 'skipped (no test report)'}: {d.name}")
            if row:
                rows.append(row)
    if rows:
        with open(results / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    splits = Path(a.splits)
    splits.mkdir(parents=True, exist_ok=True)
    for src in a.split_files:
        if Path(src).exists():
            shutil.copy2(src, splits / Path(src).name)
            print(f"split file copied: {Path(src).name}")
        else:
            print(f"split file NOT found (add it by hand): {src}")
    print(f"\n{len(rows)} runs exported to {results}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", nargs="+", required=True, help="run-name prefixes to export")
    p.add_argument("--outputs", default=f"{ROOT}/outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--splits", default="splits")
    p.add_argument("--split_files", nargs="*",
                   default=["bracs_official_membership.csv", f"{ROOT}/splits/isic2018_lesion_split.csv",
                            f"{ROOT}/splits/isic2018_near_duplicates_cross_lesion.csv"])
    main(p.parse_args())
