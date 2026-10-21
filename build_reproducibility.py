"""
build_reproducibility.py - build the reproducibility/ folder for the Zenodo archive (Editor point 2):

reproducibility/
├── README.md                      what every file contains (written by this script)
├── SHA256SUMS                     checksums of every file below
├── splits/
│   ├── bracs/bracs_split.csv      roi, slide, class, split   (from the graph folders actually used for training)
│   ├── camelyon16/...             slide lists (copied from --camelyon_split, if given)
│   └── isic2018/...               lesion-level split + near-duplicate pairs (copied, if found)
└── predictions/
    ├── bracs/<run>/test_predictions.csv + test_report.json, and bracs/index.csv (one line per run)
    ├── camelyon16/...
    └── isic2018/...

Reads files only; writes nothing outside --out. Re-running replaces the folder's contents.

Usage (from the repository root, cgt environment)
  export SGT_ROOT=/workdir1.8t/fei27/CGT/GNNMD/sgt_impl13
  python build_reproducibility.py --runs "sgt200_bracs*" "ctl_*sgt200_bracs*"
  python build_reproducibility.py --runs "sgt200_bracs*" "ctl_*" "cgt_bracs*" "sgt_isic*" \\
         --outputs $SGT_ROOT/outputs /path/to/other/outputs --camelyon_split /path/to/camelyon16_split.csv
--runs are shell-style patterns of run folder names (quote them). Only runs with a test report are exported.
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, ROOT  # noqa: E402

TABLE1 = {"bracs": {"train": (3155, 193), "val": (602, 68), "test": (626, 59)}}   # (ROIs, slides) in Table 1


def dataset_of(report: dict, run: str) -> str:
    d = str(report.get("dataset") or report.get("config", {}).get("dataset") or run).lower()
    return "isic2018" if "isic" in d else "camelyon16" if "camelyon" in d else "bracs"


# ── splits ───────────────────────────────────────────────────────────────────

def bracs_split(graph_root: Path, dst: Path):
    rows = []
    for split in ("train", "val", "test"):
        for f in sorted((graph_root / f"{split}_official").glob("BRACS_*.bin")):
            parts = f.stem.split("_")                    # BRACS_<slide>_<class>_<n>
            rows.append({"roi": f.stem, "slide": "_".join(parts[:2]), "class": parts[2], "split": split})
    if not rows:
        print(f"  [bracs] no graphs under {graph_root} - split file not written")
        return
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "bracs_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["roi", "slide", "class", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [bracs] bracs_split.csv: {len(rows)} ROIs")
    slides = {s: {r["slide"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        n, k = sum(r["split"] == s for r in rows), len(slides[s])
        exp = TABLE1["bracs"][s]
        flag = "" if (n, k) == exp else f"   <-- Table 1 says {exp[0]} ROIs / {exp[1]} slides"
        cls = Counter(r["class"] for r in rows if r["split"] == s)
        print(f"      {s:5s} {n:5d} ROIs {k:4d} slides  " + " ".join(f"{c}={cls.get(c, 0)}" for c in CLASS_NAMES["bracs"]) + flag)
    shared = [(a, b, len(slides[a] & slides[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))]
    for a, b, n in shared:
        print(f"      slides shared {a}-{b}: {n}" + ("" if n == 0 else "   <-- LEAKAGE"))
    rois = Counter(r["roi"] for r in rows)
    dup = [r for r, c in rois.items() if c > 1]
    if dup:
        print(f"      ROIs in more than one split: {len(dup)}   <-- LEAKAGE, e.g. {dup[:3]}")


def copy_files(paths, dst: Path, label: str, flag: str):
    found = [Path(p) for p in paths if p and Path(p).exists()]
    for p in found:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / p.name)
        print(f"  [{label}] copied {p.name}")
    if not found:
        print(f"  [{label}] no split file found - add it with {flag} /path/to/file.csv")


# ── predictions ──────────────────────────────────────────────────────────────

def export_run(run_dir: Path, dst_root: Path):
    rep, npz = run_dir / "report" / "test_report.json", run_dir / "report" / "test_predictions.npz"
    if not rep.exists() or not npz.exists():
        return None
    r = json.load(open(rep))
    ds = dataset_of(r, run_dir.name)
    z = np.load(npz, allow_pickle=True)
    key = {"bracs": "bracs", "isic2018": "isic", "camelyon16": "camelyon16"}[ds]
    classes = [str(c) for c in z["classes"]] if "classes" in z.files else CLASS_NAMES[key]
    pred = z["y_pred"] if "y_pred" in z.files else z["probs"].argmax(1)
    dst = dst_root / ds / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "y_true", "y_pred", "true_class", "pred_class"] + [f"prob_{c}" for c in classes])
        for sid, yt, yp, pr in zip(z["sample_id"], z["y_true"], pred, z["probs"]):
            w.writerow([sid, int(yt), int(yp), classes[int(yt)], classes[int(yp)]] + [f"{v:.6f}" for v in pr])
    shutil.copy2(rep, dst / "test_report.json")
    m = re.match(r"(.*)_s(\d+)$", run_dir.name)
    t = r["test"]
    return ds, {"run": run_dir.name, "group": m.group(1) if m else run_dir.name,
                "seed": int(m.group(2)) if m else "", "n_test": len(z["y_true"]),
                "weighted_f1": round(t["weighted_f1"], 6), "macro_f1": round(t["macro_f1"], 6),
                "accuracy": round(t["accuracy"], 6), "auc_macro_ovr": round(t.get("auc_macro_ovr", float("nan")), 6),
                "n_params": r.get("n_params", ""), "best_epoch": r.get("best_epoch", "")}


README = """# Reproducibility files

Split files and per-sample test predictions for every run reported in the manuscript
"Structural-Morphology-Aware Graph Transformer for Fine-Grained Digital Pathology Image Classification".
Image data are not included; obtain them from the original providers (see the main README).

## splits/
- `bracs/bracs_split.csv`: one row per ROI (`roi`, `slide` = WSI identifier, `class`, `split`), read from the graph
  folders used for training. Official BRACS partition; no WSI occurs in more than one split.
- `isic2018/`: lesion-level split (`image`, `lesion_id`, `group`, `label`, `class`, `split`); `group` merges
  near-duplicate images (perceptual hash, Hamming distance <= 2), listed in the near-duplicate file.
- `camelyon16/`: training/validation slide lists drawn from the official training set; test = official test set.

## predictions/<dataset>/
- `index.csv`: one row per run: run name, group (configuration), seed, test metrics, parameters, best epoch.
- `<run>/test_predictions.csv`: one row per test sample: `sample_id`, `y_true`, `y_pred` (class indices),
  `true_class`, `pred_class`, `prob_<class>` (softmax probabilities).
- `<run>/test_report.json`: all metrics (overall and per class, confusion matrix) and every setting of the run.
Run groups: `sgt200_bracs` = SGT; `*_morphshuf` / `*_posshuf` = morphology / coordinates permuted within each graph;
`*_morphnone` = morphology set to zero; `*_morphdeg` = degree only; `ctl_concat_*` = parameter-matched concatenation;
`ctl_sgt_rerun_*` = SGT re-trained with the archived code. Seeds 0-4.

`SHA256SUMS` lists checksums of all files (`sha256sum -c SHA256SUMS` from this folder).
"""


def main(a):
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "splits").mkdir(parents=True)
    print("splits:")
    bracs_split(Path(a.bracs_graphs), out / "splits" / "bracs")
    copy_files(a.isic_split, out / "splits" / "isic2018", "isic2018", "--isic_split")
    copy_files(a.camelyon_split, out / "splits" / "camelyon16", "camelyon16", "--camelyon_split")

    print("predictions:")
    index = {}
    for od in a.outputs:
        for d in sorted(p for p in Path(od).iterdir() if p.is_dir()):
            if not any(fnmatch.fnmatch(d.name, pat) for pat in a.runs):
                continue
            res = export_run(d, out / "predictions")
            if res:
                index.setdefault(res[0], []).append(res[1])
    for ds, rows in index.items():
        with open(out / "predictions" / ds / "index.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for g, n in sorted(Counter(r["group"] for r in rows).items()):
            print(f"  [{ds}] {g}: {n} run(s)")
    (out / "README.md").write_text(README)
    with open(out / "SHA256SUMS", "w") as fh:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n")
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"\n{out}: {n_files} files, {sum(len(v) for v in index.values())} runs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["sgt200_bracs*", "ctl_*sgt200_bracs*"],
                   help="run folder name patterns to export (quote them)")
    p.add_argument("--outputs", nargs="+", default=[f"{ROOT}/outputs"])
    p.add_argument("--bracs_graphs", default="/workdir1.8t/fei27/CGT/GNNMD/data/tissue_graphs_official_morph")
    p.add_argument("--isic_split", nargs="*", default=[f"{ROOT}/splits/isic2018_lesion_split.csv",
                                                       f"{ROOT}/splits/isic2018_near_duplicates_cross_lesion.csv"])
    p.add_argument("--camelyon_split", nargs="*", default=[])
    p.add_argument("--out", default="reproducibility")
    main(p.parse_args())
