"""
list_results.py - test (and validation) results of every run group in outputs/ (reads files only).

Usage:  python list_results.py            (SGT_ROOT = folder that contains outputs/)
        python list_results.py bracs      (only groups whose name contains "bracs")
A group is all runs whose names differ only in the final _s<seed>.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(os.environ.get("SGT_ROOT", "."))
filt = sys.argv[1] if len(sys.argv) > 1 else ""
groups = defaultdict(list)
for rep in sorted((root / "outputs").glob("*/report/test_report.json")):
    run = rep.parent.parent.name
    if filt not in run:
        continue
    m = re.match(r"(.*)_s(\d+)$", run)
    g, s = (m.group(1), int(m.group(2))) if m else (run, -1)
    d = json.load(open(rep))
    a = d.get("args") or d.get("config", {}).get("train", {}) or {}
    groups[g].append({"seed": s, "test_wf1": d["test"]["weighted_f1"] * 100, "test_mf1": d["test"]["macro_f1"] * 100,
                      "val_wf1": d.get("validation", {}).get("weighted_f1", float("nan")) * 100,
                      "best_epoch": d.get("best_epoch"), "epochs": a.get("epochs"), "patience": a.get("patience"),
                      "n_params": d.get("n_params"), "time": rep.stat().st_mtime})

print(f"{'group':34s} {'n':>2s} {'test wF1':>14s} {'test mF1':>14s} {'val wF1':>8s} {'epochs/pat':>10s} "
      f"{'best ep':>14s} {'params':>10s}  per-seed test wF1")
for g, rs in sorted(groups.items()):
    rs.sort(key=lambda r: r["seed"])
    w = np.array([r["test_wf1"] for r in rs]); mf = np.array([r["test_mf1"] for r in rs])
    v = np.nanmean([r["val_wf1"] for r in rs])
    sd = lambda x: x.std(ddof=1) if len(x) > 1 else 0.0
    ep = f"{rs[0]['epochs']}/{rs[0]['patience']}"
    be = ",".join(str(r["best_epoch"]) for r in rs)
    print(f"{g:34s} {len(rs):2d} {w.mean():6.2f} ± {sd(w):5.2f} {mf.mean():6.2f} ± {sd(mf):5.2f} {v:8.2f} {ep:>10s} "
          f"{be:>14s} {str(rs[0]['n_params']):>10s}  " + " ".join(f"{x:.1f}" for x in w))
