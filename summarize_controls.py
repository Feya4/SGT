"""
summarize_controls.py - table and statistics for the control experiments (run_controls.py).

For SGT (the reference runs) and every control it reports, over the seeds:
  * test weighted F1, macro F1, accuracy and macro one-vs-rest AUC: mean +/- sample SD
  * trainable parameters
  * difference SGT - control in weighted F1 and macro F1 (percentage points) with two tests:
      seed-paired t-test over the seeds (95% CI from the t distribution, df = seeds - 1)
      paired bootstrap over test samples (resample test graphs with replacement, the difference averaged over
      the seeds in every resample; 95% percentile CI, two-sided p)
    Use the same test as for the SGT-CGT comparison in the manuscript.
It also checks that every control run used exactly the settings of the reference runs (apart from the property
under test) and that all runs were evaluated on the same test samples; any difference is printed as an error.

Usage
  python summarize_controls.py --ref sgt_bracs
Writes outputs/controls_summary_<ref>/{summary.csv, summary.md, summary.json}.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT  # noqa: E402

LABELS = {"sgt": "SGT (reference)", "sgt_rerun": "SGT re-run with archived code", "concat": "Concatenation, parameter-matched (no MASE)",
          "morph_perm": "Morphology permuted within graph", "pos_perm": "Coordinates permuted within graph",
          "morph_none": "Morphology set to zero", "morph_degree": "Degree only (area, convexity = 0)"}
CHANGED = {"sgt_rerun": set(), "concat": {"model.variant", "model.ffn_dim"}, "morph_perm": {"morph_mode"}, "pos_perm": {"pos_mode"},
           "morph_none": {"morph_mode"}, "morph_degree": {"morph_mode"}}
# names of earlier runs of the same controls (used when ctl_<control>_<ref> does not exist)
LEGACY_NAMES = {"morph_perm": "{ref}_morphshuf", "pos_perm": "{ref}_posshuf",
                "morph_none": "{ref}_morphnone", "morph_degree": "{ref}_morphdeg"}
IGNORE = {"exp_name", "seed", "output_dir", "checkpoint_dir"}
METRICS = [("weighted_f1", "wF1"), ("macro_f1", "mF1"), ("accuracy", "Acc"), ("auc_macro_ovr", "AUC")]


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def f1_scores(y, p, C):
    """(weighted F1, macro F1) from integer labels; identical to sklearn with zero_division=0."""
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(C), where=denom > 0)
    present = denom > 0                       # sklearn: labels occurring in y_true or y_pred
    weighted = float((f1 * support).sum() / support.sum())
    macro = float(f1[present].mean())
    return weighted, macro


def load_runs(outputs, name, seeds):
    runs = []
    for s in seeds:
        d = outputs / f"{name}_s{s}" / "report"
        rep, pred = d / "test_report.json", d / "test_predictions.npz"
        if not rep.exists() or not pred.exists():
            return None, f"missing {d}"
        z = np.load(pred, allow_pickle=True)
        runs.append({"seed": s, "report": json.load(open(rep)), "ids": z["sample_id"].astype(str),
                     "y": z["y_true"].astype(int), "pred": z["probs"].argmax(1).astype(int),
                     "C": z["probs"].shape[1]})
    return runs, ""


def t_ci(d, alpha=0.05):
    from scipy import stats
    d = np.asarray(d, float)
    n, m = len(d), float(np.mean(d))
    if n < 2:
        return m, float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    q = stats.t.ppf(1 - alpha / 2, n - 1)
    p = float(stats.ttest_rel(d, np.zeros(n)).pvalue) if se > 0 else 0.0
    return m, m - q * se, m + q * se, p


def bootstrap(ref_runs, ctl_runs, n_boot, seed=0):
    """Paired bootstrap over test samples of the seed-averaged difference ref - control (weighted, macro F1)."""
    rng = np.random.default_rng(seed)
    n, C = len(ref_runs[0]["y"]), ref_runs[0]["C"]
    y = ref_runs[0]["y"]
    obs = np.mean([np.subtract(f1_scores(y, r["pred"], C), f1_scores(y, c["pred"], C))
                   for r, c in zip(ref_runs, ctl_runs)], axis=0)
    diffs = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        diffs[b] = np.mean([np.subtract(f1_scores(yb, r["pred"][idx], C), f1_scores(yb, c["pred"][idx], C))
                            for r, c in zip(ref_runs, ctl_runs)], axis=0)
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    p = np.minimum(1.0, 2 * np.minimum((diffs <= 0).mean(0), (diffs >= 0).mean(0)))
    return obs, lo, hi, p


def settings(report):
    """Flat dict of the run settings: current reports ("config") or sgt_impl13 reports ("args")."""
    if "config" in report:
        return flat(report["config"])
    from utils.checkpoint import config_from_report
    return flat(config_from_report(report).to_dict())


def check_config(ref_runs, runs, ctl):
    a = settings(ref_runs[0]["report"])
    problems = []
    for r in runs:
        b = settings(r["report"])
        allowed = IGNORE | CHANGED.get(ctl, set())
        diff = sorted(k for k in set(a) | set(b) if k not in allowed and a.get(k) != b.get(k))
        # fields absent from pre-revision reports are filled with their legacy values on loading
        diff = [k for k in diff if k in a and k in b]
        if diff:
            problems.append(f"seed {r['seed']}: " + ", ".join(f"{k}: ref={a.get(k)} run={b.get(k)}" for k in diff))
    return problems


def main(a):
    outputs = Path(a.outputs)
    ref_runs, msg = load_runs(outputs, a.ref, a.seeds)
    if ref_runs is None:
        sys.exit(f"reference runs incomplete: {msg}")
    order = ref_runs[0]["ids"]
    base_runs, base_name = ref_runs, "SGT (reference)"
    rows, errors = [], []
    for ctl in ["sgt"] + a.controls:
        name = a.ref if ctl == "sgt" else f"ctl_{ctl}_{a.ref}"
        runs, msg = (ref_runs, "") if ctl == "sgt" else load_runs(outputs, name, a.seeds)
        if runs is None and ctl in LEGACY_NAMES:
            name = LEGACY_NAMES[ctl].format(ref=a.ref)
            runs, msg = load_runs(outputs, name, a.seeds)
        if runs is not None and ctl != "sgt":
            print(f"[{ctl}] using runs {name}_s*")
        if runs is None:
            print(f"[skip] {ctl}: {msg}")
            continue
        for r in runs:                                            # same test samples, same order
            if len(r["ids"]) != len(order) or not np.array_equal(np.sort(r["ids"]), np.sort(order)):
                errors.append(f"{ctl} seed {r['seed']}: different test samples")
            elif not np.array_equal(r["ids"], order):
                pos = {k: i for i, k in enumerate(r["ids"])}
                ix = np.array([pos[k] for k in order])
                r["y"], r["pred"], r["ids"] = r["y"][ix], r["pred"][ix], r["ids"][ix]
        if ctl != "sgt":
            errors += [f"{ctl} {p}" for p in check_config(ref_runs, runs, ctl)]
        if ctl == a.against:
            base_runs, base_name = runs, LABELS[ctl]
        row = {"control": ctl, "label": LABELS[ctl], "n_seeds": len(runs),
               "params": sorted({r["report"].get("n_params") for r in runs})}
        for key, short in METRICS:
            v = np.array([r["report"]["test"][key] for r in runs], float) * 100
            row[f"{short}_mean"], row[f"{short}_sd"] = float(v.mean()), float(v.std(ddof=1))
        if ctl != "sgt" and ctl != a.against:
            row["against"] = base_name
            for key, short in METRICS[:2]:
                d = (np.array([r["report"]["test"][key] for r in base_runs]) -
                     np.array([r["report"]["test"][key] for r in runs])) * 100
                m, lo, hi, p = t_ci(d)
                row.update({f"d{short}_mean": m, f"d{short}_t_lo": lo, f"d{short}_t_hi": hi, f"d{short}_t_p": p})
            obs, lo, hi, p = bootstrap(base_runs, runs, a.boot)
            for i, short in enumerate(["wF1", "mF1"]):
                row.update({f"d{short}_boot_lo": lo[i] * 100, f"d{short}_boot_hi": hi[i] * 100,
                            f"d{short}_boot_p": float(p[i])})
        rows.append(row)

    out = outputs / f"controls_summary_{a.ref}"
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("control", "label", "n_seeds", "params"), k))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump({"ref": a.ref, "seeds": a.seeds, "bootstrap_resamples": a.boot, "rows": rows, "errors": errors},
              open(out / "summary.json", "w"), indent=2, default=str)

    lines = [f"| Model | Params | wF1 (%) | Macro F1 (%) | Acc (%) | AUC (%) | ΔwF1 ({base_name} − row) [95% CI], p (t / bootstrap) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        params = "/".join(f"{p:,}" for p in r["params"] if p)
        cells = [f"{r[s + '_mean']:.2f} ± {r[s + '_sd']:.2f}" for _, s in METRICS]
        if "dwF1_mean" not in r:
            delta = "–"
        else:
            delta = (f"{r['dwF1_mean']:+.2f} [{r['dwF1_t_lo']:.2f}, {r['dwF1_t_hi']:.2f}], p={r['dwF1_t_p']:.3f} / "
                     f"[{r['dwF1_boot_lo']:.2f}, {r['dwF1_boot_hi']:.2f}], p={r['dwF1_boot_p']:.3f}")
        lines.append(f"| {r['label']} | {params} | " + " | ".join(cells) + f" | {delta} |")
    md = "\n".join(lines)
    (out / "summary.md").write_text(md + "\n")
    print(f"\n{a.ref}, seeds {a.seeds}, test split\n\n{md}\n")
    if errors:
        print("ERRORS - the comparison is not valid until these are resolved:")
        for e in errors:
            print("  " + e)
    else:
        print("checks passed: identical settings apart from the tested property; identical test samples")
    print(f"\nsaved {out}/summary.{{md,csv,json}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--against", default="sgt", help="row the differences are computed against, e.g. sgt_rerun")
    p.add_argument("--controls", nargs="+", default=["sgt_rerun", "concat", "morph_perm", "pos_perm"],
                   choices=list(CHANGED), help="add morph_none morph_degree for the other morphology controls")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--outputs", default=f"{ROOT}/outputs")
    p.add_argument("--boot", type=int, default=10000)
    main(p.parse_args())
