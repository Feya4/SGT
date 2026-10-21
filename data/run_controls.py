"""
run_controls.py - the control experiments requested in the minor revision (Editor point 1), on BRACS.

Every control is trained with EXACTLY the settings of the reference SGT runs (the Table 7 runs), read from their
saved reports, and with the same seeds and the same checkpoint-selection rule (best validation weighted F1,
test split evaluated once). Only the one property under test changes:

  concat      morphological attributes [degree, area, convexity] concatenated to the node features; no MASE
              stream; the FFN is widened so the parameter count matches SGT (model.variant=concat). SRE kept.
  morph_perm  morphological attributes permuted across nodes within each graph, at training and test time
  pos_perm    node coordinates permuted across nodes within each graph, at training and test time
              (coordinates are used only by the SRE, so this removes the spatial information of the bias)
  sgt_rerun   SGT itself re-trained with this code and the reference settings. Run it together with concat
              when the reference runs were made with other code (e.g. sgt_impl13): concat is then compared
              with SGT trained by the same code, and the re-run shows whether this code reproduces the reference.

Usage (from the repository root; one short command, runs everything in sequence, skips finished runs)
  python run_controls.py --ref sgt_bracs
  python run_controls.py --ref sgt_bracs --controls concat --seeds 0 1      # a subset
  python run_controls.py --ref sgt_bracs --dry_run                          # show what would run
  python run_controls.py --ref sgt_bracs --wait_mb 4000                     # start when a GPU has 4 GB free
  bash start_controls.sh sgt_bracs                                          # same, in the background

--ref is the experiment-name prefix of the reference runs: outputs/<ref>_s<seed>/report/test_report.json must
exist for every seed. Results: outputs/ctl_<control>_<ref>_s<seed>/. Then: python summarize_controls.py --ref ...
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, apply_overrides  # noqa: E402
from utils.checkpoint import config_from_report  # noqa: E402

CONTROLS = {
    "concat": {"model.variant": "concat"},
    "morph_perm": {"morph_mode": "shuffle"},
    "pos_perm": {"pos_mode": "shuffle"},
    "sgt_rerun": {},          # SGT itself, re-trained with this code: checks that it reproduces the reference
}
IGNORE = {"exp_name", "seed"}
# names used by earlier runs of the same controls (accepted instead of ctl_<control>_<ref>)
LEGACY_NAMES = {"morph_perm": "{ref}_morphshuf", "pos_perm": "{ref}_posshuf"}


def existing_run(outputs: Path, ctl: str, ref: str, seed: int):
    """Name of a finished run of this control and seed (new or older naming), or None."""
    names = [f"ctl_{ctl}_{ref}"] + ([LEGACY_NAMES[ctl].format(ref=ref)] if ctl in LEGACY_NAMES else [])
    for n in names:
        if (outputs / f"{n}_s{seed}" / "report" / "test_report.json").exists():
            return f"{n}_s{seed}"
    return None


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def load_reference(outputs: Path, ref: str, seeds):
    """Reference config for every seed, checked to be identical apart from seed and exp_name."""
    cfgs = {}
    for s in seeds:
        rep = outputs / f"{ref}_s{s}" / "report" / "test_report.json"
        if not rep.exists():
            raise FileNotFoundError(f"reference report not found: {rep}")
        saved = json.load(open(rep))
        if saved.get("controls", {}) not in ({}, {"morph_mode": "real", "pos_mode": "real"}):
            raise ValueError(f"{rep}: the reference run is itself a control run")
        cfgs[s] = config_from_report(saved)
    first = flat(cfgs[seeds[0]].to_dict())
    for s in seeds[1:]:
        other = flat(cfgs[s].to_dict())
        diff = [k for k in first if k not in IGNORE and first[k] != other.get(k)]
        if diff:
            raise ValueError(f"reference runs differ in more than the seed (seed {seeds[0]} vs {s}): {diff}")
    return cfgs


def free_gpu_mb():
    """{gpu index: free memory in MB} from nvidia-smi ({} if unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines() if "," in line)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def wait_for_gpu(gpu, need_mb, poll_s):
    """Block until the chosen GPU (or, with gpu='auto', any GPU) has need_mb free; return its index."""
    t0 = time.time()
    while True:
        free = free_gpu_mb()
        if not free:
            raise RuntimeError("nvidia-smi not available; run without --wait_mb")
        cand = free if gpu == "auto" else {int(gpu): free.get(int(gpu), 0)}
        best = max(cand, key=cand.get)
        if cand[best] >= need_mb:
            print(f"GPU {best} has {cand[best]} MB free - starting "
                  f"(waited {(time.time() - t0) / 60:.0f} min)", flush=True)
            return best
        print(f"[{time.strftime('%H:%M')}] waiting: free MB per GPU {free}, need {need_mb}", flush=True)
        time.sleep(poll_s)


def main(a):
    outputs = Path(a.outputs)
    cfgs = load_reference(outputs, a.ref, a.seeds)
    c0 = cfgs[a.seeds[0]]
    print(f"reference: {a.ref}_s{{{','.join(map(str, a.seeds))}}} | dataset={c0.dataset} | epochs={c0.train.epochs} "
          f"patience={c0.train.patience} | fuse_prev_morph={c0.model.fuse_prev_morph} "
          f"drop_path_per_graph={c0.model.drop_path_per_graph} | bracs_pos={c0.data.bracs_pos} "
          f"bracs_area={c0.data.bracs_area}")
    if c0.dataset == "isic":
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    jobs = []
    for ctl in a.controls:
        for s in a.seeds:
            found = existing_run(outputs, ctl, a.ref, s)
            jobs.append((ctl, s, found or f"ctl_{ctl}_{a.ref}_s{s}", found is not None))
    for ctl, s, name, done in jobs:
        print(f"  {'done   ' if done else 'to run '} {name}")
    print(f"{sum(not d for *_, d in jobs)} run(s) to do")
    if a.dry_run:
        return
    if not any(not done for *_, done in jobs):
        print("all runs finished")
        return

    # choose the device BEFORE torch is imported
    if a.gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif a.wait_mb > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu(a.gpu, a.wait_mb, a.poll))
    elif a.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    import train                                                   # imported late: loads torch
    for ctl, s, name, done in jobs:
        if done:
            continue
        cfg = copy.deepcopy(cfgs[s])
        cfg.exp_name, cfg.output_dir = name, str(outputs)
        if a.checkpoints:
            cfg.checkpoint_dir = a.checkpoints
        apply_overrides(cfg, [f"{k}={v}" for k, v in CONTROLS[ctl].items()] + list(a.set))
        print(f"\n===== {name} =====", flush=True)
        train.main(cfg, resume=a.resume)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the control experiments of the revision")
    p.add_argument("--ref", required=True, help="exp-name prefix of the reference SGT runs, e.g. sgt_bracs")
    p.add_argument("--controls", nargs="+", default=["sgt_rerun", "concat", "morph_perm", "pos_perm"],
                   choices=list(CONTROLS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--outputs", default=f"{ROOT}/outputs")
    p.add_argument("--checkpoints", default=None, help="default: checkpoint_dir of the reference config")
    p.add_argument("--set", nargs="*", default=[], help="extra overrides applied to every control run")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run from latest.pt")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--gpu", default="auto", help="GPU index, 'auto' (any GPU) or 'cpu'")
    p.add_argument("--wait_mb", type=int, default=0, help="wait until the GPU has this much free memory (MB)")
    p.add_argument("--poll", type=int, default=300, help="seconds between GPU checks while waiting")
    main(p.parse_args())
