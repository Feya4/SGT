"""
profile_graphs.py - graph-size distribution, parameter count, GPU memory and inference latency of SGT.

Measured on the TEST split of the chosen dataset with a trained checkpoint (or random weights with --no_checkpoint;
memory and latency do not depend on the weights).

  * graph size : nodes / edges per graph (mean, SD, median, IQR, min, max), overall and per class
  * latency    : milliseconds per graph, batch size 1, after warm-up, torch.cuda.synchronize() around every call;
                 mean, SD, median, 95th percentile over all test graphs (data already in memory, so loading and
                 feature extraction are NOT included - state this in the paper)
  * throughput : graphs per second at --batch_size (default 16), same protocol
  * memory     : peak GPU memory (torch.cuda.max_memory_allocated) during inference, at batch size 1 (largest graph
                 and mean over graphs) and at --batch_size
  * parameters : trainable parameters

Example
  python profile_graphs.py --dataset bracs --checkpoint $SGT_ROOT/checkpoints/sgt200_bracs_s0/best.pt
  python profile_graphs.py --dataset isic  --checkpoint .../best.pt --set data.isic_image_dir=... (as train.py)
Writes <out>/profile_<dataset>.json and prints a table. Report the GPU model printed in the header; other users'
jobs on the same GPU change latency, so run it when the GPU is idle.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import apply_overrides, get_config
from dataset import build_datasets
from model import build_model
from utils import load_model


def stats(v):
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(med),
            "q1": float(q1), "q3": float(q3), "min": float(v.min()), "max": float(v.max())}


def size_table(ds, class_names):
    n, e, y = [], [], []
    for i in range(len(ds)):
        d = ds[i]
        n.append(d.num_nodes), e.append(d.edge_index.size(1)), y.append(int(d.y))
    n, e, y = np.array(n), np.array(e), np.array(y)
    out = {"all": {"graphs": int(len(n)), "nodes": stats(n), "edges": stats(e)}}
    for c, name in enumerate(class_names):
        m = y == c
        if m.any():
            out[name] = {"graphs": int(m.sum()), "nodes": stats(n[m]), "edges": stats(e[m])}
    return out, n


@torch.no_grad()
def timed(model, batches, device, warmup):
    cuda = device.type == "cuda"
    for b in batches[:warmup]:
        model(b)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for b in batches:
        t0 = time.perf_counter()
        model(b)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


@torch.no_grad()
def peak_mem_mb(model, batch, device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    model(batch)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if a.checkpoint:
        model, cfg, _ = load_model(a.checkpoint, device)       # data settings as used in training
        if cfg.dataset != a.dataset:
            raise ValueError(f"checkpoint is for {cfg.dataset}, not {a.dataset}")
        cfg = apply_overrides(cfg, a.set)
    else:
        cfg = apply_overrides(get_config(a.dataset), a.set)
        model = build_model(cfg.model).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    mem0 = torch.cuda.memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0

    ds = build_datasets(cfg)[2]                                   # test split
    sizes, n_nodes = size_table(ds, cfg.class_names)
    graphs = [ds[i] for i in range(len(ds))]
    single = [Batch.from_data_list([g]).to(device) for g in graphs]
    bs = a.batch_size
    batched = [Batch.from_data_list(graphs[i:i + bs]).to(device) for i in range(0, len(graphs), bs)]

    # latency, batch size 1
    t1 = timed(model, single, device, a.warmup)
    # throughput at batch size bs
    tb = timed(model, batched, device, min(a.warmup, 3))
    thr = len(graphs) / (tb.sum() / 1000.0)

    # peak memory (weights stay allocated; the figure includes them)
    base = torch.cuda.memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    resident_data = (base - mem0) if base is not None else 0.0
    sub = (lambda v: None if v is None else v - resident_data)
    biggest = single[int(np.argmax(n_nodes))]
    mem_big = sub(peak_mem_mb(model, biggest, device))
    mem_mean = float(np.mean([sub(peak_mem_mb(model, b, device)) for b in single[:200]])) if device.type == "cuda" else None
    mem_batch = sub(peak_mem_mb(model, max(batched, key=lambda b: b.x.size(0)), device))

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    res = {"dataset": a.dataset, "device": gpu, "torch": torch.__version__, "python": platform.python_version(),
           "parameters": n_params, "n_test_graphs": len(graphs), "batch_size_throughput": bs,
           "graph_size": sizes,
           "latency_ms_per_graph_bs1": {"mean": float(t1.mean()), "sd": float(t1.std(ddof=1)),
                                        "median": float(np.median(t1)), "p95": float(np.percentile(t1, 95))},
           f"throughput_graphs_per_s_bs{bs}": float(thr),
           "peak_memory_mb": {"weights": weights_mb, "resident_test_data_excluded": resident_data, "bs1_largest_graph": mem_big,
                              "bs1_mean_over_first_200": mem_mean, f"bs{bs}_largest_batch": mem_batch},
           "note": "inference only, data already in memory; excludes data loading and ISIC feature extraction"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"profile_{a.dataset}.json", "w"), indent=2)

    print(f"\n{a.dataset} | {gpu} | torch {torch.__version__} | {n_params:,} parameters | {len(graphs)} test graphs")
    print(f"{'class':6s} {'graphs':>6s} {'nodes mean±SD':>16s} {'median [IQR]':>16s} {'max':>6s} {'edges mean±SD':>16s}")
    for k, v in sizes.items():
        nd, ed = v["nodes"], v["edges"]
        print(f"{k:6s} {v['graphs']:6d} {nd['mean']:7.1f}±{nd['sd']:<7.1f} "
              f"{nd['median']:5.0f} [{nd['q1']:.0f}-{nd['q3']:.0f}] {nd['max']:6.0f} {ed['mean']:7.1f}±{ed['sd']:<7.1f}")
    l = res["latency_ms_per_graph_bs1"]
    print(f"\nlatency (bs=1): {l['mean']:.2f} ± {l['sd']:.2f} ms  (median {l['median']:.2f}, p95 {l['p95']:.2f})")
    print(f"throughput (bs={bs}): {thr:.1f} graphs/s")
    if device.type == "cuda":
        m = res["peak_memory_mb"]
        print(f"peak GPU memory: bs=1 largest graph {m['bs1_largest_graph']:.0f} MB, bs=1 mean {m['bs1_mean_over_first_200']:.0f} MB, "
              f"bs={bs} {m[f'bs{bs}_largest_batch']:.0f} MB (weights alone {m['weights']:.0f} MB; test data on the GPU, {m['resident_test_data_excluded']:.0f} MB, excluded)")
    print(f"\nsaved {out / ('profile_' + a.dataset + '.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["bracs", "isic", "camelyon16"])
    p.add_argument("--checkpoint", default=None, help="best.pt; omit for random weights (same speed and memory)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default="figures/profile")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.isic_image_dir=...")
    main(p.parse_args())