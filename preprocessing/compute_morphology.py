"""
compute_morphology.py
=====================
Add REAL morphological attributes to the clean BRACS graphs.

For every graph in <graph_root>/{train,val,test}_official/ it
  1. reproduces the tissue-region map with the original pipeline
     (Vahadane normalisation with target.png + ColorMergedSuperpixelExtractor,
     exactly as in build_tg.py; verified to reproduce the stored graphs),
  2. measures per region: area (pixels at full resolution) and solidity
     (area / convex-hull area) with skimage.measure.regionprops,
  3. checks that regions and graph nodes correspond one-to-one
     (same count and matching centroids),
  4. writes a copy of the graph to <out_root>/<split>_official/ with
     ndata['area'] and ndata['solidity'] added (features unchanged).

Graphs whose regions cannot be aligned with the nodes are NOT written and are
listed in morphology_problems.txt.  Already-written outputs are skipped, so the
script can be re-run (e.g. after the missing test graphs are built).

Requires the stain-normalisation target at assets/target.png.

Usage
  python preprocessing/compute_morphology.py --check 5      # alignment check on 5 graphs
  python preprocessing/compute_morphology.py --workers 8    # all splits, 8 CPU processes
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only

import numpy as np
import torch
from PIL import Image

G = os.environ.get("SGT_ROOT", ".")
TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "target.png")
_W = {}


def _init_worker():
    from histocartography.preprocessing import (
        VahadaneStainNormalizer, ColorMergedSuperpixelExtractor)
    _W["norm"] = VahadaneStainNormalizer(target_path=TARGET)
    _W["sp"] = ColorMergedSuperpixelExtractor(
        superpixel_size=500, compactness=20, blur_kernel_size=1,
        threshold=0.05, downsampling_factor=4)


def _region_props(label_map):
    from skimage.measure import regionprops
    props = regionprops(label_map)                    # sorted by label, background 0 ignored
    labels = np.array([p.label for p in props])
    area = np.array([p.area for p in props], dtype=np.float32)
    solidity = np.array([p.solidity for p in props], dtype=np.float32)
    cent_rc = np.array([p.centroid for p in props], dtype=np.float32)   # (row, col)
    return labels, area, solidity, cent_rc


def _alignment(g, cent_rc):
    """Max centroid distance (pixels) between graph nodes and regions, trying (x,y) and (row,col)."""
    if "centroid" not in g.ndata:
        return float("nan")
    gc = g.ndata["centroid"].numpy().astype(np.float32)
    d_xy = np.abs(gc - cent_rc[:, ::-1]).max()
    d_rc = np.abs(gc - cent_rc).max()
    return float(min(d_xy, d_rc))


def process(job):
    from dgl.data.utils import load_graphs, save_graphs
    src, png, dst, tol = job
    if Path(dst).exists():
        return ("skip", src, "")
    try:
        gs, labels = load_graphs(src)
        g = gs[0]
        fdim = g.ndata["feat"].shape[1]
        if fdim != 512:
            return ("problem", src, f"feature dimension {fdim} != 512 (location features not removed?)")
        image = np.array(Image.open(png).convert("RGB"))
        note = ""
        try:
            image = _W["norm"].process(image)
        except Exception as e:
            note = f"stain normalisation failed ({e}); unnormalised image used as in build_tg.py"
        label_map, _ = _W["sp"].process(image)
        lab, area, solidity, cent = _region_props(np.asarray(label_map))
        if len(lab) != g.num_nodes():
            return ("problem", src, f"regions {len(lab)} != nodes {g.num_nodes()}")
        dmax = _alignment(g, cent)
        if not (dmax <= tol):
            return ("problem", src, f"centroid mismatch, max distance {dmax:.2f} px")
        g.ndata["area"] = torch.from_numpy(area)
        g.ndata["solidity"] = torch.from_numpy(solidity)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        save_graphs(dst, [g], labels) if labels else save_graphs(dst, [g])
        return ("ok", src, note)
    except Exception as e:
        return ("problem", src, f"error: {e}")


def main(a):
    png_index = {p.stem: str(p) for p in Path(a.png_root).rglob("BRACS_*.png")}
    jobs = []
    for split in a.splits:
        for f in sorted((Path(a.graph_root) / f"{split}_official").glob("BRACS_*.bin")):
            if f.stem not in png_index:
                print(f"[WARN] no image for {f.name}")
                continue
            dst = Path(a.out_root) / f"{split}_official" / f.name
            jobs.append((str(f), png_index[f.stem], str(dst), a.tol))
    if a.check:
        jobs = jobs[:a.check]
    print(f"{len(jobs)} graphs to process with {a.workers} worker(s)")

    counts = {"ok": 0, "skip": 0, "problem": 0}
    problems, notes = [], []
    with Pool(a.workers, initializer=_init_worker) as pool:
        for i, (status, src, msg) in enumerate(pool.imap_unordered(process, jobs), 1):
            counts[status] += 1
            if status == "problem":
                problems.append(f"{Path(src).name}\t{msg}")
            elif msg:
                notes.append(f"{Path(src).name}\t{msg}")
            if a.check:
                print(f"  {Path(src).name:30s} {status:8s} {msg}")
            elif i % 100 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] ok={counts['ok']} skipped={counts['skip']} "
                      f"problems={counts['problem']}", flush=True)
    Path("morphology_problems.txt").write_text("\n".join(problems) + "\n")
    Path("morphology_notes.txt").write_text("\n".join(notes) + "\n")
    print(f"\nDone: {counts}. See morphology_problems.txt and morphology_notes.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph_root", default=f"{G}/data/tissue_graphs_official")
    p.add_argument("--out_root", default=f"{G}/data/tissue_graphs_official_morph")
    p.add_argument("--png_root", default=f"{G}/data")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tol", type=float, default=2.0, help="max centroid distance in pixels")
    p.add_argument("--check", type=int, default=0)
    main(p.parse_args())
