"""
build_bracs_graphs.py
=====================
Build BRACS tissue graphs for the ROIs listed in missing_graphs_<split>.txt
(written by organize_bracs_official.py), with
EXACTLY the pipeline of CGT/core/build_tg.py, then keep the first 512 feature
columns (ResNet-34 features; the 2 appended location features are dropped),
matching the existing *_fixed_512 graphs.

Pipeline (unchanged from build_tg.py):
  Vahadane stain normalisation (target.png)
  ColorMergedSuperpixelExtractor(superpixel_size=500, compactness=20,
      blur_kernel_size=1, threshold=0.05, downsampling_factor=4)
  DeepFeatureExtractor('resnet34', patch_size=144, resize_size=224)
  RAGGraphBuilder(add_loc_feats=True)  -> 514 dims -> first 512 kept

Fixes relative to build_tg.py:
  * the size check was inverted (only invalid-size images were processed);
    here EVERY listed image is processed and its size is logged
  * stain-normalisation failures are recorded, not silently ignored

Modes
  --verify N : rebuild N ROIs that already have graphs (from --verify_dir) and
               compare node count, edges and features with the stored graphs.
               Run this FIRST to confirm the environment reproduces the old graphs.
  (default)  : build all ROIs in --list into --out_dir.

Usage
  python preprocessing/build_bracs_graphs.py --verify 5          # check reproducibility first
  python preprocessing/build_bracs_graphs.py --split test
  python preprocessing/build_bracs_graphs.py --split train       # if missing_graphs_train.txt is not empty

Afterwards run compute_morphology.py again: new graphs have no area/solidity yet.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dgl.data.utils import load_graphs, save_graphs
from histocartography.preprocessing import (
    VahadaneStainNormalizer, ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor, RAGGraphBuilder,
)

G = os.environ.get("SGT_ROOT", ".")
LABEL = {"N": 0, "PB": 1, "UDH": 2, "ADH": 3, "FEA": 4, "DCIS": 5, "IC": 6}
MIN_NR_PIXELS, MAX_NR_PIXELS = 50000, 50000000   # only logged, never used to skip


class Builder:
    def __init__(self, target):
        if not Path(target).exists():
            raise FileNotFoundError(f"Stain-normalisation target not found: {target}")
        self.normalizer = VahadaneStainNormalizer(target_path=target)
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500, compactness=20, blur_kernel_size=1,
            threshold=0.05, downsampling_factor=4)
        self.feature_extractor = DeepFeatureExtractor(
            architecture="resnet34", patch_size=144, resize_size=224)
        self.rag = RAGGraphBuilder(add_loc_feats=True)

    def build(self, png):
        image = np.array(Image.open(png).convert("RGB"))
        npx = image.shape[0] * image.shape[1]
        note = "" if MIN_NR_PIXELS < npx < MAX_NR_PIXELS else f"size {npx} outside original range"
        try:
            image = self.normalizer.process(image)
        except Exception as e:  # recorded, image used unnormalised (as in original)
            note = (note + "; " if note else "") + f"stain normalisation failed: {e}"
        superpixels, _ = self.tissue_detector.process(image)
        feats = self.feature_extractor.process(image, superpixels)
        g = self.rag.process(superpixels, feats)
        g.ndata["feat"] = g.ndata["feat"][:, :512].contiguous()
        return g, note


def label_of(png):
    return LABEL[Path(png).stem.split("_")[2]]


def verify(b, a):
    stored = sorted(Path(a.verify_dir).glob("BRACS_*.bin"))
    random.Random(0).shuffle(stored)
    png_index = {p.stem: p for p in Path(a.png_root).rglob("BRACS_*.png")}
    done = 0
    for f in stored:
        if f.stem not in png_index:
            continue
        g_old = load_graphs(str(f))[0][0]
        g_new, note = b.build(png_index[f.stem])
        same_n = g_old.num_nodes() == g_new.num_nodes()
        same_e = same_n and g_old.num_edges() == g_new.num_edges()
        diff = ((g_old.ndata["feat"] - g_new.ndata["feat"]).abs().max().item()
                if same_n else float("nan"))
        print(f"{f.stem:28s} nodes {g_old.num_nodes():4d}/{g_new.num_nodes():4d} "
              f"edges {g_old.num_edges():5d}/{g_new.num_edges():5d} "
              f"same_structure={same_e} max|feat diff|={diff:.2e} {note}")
        done += 1
        if done >= a.verify:
            break


def build_all(b, a):
    pngs = [l.strip() for l in open(a.list) if l.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, ok = [], 0
    for i, png in enumerate(pngs, 1):
        dst = out / (Path(png).stem + ".bin")
        if dst.exists():
            ok += 1
            continue
        try:
            g, note = b.build(png)
            save_graphs(str(dst), [g], {"label": torch.tensor([label_of(png)])})
            ok += 1
            if note:
                log.append(f"{Path(png).stem}\t{note}")
        except Exception as e:
            log.append(f"{Path(png).stem}\tFAILED: {e}")
        if i % 20 == 0 or i == len(pngs):
            print(f"[{i}/{len(pngs)}] built or present: {ok}")
    log_file = f"build_bracs_graphs_{a.split}_log.txt"
    Path(log_file).write_text("\n".join(log) + "\n")
    print(f"\nDone: {ok}/{len(pngs)} graphs present in {out}")
    print(f"Notes/failures: {len(log)} (see {log_file})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--list", default=None, help="default: missing_graphs_<split>.txt")
    p.add_argument("--out_dir", default=None, help="default: data/tissue_graphs_official/<split>_official")
    p.add_argument("--target", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "target.png"),
                   help="stain-normalisation target used by build_tg.py (../data/target.png)")
    p.add_argument("--png_root", default=f"{G}/data")
    p.add_argument("--verify", type=int, default=0)
    p.add_argument("--verify_dir", default=f"{G}/data/tissue_graphs_official/val_official")
    a = p.parse_args()
    a.list = a.list or f"missing_graphs_{a.split}.txt"
    a.out_dir = a.out_dir or f"{G}/data/tissue_graphs_official/{a.split}_official"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    b = Builder(a.target)
    verify(b, a) if a.verify else build_all(b, a)
