"""
build_bach_tg.py
================
Build tissue graphs for the 400 labelled BACH microscopy images with EXACTLY the
BRACS pipeline, so that BRACS-trained models can be applied without retraining.

  1. image rescaled from BACH resolution (0.42 um/px) to BRACS resolution
     (0.25 um/px): factor 0.42/0.25 = 1.68 (2048x1536 -> 3441x2580)
  2. Vahadane stain normalisation to the SAME target.png as BRACS
  3. ColorMergedSuperpixelExtractor(500, 20, blur 1, threshold 0.05, downsampling 4)
  4. ResNet-34 features (patch 144 -> 224); first 512 columns kept
  5. region adjacency graph
  6. real morphology per region: area (pixels at BRACS resolution), solidity

Output: <out_dir>/BACH_<Class>_<image>.bin with ndata feat, centroid, area,
solidity and label 0=Normal 1=Benign 2=InSitu 3=Invasive.

Parallel use: run several processes with --part i --nparts k (existing outputs
are skipped, so the script can be restarted safely).
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dgl.data.utils import save_graphs
from skimage.measure import regionprops
from histocartography.preprocessing import (
    VahadaneStainNormalizer, ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor, RAGGraphBuilder,
)

BACH_CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
KEYWORDS = {"normal": 0, "benign": 1, "situ": 2, "invasive": 3}
SCALE = 0.42 / 0.25


def class_of(path: Path):
    for part in reversed(path.parts[:-1]):
        p = part.lower()
        for k, v in KEYWORDS.items():
            if k in p:
                return v
    return None


def main(a):
    imgs = sorted(p for p in Path(a.bach_dir).rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
    items = [(p, class_of(p)) for p in imgs]
    items = [(p, c) for p, c in items if c is not None]
    counts = np.bincount([c for _, c in items], minlength=4)
    print(f"found {len(items)} labelled images: " +
          "  ".join(f"{n}={k}" for n, k in zip(BACH_CLASSES, counts)))
    items = items[a.part::a.nparts]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    norm = VahadaneStainNormalizer(target_path=a.target)
    sp = ColorMergedSuperpixelExtractor(superpixel_size=500, compactness=20, blur_kernel_size=1,
                                        threshold=0.05, downsampling_factor=4)
    fe = DeepFeatureExtractor(architecture="resnet34", patch_size=144, resize_size=224)
    rag = RAGGraphBuilder(add_loc_feats=True)

    log = []
    for i, (p, c) in enumerate(items, 1):
        dst = out / f"BACH_{BACH_CLASSES[c]}_{p.stem}.bin"
        if dst.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            img = np.array(img.resize((round(w * SCALE), round(h * SCALE)), Image.BICUBIC))
            try:
                img = norm.process(img)
            except Exception as e:
                log.append(f"{p.name}\tstain normalisation failed: {e}")
            label_map, _ = sp.process(img)
            label_map = np.asarray(label_map)
            g = rag.process(label_map, fe.process(img, label_map))
            g.ndata["feat"] = g.ndata["feat"][:, :512].contiguous()
            props = regionprops(label_map)
            if len(props) != g.num_nodes():
                log.append(f"{p.name}\tFAILED: regions {len(props)} != nodes {g.num_nodes()}")
                continue
            g.ndata["area"] = torch.tensor([q.area for q in props], dtype=torch.float32)
            g.ndata["solidity"] = torch.tensor([q.solidity for q in props], dtype=torch.float32)
            save_graphs(str(dst), [g], {"label": torch.tensor([c])})
        except Exception as e:
            log.append(f"{p.name}\tFAILED: {e}")
        if i % 10 == 0 or i == len(items):
            print(f"[part {a.part}] {i}/{len(items)}", flush=True)
    Path(f"build_bach_log_part{a.part}.txt").write_text("\n".join(log) + "\n")
    print(f"done; notes/failures: {len(log)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bach_dir", default=os.path.join(os.environ.get("SGT_ROOT", "."), "BACH", "BACH_data", "images"))
    p.add_argument("--out_dir", default=os.path.join(os.environ.get("SGT_ROOT", "."), "data", "bach_graphs"))
    p.add_argument("--target", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "target.png"))
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--nparts", type=int, default=1)
    main(p.parse_args())
