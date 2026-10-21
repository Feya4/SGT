"""
preprocess_isic.py - build and cache the ISIC 2018 superpixel graphs (run once, before train.py).

Paths, n_segments and k_neighbours come from config.py (the same settings train.py uses), and the cache file
names come from ISICDataset._cache_path, so the cache always matches what training looks for. Graphs already
cached are skipped, so the script can be re-run.

Usage (from the repository root)
  export SGT_ROOT=/path/to/workspace
  python preprocessing/preprocess_isic.py
  python preprocessing/preprocess_isic.py --set data.isic_image_dir=/other/path
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISIC_CLASSES, apply_overrides, get_config  # noqa: E402
from dataset import ISICDataset, image_to_graph  # noqa: E402


def main(a):
    d = apply_overrides(get_config("isic"), a.set).data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = present = 0
    t0 = time.time()
    for split in ("train", "val", "test"):
        ds = ISICDataset(d.isic_image_dir, d.isic_annotations, d.isic_split_csv, split, d.isic_cache_dir,
                         device, d.n_segments, d.k_neighbours)
        for img_path, label in ds.samples:
            cache = ds._cache_path(img_path)
            if cache.exists():
                present += 1
                continue
            data = image_to_graph(img_path, label, device, n_segments=ds.n_segments, k_neighbours=ds.k_neighbours)
            torch.save(data, str(cache))
            built += 1
            if built % 200 == 0:
                print(f"built {built} graphs ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: built {built}, already cached {present}, classes {ISIC_CLASSES}, cache {d.isic_cache_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.isic_image_dir=...")
    main(p.parse_args())
