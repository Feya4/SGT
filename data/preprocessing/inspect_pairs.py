"""
inspect_pairs.py - save side-by-side near-duplicate candidate pairs for visual review.

For each Hamming distance it writes up to --per_dist pairs into
pair_review/d<distance>/ as one JPG per pair (left | right, IDs in file name).
Look through them and note the largest distance at which pairs are still
genuinely the same lesion; use that as --threshold in make_isic_lesion_split.py (paper: 2).
NOTE: make_isic_lesion_split.py only reports pairs up to its --threshold; run it once with a larger
threshold (e.g. --threshold 8) to collect candidates for this review, then re-run with the chosen value.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image


def main(a):
    d = pd.read_csv(a.pairs)
    out = Path(a.out)
    for dist, grp in d.groupby("hamming"):
        sub = grp if len(grp) <= a.per_dist else grp.sample(a.per_dist, random_state=0)
        folder = out / f"d{dist}"
        folder.mkdir(parents=True, exist_ok=True)
        for _, r in sub.iterrows():
            im1 = Image.open(Path(a.img_dir) / f"{r.image_a}.jpg").convert("RGB").resize((300, 225))
            im2 = Image.open(Path(a.img_dir) / f"{r.image_b}.jpg").convert("RGB").resize((300, 225))
            canvas = Image.new("RGB", (610, 225), "white")
            canvas.paste(im1, (0, 0))
            canvas.paste(im2, (310, 0))
            canvas.save(folder / f"{r.image_a}__{r.image_b}.jpg", quality=85)
        print(f"distance {dist}: saved {len(sub)} of {len(grp)} pairs to {folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=os.path.join(os.environ.get("SGT_ROOT", "."), "splits",
                                                   "isic2018_near_duplicates_cross_lesion.csv"))
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out", default="pair_review")
    p.add_argument("--per_dist", type=int, default=40)
    main(p.parse_args())
