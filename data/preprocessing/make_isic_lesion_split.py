"""
make_isic_lesion_split.py
=========================
Builds the lesion-level, stratified 70/15/15 split for ISIC 2018 Task 3 (HAM10000) used in the paper
(manuscript Sec. 0.6.3: seed 42, near-duplicates with Hamming distance <= 2 kept in one partition).

Steps
  1. Read diagnosis labels (GroundTruth CSV) and lesion IDs (LesionGroupings CSV).
  2. (optional) Perceptual-hash every image and merge near-duplicate images into
     the same group, even if they carry different lesion IDs.
  3. StratifiedGroupKFold(20 folds): 14 folds train, 3 val, 3 test
     -> every lesion group lands in exactly one partition.
  4. Verify zero overlap and print the counts needed for Table S1.

Usage
  python make_isic_lesion_split.py \
      --gt      .../ISIC2018_Task3_Training_GroundTruth.csv \
      --groups  .../ISIC2018_Task3_Training_LesionGroupings.csv \
      --img_dir .../ISIC2018_Task3_Training_Input \
      --out     splits/isic2018_lesion_split.csv

  The threshold (default 2) was chosen by inspecting candidate pairs with inspect_pairs.py.

  (HAM10000_metadata.csv also works for --groups: it has image_id + lesion_id.)
Requires: numpy, pandas, scikit-learn; for --img_dir also pillow + imagehash.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.environ.get("SGT_ROOT", ".")

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_table(gt_csv, groups_csv):
    gt = pd.read_csv(gt_csv)
    gt["label"] = gt[CLASSES].values.argmax(1)
    grp = pd.read_csv(groups_csv)
    if "image" not in grp.columns and "image_id" in grp.columns:
        grp = grp.rename(columns={"image_id": "image"})
    df = gt[["image", "label"]].merge(grp[["image", "lesion_id"]], on="image", how="left")
    n_missing = int(df["lesion_id"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} images have no lesion_id - check the groupings file.")
    mixed = df.groupby("lesion_id")["label"].nunique()
    if (mixed > 1).any():
        print(f"WARNING: {(mixed > 1).sum()} lesions have images with different labels.")
    return df


def phash_uint64(paths):
    import imagehash
    from PIL import Image
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        bits = imagehash.phash(Image.open(p).convert("RGB")).hash.flatten()
        out[i] = int.from_bytes(np.packbits(bits).tobytes(), "big")
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}")
    return out


def near_duplicate_pairs(H, threshold, chunk=512):
    pairs = []
    for s in range(0, len(H), chunk):
        x = np.bitwise_xor(H[s:s + chunk, None], H[None, :])          # (c, n) uint64
        d = POPCOUNT[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        i, j = np.nonzero(d <= threshold)
        i = i + s
        keep = j > i
        pairs += list(zip(i[keep].tolist(), j[keep].tolist(), d[i[keep] - s, j[keep]].tolist()))
    return pairs


def merge_groups(df, pairs):
    parent = {g: g for g in df["lesion_id"].unique()}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    lid = df["lesion_id"].tolist()
    n_merged = 0
    for i, j, _ in pairs:
        a, b = find(lid[i]), find(lid[j])
        if a != b:
            parent[a] = b
            n_merged += 1
    df["group"] = [find(g) for g in lid]
    return n_merged


def main(a):
    df = load_table(a.gt, a.groups)
    print(f"{len(df)} images, {df['lesion_id'].nunique()} unique lesions")
    print("Images per lesion:", df.groupby("lesion_id").size().value_counts().sort_index().to_dict())

    df["group"] = df["lesion_id"]
    if a.img_dir:
        paths = [Path(a.img_dir) / f"{i}.jpg" for i in df["image"]]
        print("Computing perceptual hashes ...")
        H = phash_uint64(paths)
        pairs = near_duplicate_pairs(H, a.threshold)
        cross = [(i, j, d) for i, j, d in pairs if df["lesion_id"][i] != df["lesion_id"][j]]
        print(f"Near-duplicate pairs (Hamming <= {a.threshold}): {len(pairs)} total, "
              f"{len(cross)} across different lesion IDs")
        dup_csv = Path(a.out).parent / "isic2018_near_duplicates_cross_lesion.csv"
        pd.DataFrame([(df["image"][i], df["image"][j], d) for i, j, d in cross],
                     columns=["image_a", "image_b", "hamming"]).to_csv(dup_csv, index=False)
        n_merged = merge_groups(df, cross)
        print(f"Lesion groups merged because of near-duplicates: {n_merged}")
        print(f"Pairs written to {dup_csv}; review them with preprocessing/inspect_pairs.py")

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=a.seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
        fold[idx] = k
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))

    # ---- verification: no group / lesion crosses partitions ----
    for unit in ["group", "lesion_id"]:
        sets = {s: set(df.loc[df["split"] == s, unit]) for s in ["train", "val", "test"]}
        for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
            n = len(sets[x] & sets[y])
            assert n == 0, f"{unit} overlap {x}-{y}: {n}"
    print("Overlap check passed: no lesion or duplicate group spans two partitions.")

    df["class"] = df["label"].map(dict(enumerate(CLASSES)))
    print("\nCounts per split (for Table S1):")
    print(df.groupby("split").agg(lesions=("lesion_id", "nunique"),
                                  images=("image", "count")).loc[["train", "val", "test"]])
    print("\nImages per class per split:")
    print(pd.crosstab(df["class"], df["split"], margins=True)[["train", "val", "test", "All"]])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[["image", "lesion_id", "group", "label", "class", "split"]].to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--img_dir", default=None, help="enable near-duplicate grouping")
    p.add_argument("--threshold", type=int, default=2, help="max Hamming distance (of 64 bits); paper: 2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=f"{ROOT}/splits/isic2018_lesion_split.csv")
    main(p.parse_args())
