"""
organize_bracs_official.py
==========================
Rebuild clean BRACS graph folders that follow the OFFICIAL split, as defined by the
ROI image folders  <png_root>/{train,val,test}/<class>/**/BRACS_*.png .

What it does (nothing is moved or deleted; graphs are COPIED):
  1. Reads official split membership of every ROI from the image folders.
  2. Audits the official split: ROI duplicates and slide overlap between splits.
  3. Finds an existing graph (.bin) for every ROI in the given graph folders.
  4. Copies graphs into <out_root>/{train,val,test}_official/ and the augmented
     graphs of OFFICIAL TRAINING ROIs ONLY into <out_root>/aug_official/.
  5. Writes:
       bracs_official_membership.csv   roi, slide, class, split, graph_source
       missing_graphs_<split>.txt      ROI image paths that still need a graph
       old_folder_leakage.csv          ROIs of val/test found in the OLD train folder
  Run first with --dry_run to see the summary without copying.

The augmented graphs copied to aug_official/ are NOT used by train.py (no augmentation for BRACS).

Usage:
  python preprocessing/organize_bracs_official.py --dry_run
  python preprocessing/organize_bracs_official.py
"""
import os
import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

G = os.environ.get("SGT_ROOT", ".")
SPLITS = ["train", "val", "test"]
SLIDE_RE = re.compile(r"(BRACS_\d+)_")


def class_of(png: Path, split_dir: Path) -> str:
    """Top-level class folder under the split dir, e.g. '3_ADH' (also for .../3_ADH/selected/)."""
    return png.relative_to(split_dir).parts[0]


def main(a):
    png_root = Path(a.png_root)

    # ---- 1. official membership from image folders ----
    member = {}            # roi -> (split, class, png_path)
    dup = []
    for split in SPLITS:
        sd = png_root / split
        for png in sorted(sd.rglob("BRACS_*.png")):
            roi = png.stem
            if roi in member:
                dup.append((roi, member[roi][0], split, str(png)))
                continue
            member[roi] = (split, class_of(png, sd), str(png))
    print(f"ROI images found: {len(member)}  (duplicate ROI names: {len(dup)})")
    for d in dup[:10]:
        print("  duplicate:", d)

    # ---- 2. audit the official split at slide level ----
    slides = defaultdict(set)
    for roi, (split, _, _) in member.items():
        m = SLIDE_RE.match(roi + "_")
        slides[split].add(m.group(1) if m else roi)
    print("\nOfficial split (from image folders):")
    for s in SPLITS:
        n = sum(1 for v in member.values() if v[0] == s)
        print(f"  {s:5s}: {n:5d} ROIs from {len(slides[s]):3d} slides")
    for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"  shared slides {x}-{y}: {len(slides[x] & slides[y])}")
    print("\nROIs per class per split:")
    cnt = Counter((v[1], v[0]) for v in member.values())
    for c in sorted({v[1] for v in member.values()}):
        print(f"  {c:8s} " + "  ".join(f"{s}={cnt[(c, s)]:4d}" for s in SPLITS))

    # ---- 3. index existing graphs ----
    graph_of = {}          # roi -> path (first match in priority order)
    for gd in a.graph_dirs:
        for f in Path(gd).glob("BRACS_*.bin"):
            if f.stem.endswith("_aug"):
                continue
            graph_of.setdefault(f.stem, str(f))
    aug_of = {}
    for f in Path(a.aug_dir).glob("BRACS_*_aug.bin"):
        aug_of[f.stem[:-4]] = str(f)

    # leakage report for the OLD training folder
    old_train = {f.stem for f in Path(a.old_train_dir).glob("BRACS_*.bin")}
    leak = [(r, member[r][0]) for r in sorted(old_train) if r in member and member[r][0] != "train"]
    not_official = sorted(r for r in old_train if r not in member)
    print(f"\nOLD train folder: {len(old_train)} graphs; "
          f"{len(leak)} belong to official val/test "
          f"({sum(1 for _, s in leak if s == 'val')} val, {sum(1 for _, s in leak if s == 'test')} test); "
          f"{len(not_official)} are not in the official image folders")

    # ---- 4. copy into clean folders ----
    out = Path(a.out_root)
    missing = defaultdict(list)
    rows = []
    n_copied = Counter()
    for roi, (split, cls, png) in sorted(member.items()):
        src = graph_of.get(roi)
        rows.append([roi, SLIDE_RE.match(roi + "_").group(1) if SLIDE_RE.match(roi + "_") else "",
                     cls, split, src or ""])
        if src is None:
            missing[split].append(png)
            continue
        if not a.dry_run:
            dst = out / f"{split}_official" / Path(src).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        n_copied[split] += 1
    n_aug = 0
    for roi, (split, _, _) in member.items():
        if split == "train" and roi in aug_of:
            if not a.dry_run:
                dst = out / "aug_official" / Path(aug_of[roi]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(aug_of[roi], dst)
            n_aug += 1

    print("\nGraphs available for the official split:")
    for s in SPLITS:
        tot = sum(1 for v in member.values() if v[0] == s)
        print(f"  {s:5s}: {n_copied[s]:5d} / {tot:5d}   missing {len(missing[s])}")
    print(f"  augmented graphs of official training ROIs: {n_aug}")

    # ---- 5. write reports ----
    with open("bracs_official_membership.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "slide", "class", "split", "graph_source"])
        w.writerows(rows)
    for s in SPLITS:
        Path(f"missing_graphs_{s}.txt").write_text("\n".join(missing[s]) + ("\n" if missing[s] else ""))
    with open("old_folder_leakage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_in_old_train_folder", "official_split"])
        w.writerows(leak)
    print("\nWrote bracs_official_membership.csv, missing_graphs_{train,val,test}.txt, "
          "old_folder_leakage.csv")
    print("DRY RUN - nothing copied." if a.dry_run else f"Clean folders written under {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--png_root", default=f"{G}/data")
    p.add_argument("--graph_dirs", nargs="+",
                   default=[f"{G}/data/tissue_graphs/val_fixed_512",
                            f"{G}/data/tissue_graphs/train_fixed_512",
                            f"{G}/data/tissue_graphs/test"],
                   help="folders searched for existing graphs, in priority order")
    p.add_argument("--aug_dir", default=f"{G}/data/tissue_graphs/Augmented_tg_512")
    p.add_argument("--old_train_dir", default=f"{G}/data/tissue_graphs/train_fixed_512")
    p.add_argument("--out_root", default=f"{G}/data/tissue_graphs_official")
    p.add_argument("--dry_run", action="store_true")
    main(p.parse_args())
