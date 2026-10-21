"""
train.py - train and evaluate SGT on BRACS, ISIC 2018 (and CAMELYON16 once its loader is added).

    python train.py --dataset bracs --seed 0 --exp_name sgt_bracs_s0
    python train.py --dataset isic  --seed 0 --exp_name sgt_isic_s0
    python train.py --dataset bracs --seed 0 --exp_name x --morph_mode shuffle        # control experiment
    python train.py --dataset bracs --exp_name x --set train.lr=1e-4 model.dropout=0.2  # any config field
    python train.py --dataset bracs --exp_name x --test_only                          # re-evaluate best.pt
    python train.py --dataset bracs --exp_name abl_noSRE  --set model.use_sre=false    # ablation: no SRE
    python train.py --dataset bracs --exp_name abl_noMASE --set model.use_mase=false   # ablation: no MASE
    python train.py --dataset bracs --exp_name ctl_concat --set model.variant=concat    # parameter-matched concat
    (all controls of the revision, 5 seeds each: python run_controls.py --ref sgt_bracs)

Reproduce / re-evaluate runs made with the code before the October 2026 revision:
    --set model.fuse_prev_morph=false model.drop_path_per_graph=false data.bracs_pos=mm data.bracs_area=zlog

Protocol (identical for all datasets)
  * train on the training split only;
  * after every epoch compute weighted F1 on the validation split; the checkpoint with the highest validation
    weighted F1 is kept (best.pt); training stops after `patience` epochs without improvement;
  * the test split is evaluated exactly once, with best.pt, after training;
  * per-sample predictions (id, label, softmax probabilities) and every setting used are written to
    <output_dir>/<exp_name>/report/.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import apply_overrides, get_config
from dataset import build_dataloaders
from model import build_model
from utils import (apply_controls, build_criterion, check_compatible, compute_metrics, get_device, get_logger,
                   set_seed)


def get_targets(batch, num_classes):
    t = batch.y.view(-1)
    if t.min().item() < 0 or t.max().item() >= num_classes:
        raise ValueError(f"Invalid labels in batch: {t.unique().tolist()}")
    return t


def train_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    t, losses = cfg.train, []
    for batch in loader:
        batch = apply_controls(batch.to(device), cfg.morph_mode, cfg.pos_mode)
        targets = get_targets(batch, cfg.model.num_classes)
        if t.feat_noise_std > 0:                                   # graph augmentation (training only)
            batch.x = batch.x + torch.randn_like(batch.x) * t.feat_noise_std
        if t.edge_drop_p > 0 and batch.edge_index.size(1) > 0:
            keep = torch.rand(batch.edge_index.size(1), device=device) > t.edge_drop_p
            if keep.sum() > 0:
                batch.edge_index = batch.edge_index[:, keep]
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        if t.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, criterion, device, cfg):
    """(labels, softmax probabilities, mean loss) in dataset order."""
    model.eval()
    ys, ps, losses = [], [], []
    gen = None
    if cfg.morph_mode != "real" or cfg.pos_mode != "real":       # same permutation at every evaluation
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed + 10000)
    for batch in loader:
        batch = apply_controls(batch.to(device), cfg.morph_mode, cfg.pos_mode, gen)
        targets = get_targets(batch, cfg.model.num_classes)
        logits = model(batch)
        losses.append(criterion(logits, targets).item())
        ps.append(F.softmax(logits, -1).cpu())
        ys.append(targets.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    if len(y) != len(loader.dataset):
        raise RuntimeError(f"evaluated {len(y)} graphs, dataset has {len(loader.dataset)}")
    return y, p, float(np.mean(losses))


def save_predictions(path, ds, y, probs, class_names):
    ids = ds.get_ids()
    assert len(ids) == len(y), "prediction order does not match dataset order"
    np.savez(path, sample_id=np.array(ids), y_true=y, y_pred=probs.argmax(1), probs=probs,
             classes=np.array(class_names))


def main(cfg, resume=False, test_only=False):
    cfg.validate()
    set_seed(cfg.seed)
    device = get_device("auto")
    exp_dir = Path(cfg.output_dir) / cfg.exp_name
    rep_dir = exp_dir / "report"
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.exp_name
    rep_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("SGT", str(exp_dir / "train.log"))
    names, t = cfg.class_names, cfg.train

    logger.info("=" * 65)
    logger.info(f"SGT {cfg.dataset} | seed={cfg.seed} | morph={cfg.morph_mode} | pos={cfg.pos_mode} | device={device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    sizes = {k: len(l.dataset) for k, l in (("train", train_loader), ("val", val_loader), ("test", test_loader))}
    logger.info(f"split sizes: {sizes}")

    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"parameters: {n_params:,}")
    criterion, loss_name = build_criterion(t, train_loader.dataset.get_labels(), cfg.model.num_classes, device)
    logger.info(f"loss: {loss_name}")

    optimizer = AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    warm = LinearLR(optimizer, start_factor=0.1, total_iters=t.warmup_epochs)
    cos = CosineAnnealingLR(optimizer, T_max=max(t.epochs - t.warmup_epochs, 1), eta_min=t.lr_min)

    best_pt, last_pt = ckpt_dir / "best.pt", ckpt_dir / "latest.pt"
    start_epoch, best_f1, best_epoch, no_improve = 1, -1.0, 0, 0
    if resume and last_pt.exists():
        s = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); optimizer.load_state_dict(s["optimizer"])
        warm.load_state_dict(s["warm"]); cos.load_state_dict(s["cos"])
        start_epoch, best_f1, best_epoch, no_improve = s["epoch"] + 1, s["best_f1"], s["best_epoch"], s["no_improve"]
        logger.info(f"resumed after epoch {s['epoch']} (best val wF1 {best_f1:.4f} @ {best_epoch})")

    if not test_only:
        for epoch in range(start_epoch, t.epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg)
            (warm if epoch <= t.warmup_epochs else cos).step()
            y_v, p_v, v_loss = predict(model, val_loader, criterion, device, cfg)
            vm = compute_metrics(y_v, p_v, names)
            improved = vm["weighted_f1"] > best_f1 + 1e-5
            if improved:
                best_f1, best_epoch, no_improve = vm["weighted_f1"], epoch, 0
                torch.save({"epoch": epoch, "model": model.state_dict(), "val_weighted_f1": best_f1,
                            "config": cfg.to_dict()}, best_pt)
            else:
                no_improve += 1
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "warm": warm.state_dict(), "cos": cos.state_dict(), "best_f1": best_f1,
                        "best_epoch": best_epoch, "no_improve": no_improve}, last_pt)
            logger.info(f"Epoch {epoch:03d}/{t.epochs} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                        f"{time.time() - t0:.1f}s | train loss={tr_loss:.4f} | val loss={v_loss:.4f} "
                        f"wF1={vm['weighted_f1']:.4f} mF1={vm['macro_f1']:.4f} acc={vm['accuracy']:.4f} "
                        f"AUC={vm['auc_macro_ovr']:.4f} [best={best_f1:.4f} @ {best_epoch}]" + ("  *" if improved else ""))
            if no_improve >= t.patience:
                logger.info(f"early stop at epoch {epoch} (patience {t.patience})")
                break

    # ── final evaluation: best validation checkpoint, test split used once ──
    if not best_pt.exists():
        raise FileNotFoundError(f"{best_pt} not found - train first, or check --exp_name")
    ck = torch.load(best_pt, map_location=device, weights_only=False)
    check_compatible(ck, cfg)                 # e.g. a pre-revision checkpoint evaluated with new settings
    model.load_state_dict(ck["model"])
    best_epoch = ck["epoch"]
    y_v, p_v, _ = predict(model, val_loader, criterion, device, cfg)
    y_t, p_t, _ = predict(model, test_loader, criterion, device, cfg)
    save_predictions(rep_dir / "val_predictions.npz", val_loader.dataset, y_v, p_v, names)
    save_predictions(rep_dir / "test_predictions.npz", test_loader.dataset, y_t, p_t, names)
    vm, tm = compute_metrics(y_v, p_v, names), compute_metrics(y_t, p_t, names)

    logger.info("=" * 65)
    logger.info(f"TEST (best val epoch {best_epoch}) | wF1={tm['weighted_f1']:.4f} mF1={tm['macro_f1']:.4f} "
                f"acc={tm['accuracy']:.4f} AUC={tm['auc_macro_ovr']:.4f}")
    logger.info("\n" + tm["report"])

    report = {
        "dataset": cfg.dataset, "seed": cfg.seed, "split_sizes": sizes, "best_epoch": int(best_epoch),
        "controls": {"morph_mode": cfg.morph_mode, "pos_mode": cfg.pos_mode},
        "loss": loss_name, "n_params": n_params, "variant": cfg.model.variant,
        "metric_definitions": {
            "weighted_f1": "sklearn f1_score(average='weighted')",
            "macro_f1": "sklearn f1_score(average='macro')",
            "auc_macro_ovr": "sklearn roc_auc_score(multi_class='ovr', average='macro')",
            "model_selection": "checkpoint with highest validation weighted F1"},
        "validation": {k: v for k, v in vm.items() if k != "report"},
        "test": {k: v for k, v in tm.items() if k != "report"},
        "config": cfg.to_dict(),
    }
    json.dump(report, open(rep_dir / "test_report.json", "w"), indent=2)
    logger.info(f"report and predictions: {rep_dir}")
    return tm


def parse_args():
    p = argparse.ArgumentParser(description="Train / evaluate SGT",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["bracs", "isic", "camelyon16"], required=True)
    p.add_argument("--exp_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--morph_mode", choices=["real", "shuffle", "degree", "none"], default=None,
                   help="control experiment on the morphological attributes")
    p.add_argument("--pos_mode", choices=["real", "shuffle"], default=None,
                   help="control experiment on the node positions")
    p.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override any config field, e.g. train.lr=1e-4 model.dropout=0.2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--test_only", action="store_true", help="skip training, evaluate best.pt")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = get_config(a.dataset)
    for name in ("exp_name", "seed", "morph_mode", "pos_mode"):
        if getattr(a, name) is not None:
            setattr(cfg, name, getattr(a, name))
    if a.epochs is not None:
        cfg.train.epochs = a.epochs
    if a.patience is not None:
        cfg.train.patience = a.patience
    apply_overrides(cfg, a.set)
    if cfg.dataset == "isic":                       # graph building uses CUDA in the main process
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main(cfg, resume=a.resume, test_only=a.test_only)
