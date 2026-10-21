"""
utils.py - seeding, logging, losses, class/sample weights, control experiments, metrics.
"""
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Reproducibility / environment ───────────────────────────────────────────

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Logger writing to the console and to log_file. Handlers are replaced on every call, so that several
    runs in one process (run_controls.py) each log to their own file."""
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─── Losses ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss  FL = (1 - p_t)^gamma * CE   (gamma = 0 gives cross-entropy)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


def build_criterion(tcfg, labels: List[int], num_classes: int, device) -> (nn.Module, str):
    """Loss from TrainConfig; returns (criterion, human-readable description)."""
    w = compute_class_weights(labels, num_classes, device) if tcfg.class_weights else None
    if tcfg.loss == "focal":
        crit, name = FocalLoss(tcfg.focal_gamma, w), f"focal loss (gamma={tcfg.focal_gamma})"
    elif tcfg.loss == "ce":
        crit, name = nn.CrossEntropyLoss(weight=w), "cross-entropy"
    else:
        if w is not None:
            raise ValueError("class_weights are not supported with the label-smoothing loss")
        crit, name = LabelSmoothingCrossEntropy(tcfg.label_smoothing), f"label-smoothing CE (eps={tcfg.label_smoothing})"
    name += f", class weights {'on' if w is not None else 'off'}, weighted sampling {'on' if tcfg.weighted_sampling else 'off'}"
    return crit.to(device), name


# ─── Class / sample weights (labels are validated, never clamped) ────────────

def _check_labels(labels, num_classes):
    bad = sorted({int(l) for l in labels if not 0 <= int(l) < num_classes})
    if bad:
        raise ValueError(f"labels outside [0, {num_classes}): {bad}")


def compute_class_weights(labels: List[int], num_classes: int, device) -> torch.Tensor:
    """Inverse-frequency weights  w_c = N / (K * n_c)."""
    _check_labels(labels, num_classes)
    counts = torch.zeros(num_classes)
    for l in labels:
        counts[int(l)] += 1
    return (len(labels) / (num_classes * counts.clamp(min=1))).to(device)


def compute_sample_weights(labels: List[int], num_classes: int) -> List[float]:
    """Per-sample weights 1/n_c for WeightedRandomSampler."""
    _check_labels(labels, num_classes)
    counts = [0] * num_classes
    for l in labels:
        counts[int(l)] += 1
    return [1.0 / counts[int(l)] for l in labels]


# ─── Control experiments (Editor comment 5) ──────────────────────────────────

def _permute_within_graph(t, batch, generator=None):
    """Random permutation of the rows of t, separately inside every graph of the batch."""
    noise = torch.rand(t.size(0), device=t.device, dtype=torch.double, generator=generator)
    return t[torch.argsort(batch.double() + noise)]


def apply_controls(batch, morph_mode: str = "real", pos_mode: str = "real", generator=None):
    """Applied identically in training and evaluation.

    Training draws a new permutation for every mini-batch. Evaluation passes a generator with a fixed seed
    (train.py: seed + 10000), so the permuted validation/test graphs are the same at every evaluation and are
    reproducible from the seed.

    morph_mode  real | shuffle (rows permuted within each graph) | degree (area, convexity = 0) | none (all 0)
                NOTE: "none" feeds zeros to MASE (the stream still exists); to REMOVE MASE use model.use_mase=false
    pos_mode    real | shuffle (centroids permuted within each graph)
    """
    if morph_mode == "shuffle":
        batch.morph = _permute_within_graph(batch.morph, batch.batch, generator)
    elif morph_mode == "degree":
        batch.morph = batch.morph.clone()
        batch.morph[:, 1:] = 0.0
    elif morph_mode == "none":
        batch.morph = torch.zeros_like(batch.morph)
    if pos_mode == "shuffle":
        batch.pos = _permute_within_graph(batch.pos, batch.batch, generator)
    return batch


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y: np.ndarray, probs: np.ndarray, class_names: List[str]) -> dict:
    """accuracy, weighted/macro F1, macro one-vs-rest AUC, per-class P/R/F1/AUC, confusion matrix (sklearn)."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    labels = list(range(len(class_names)))
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_auc = []
    for c in labels:
        yc = (y == c).astype(int)
        per_auc.append(float(roc_auc_score(yc, probs[:, c])) if 0 < yc.sum() < len(yc) else float("nan"))
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc_macro_ovr": auc,
        "per_class": {class_names[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                                       "auc_ovr": per_auc[c], "support": int(s[c])} for c in labels},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "report": classification_report(y, pred, labels=labels, target_names=class_names,
                                        digits=4, zero_division=0),
    }
