"""
checkpoint.py - rebuild a trained SGT model and its config from best.pt / latest.pt.

Checkpoints written by train.py contain {"model": state_dict, "config": cfg.to_dict(), "epoch": ...}.
Checkpoints made with the code before the October 2026 revision have no switch fields in their config; they are
loaded with the pre-revision behaviour so that they evaluate exactly as they were trained.
"""
from typing import Tuple

import torch

# settings of the pre-revision code, applied when a field is absent from a saved config
LEGACY_MODEL = {"fuse_prev_morph": False, "drop_path_per_graph": False, "drop_path_max": 0.15,
                "use_mase": True, "use_sre": True, "variant": "sgt", "match_params": True}
LEGACY_DATA = {"bracs_pos": "mm", "bracs_area": "zlog"}
SWITCHES_MODEL = ("fuse_prev_morph", "drop_path_per_graph", "drop_path_max", "use_mase", "use_sre", "variant",
                  "match_params")
SWITCHES_DATA = ("bracs_pos", "bracs_area")


def config_from_checkpoint(ck: dict):
    """Config object equal to the one used to train the checkpoint (legacy values for missing fields)."""
    from config import get_config
    saved = ck.get("config")
    if saved is None:
        raise ValueError("checkpoint has no 'config' entry; build the model from a config instead")
    cfg = get_config(saved.get("dataset", "bracs"))
    for section, legacy in (("model", LEGACY_MODEL), ("train", {}), ("data", LEGACY_DATA)):
        obj, vals = getattr(cfg, section), saved.get(section, {})
        for name in vars(obj):
            if name in vals:
                setattr(obj, name, vals[name])
            elif name in legacy:
                setattr(obj, name, legacy[name])
    for name in ("dataset", "exp_name", "seed", "morph_mode", "pos_mode"):
        if name in saved:
            setattr(cfg, name, saved[name])
    return cfg.validate()


def load_model(path: str, device) -> Tuple[torch.nn.Module, object, dict]:
    """(model in eval mode, config, raw checkpoint)."""
    from model import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ck)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def check_compatible(ck: dict, cfg) -> None:
    """Raise if the switches of the current config differ from those the checkpoint was trained with."""
    if ck.get("config") is None:
        return
    trained = config_from_checkpoint(ck)
    diffs = [f"model.{k}: trained={getattr(trained.model, k)} current={getattr(cfg.model, k)}"
             for k in SWITCHES_MODEL if getattr(trained.model, k) != getattr(cfg.model, k)]
    if cfg.dataset == "bracs":
        diffs += [f"data.{k}: trained={getattr(trained.data, k)} current={getattr(cfg.data, k)}"
                  for k in SWITCHES_DATA if getattr(trained.data, k) != getattr(cfg.data, k)]
    if diffs:
        raise ValueError("checkpoint was trained with different settings; pass them with --set:\n  "
                         + "\n  ".join(diffs))


# fields of the older sgt_impl13 reports ("args") -> (section, field) of the current Config
ARGS_MAP = {"hidden_dim": ("model", "hidden_dim"), "num_heads": ("model", "num_heads"),
            "num_layers": ("model", "num_layers"), "morph_hidden": ("model", "morph_hidden_dim"),
            "ffn_dim": ("model", "ffn_dim"), "dropout": ("model", "dropout"), "beta_init": ("model", "beta_init"),
            "classifier_hidden": ("model", "classifier_hidden_dim"),
            "epochs": ("train", "epochs"), "patience": ("train", "patience"), "batch_size": ("train", "batch_size"),
            "lr": ("train", "lr"), "lr_min": ("train", "lr_min"), "weight_decay": ("train", "weight_decay"),
            "warmup_epochs": ("train", "warmup_epochs"), "grad_clip": ("train", "grad_clip"),
            "loss": ("train", "loss"), "focal_gamma": ("train", "focal_gamma"),
            "class_weights": ("train", "class_weights"), "num_workers": ("train", "num_workers"),
            "seed": (None, "seed"), "morph_mode": (None, "morph_mode"), "pos_mode": (None, "pos_mode"),
            "exp_name": (None, "exp_name")}


def config_from_report(report: dict):
    """Config of a finished run from its test_report.json: current format ("config") or sgt_impl13 ("args")."""
    if "config" in report:
        return config_from_checkpoint({"config": report["config"]})
    args = report.get("args")
    if args is None:
        raise ValueError("report has neither 'config' nor 'args'")
    from config import get_config
    ds = str(report.get("dataset", "bracs")).lower()
    cfg = get_config("isic" if "isic" in ds else "camelyon" if "camelyon" in ds else "bracs")
    for name, value in args.items():
        if name in ARGS_MAP:
            section, field = ARGS_MAP[name]
            setattr(getattr(cfg, section) if section else cfg, field, value)
    if "no_weighted_sampling" in args:
        cfg.train.weighted_sampling = not args["no_weighted_sampling"]
    if "train_dir" in args:                                     # .../tissue_graphs_official_morph/train_official
        from pathlib import Path
        cfg.data.bracs_dir = str(Path(args["train_dir"]).parent)
    for k, v in LEGACY_MODEL.items():                            # model behaviour of the pre-revision code
        setattr(cfg.model, k, v)
    for k, v in LEGACY_DATA.items():
        setattr(cfg.data, k, v)
    return cfg.validate()
