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
                "use_mase": True, "use_sre": True}
LEGACY_DATA = {"bracs_pos": "mm", "bracs_area": "zlog"}
SWITCHES_MODEL = ("fuse_prev_morph", "drop_path_per_graph", "drop_path_max", "use_mase", "use_sre")
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
