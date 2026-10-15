"""
config.py - every hyper-parameter and path of the SGT experiments in one place.

Three presets, selected by dataset name:

    cfg = get_config("bracs")       # BRACS 7-class subtyping
    cfg = get_config("isic")        # ISIC 2018 Task 3, 7-class dermoscopy
    cfg = get_config("camelyon16")  # CAMELYON16, normal vs tumor (training settings only, see dataset.py)

Presets follow the manuscript (Implementation Details, Section 0.7). Any value can be
overridden from the command line of train.py (see --set).

All paths are relative to the environment variable SGT_ROOT (default: current folder).
"""
import os
from dataclasses import dataclass, field, fields, asdict

ROOT = os.environ.get("SGT_ROOT", ".")

BRACS_CLASSES = ["N", "PB", "UDH", "ADH", "FEA", "DCIS", "IC"]
ISIC_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CAMELYON16_CLASSES = ["normal", "tumor"]
CLASS_NAMES = {"bracs": BRACS_CLASSES, "isic": ISIC_CLASSES, "camelyon16": CAMELYON16_CLASSES}


@dataclass
class ModelConfig:
    in_dim: int = 512                 # node feature dimension
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    morph_hidden_dim: int = 64        # hidden size of the morphological-token MLP
    ffn_dim: int = 512
    dropout: float = 0.1
    beta_init: float = 1.0            # initial value of the learnable spatial-bias scale
    classifier_hidden_dim: int = 64
    num_classes: int = 7
    drop_path_max: float = 0.15       # stochastic depth, linear from 0 (layer 1) to this value (layer L)
    # switches; the defaults implement the manuscript. Set the next two to False to reproduce runs made
    # with the code before the October 2026 revision.
    fuse_prev_morph: bool = True      # gate/fusion use m^(l-1) (Eqs. 6-7); False: m^(l)
    drop_path_per_graph: bool = True  # stochastic depth per graph; False: per node
    use_mase: bool = True             # ablation: False removes the morphological stream
    use_sre: bool = True              # ablation: False removes the spatial bias


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 5e-5
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    patience: int = 30                # early stopping on validation weighted F1
    grad_clip: float = 1.0
    loss: str = "focal"               # "focal" | "ce" | "ce_ls" (label-smoothing cross-entropy)
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    class_weights: bool = False       # inverse-frequency weights inside the loss
    weighted_sampling: bool = True    # inverse-frequency WeightedRandomSampler
    feat_noise_std: float = 0.0       # graph augmentation (training only); 0 = off
    edge_drop_p: float = 0.0          # graph augmentation (training only); 0 = off
    num_workers: int = 4


@dataclass
class DataConfig:
    # BRACS: folders with the official-split graphs (written by preprocessing/compute_morphology.py)
    bracs_dir: str = f"{ROOT}/data/tissue_graphs_official_morph"
    bracs_image_dir: str = f"{ROOT}/data/BRACS_RoI"   # ROI images; only their width/height are read
    bracs_pos: str = "normalised"     # "normalised": centroid / longer ROI side (manuscript) | "mm" (pre-revision)
    bracs_area: str = "max"           # "max": area / max area in the graph (manuscript) | "zlog" (pre-revision)
    # ISIC 2018
    isic_image_dir: str = f"{ROOT}/data/ISIC2018_Task3_Training_Input"
    isic_annotations: str = (f"{ROOT}/data/ISIC2018_Task3_Training_GroundTruth/"
                             "ISIC2018_Task3_Training_GroundTruth.csv")
    isic_split_csv: str = f"{ROOT}/splits/isic2018_lesion_split.csv"
    isic_cache_dir: str = f"{ROOT}/data/graph_cache/isic"
    n_segments: int = 75              # SLIC superpixels per image (ISIC)
    k_neighbours: int = 6             # k-NN graph (ISIC)


@dataclass
class Config:
    dataset: str = "bracs"
    exp_name: str = "sgt_bracs_s0"
    seed: int = 0
    morph_mode: str = "real"          # real | shuffle | degree | none   (control experiments)
    pos_mode: str = "real"            # real | shuffle                   (control experiments)
    checkpoint_dir: str = f"{ROOT}/checkpoints"
    output_dir: str = f"{ROOT}/outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def class_names(self):
        return CLASS_NAMES[self.dataset]

    def validate(self):
        assert self.dataset in CLASS_NAMES, f"unknown dataset {self.dataset!r}"
        assert self.model.num_classes == len(self.class_names)
        assert self.model.hidden_dim % self.model.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.train.loss in ("focal", "ce", "ce_ls")
        assert self.morph_mode in ("real", "shuffle", "degree", "none")
        assert self.pos_mode in ("real", "shuffle")
        assert self.data.bracs_pos in ("normalised", "mm")
        assert self.data.bracs_area in ("max", "zlog")
        assert 0.0 <= self.model.drop_path_max < 1.0
        return self

    def to_dict(self):
        return asdict(self)


def get_config(dataset: str) -> Config:
    cfg = Config(dataset=dataset)
    if dataset == "bracs":
        cfg.exp_name = "sgt_bracs_s0"
        cfg.model.dropout = 0.1
        cfg.model.classifier_hidden_dim = 64
        cfg.train.epochs, cfg.train.patience = 200, 30
        cfg.train.lr, cfg.train.weight_decay = 5e-5, 1e-4
        cfg.train.loss = "focal"
        cfg.train.num_workers = 4
    elif dataset == "isic":
        cfg.exp_name = "sgt_isic_s0"
        cfg.model.dropout = 0.5
        cfg.model.classifier_hidden_dim = 128
        cfg.train.epochs, cfg.train.patience = 200, 30
        cfg.train.lr, cfg.train.weight_decay = 3e-5, 5e-4
        cfg.train.loss = "ce_ls"
        cfg.train.feat_noise_std, cfg.train.edge_drop_p = 0.02, 0.10
        cfg.train.num_workers = 2
    elif dataset == "camelyon16":
        cfg.exp_name = "sgt_camelyon16_s0"
        cfg.model.num_classes = 2
        cfg.model.dropout = 0.1
        cfg.model.classifier_hidden_dim = 128
        cfg.train.epochs, cfg.train.patience, cfg.train.warmup_epochs = 100, 20, 5
        cfg.train.batch_size = 32
        cfg.train.lr, cfg.train.weight_decay = 1e-4, 1e-4
        cfg.train.loss = "focal"
        cfg.train.class_weights = True                  # unlike BRACS/ISIC: class weights in the loss ...
        cfg.train.weighted_sampling = True              # ... AND weighted sampling
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return cfg.validate()


def apply_overrides(cfg: Config, items):
    """items: ["train.lr=1e-4", "model.dropout=0.2", "seed=3", ...] (dotted names)."""
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like name=value, got {item!r}")
        obj, parts = cfg, key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        name = parts[-1]
        if name not in {f.name for f in fields(obj)}:
            raise KeyError(f"unknown config field {key!r}")
        cur = getattr(obj, name)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes")
        else:
            val = type(cur)(raw)
        setattr(obj, name, val)
    return cfg.validate()
