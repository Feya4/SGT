"""
dataset.py - graph datasets for SGT: BRACS (official split) and ISIC 2018 (lesion-level split).

Every sample is a torch_geometric Data object with the fields the model reads:
    x          (N, 512)  node features
    edge_index (2, E)    undirected, duplicate-free graph without self-loops
    pos        (N, 2)    node centroids normalised to [0, 1] by the longer image side (manuscript, Sec. 0.5.2)
    morph      (N, 3)    [degree / max degree,  area / max area,  convexity]   (max taken within the graph)
    y          (1,)      class index

Convexity = region area / convex-hull area (skimage "solidity").

BRACS   region-adjacency graphs of H&E ROIs (official HACT preprocessing). The ROI width/height needed to
        normalise the centroids are read once from the ROI images (data.bracs_image_dir) and cached in
        <bracs_dir>/roi_image_sizes.json.
        Pre-revision behaviour (to reproduce runs made before October 2026):
            data.bracs_pos=mm     centroids in millimetres (0.25 um/pixel)
            data.bracs_area=zlog  z-scored log(area), mean/std of the TRAINING graphs
ISIC    SLIC superpixels of the dermoscopy image (75 regions), ResNet-50 features of a 64x64 crop
        projected to 512-d by one fixed random matrix (seed 42) shared by all images, k-NN graph on centroids.
        Graphs are cached on disk (GRAPH_VERSION is part of the cache key).
CAMELYON  precomputed patch graphs (one graph per 256x256 patch: 64 tiles of 32x32 as nodes, ResNet-18 512-d features,
        spatial k-NN edges, graph-derived morphology) read from {train,val,test}_graphs.pkl. The graphs are used
        exactly as stored: label (y), node positions (pos) and morphology (morph) are NOT recomputed - the BRACS and
        ISIC morphology code is never applied to them. Labels: 0 = Normal, 1 = Tumor.

No silent fixes: labels are validated, never clamped; no synthetic morphology; no augmentation graphs.
"""
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_undirected

from config import BRACS_CLASSES, CAMELYON_CLASSES, ISIC_CLASSES
from utils import compute_sample_weights

warnings.filterwarnings("ignore")

GRAPH_VERSION = "v4"            # bump whenever the ISIC graph construction changes
MM_PER_PIXEL = 0.25e-3          # 0.25 um / pixel at 40x
BRACS_TOKEN_TO_IDX = {c: i for i, c in enumerate(BRACS_CLASSES)}
ISIC_CLASS_TO_IDX = {c: i for i, c in enumerate(ISIC_CLASSES)}


# ═════════════════════════════════════════════════════════════════════════════
# BRACS
# ═════════════════════════════════════════════════════════════════════════════

def _load_dgl(path):
    import dgl
    gs, labels = dgl.load_graphs(str(path))
    return gs[0], labels


def _log_area_stats(paths):
    vals = []
    for p in paths:
        g, _ = _load_dgl(p)
        if "area" not in g.ndata:
            raise RuntimeError(f"{p} has no 'area' - run preprocessing/compute_morphology.py first")
        vals.append(torch.log(g.ndata["area"].float().clamp(min=1.0)))
    v = torch.cat(vals)
    return float(v.mean()), float(v.std())


def bracs_graph_to_data(g, label: int, name: str = "", pos_mode: str = "normalised", area_mode: str = "max",
                        roi_side_px: float = None, area_stats: dict = None) -> Data:
    """DGL region-adjacency graph with node fields feat, centroid, area, solidity -> Data.

    pos_mode   "normalised": centroid (pixels) / roi_side_px, the longer side of the ROI   (manuscript)
               "mm"        : centroid * 0.25 um                                             (pre-revision)
    area_mode  "max"       : area / max area of the graph                                   (manuscript)
               "zlog"      : (log area - mean) / std, training-set statistics               (pre-revision)
    """
    for k in ("feat", "centroid", "area", "solidity"):
        if k not in g.ndata:
            raise RuntimeError(f"{name}: missing node field '{k}'")
    n = g.num_nodes()
    src, dst = g.edges()
    ei = to_undirected(torch.stack([src, dst]).long(), num_nodes=n)
    ei = ei[:, ei[0] != ei[1]]                                   # no self-loops
    deg = torch.zeros(n).scatter_add_(0, ei[0], torch.ones(ei.size(1)))
    area = g.ndata["area"].float()
    if area_mode == "max":
        area_term = area / area.max().clamp(min=1.0)
    elif area_mode == "zlog":
        log_a = torch.log(area.clamp(min=1.0))
        area_term = (log_a - area_stats["mean"]) / area_stats["std"]
    else:
        raise ValueError(f"unknown area_mode {area_mode!r}")
    morph = torch.stack([deg / deg.max().clamp(min=1.0), area_term, g.ndata["solidity"].float()], dim=1)
    centroid = g.ndata["centroid"].float()                       # pixels at 40x
    if pos_mode == "normalised":
        if not roi_side_px:
            raise ValueError(f"{name}: ROI size unknown, cannot normalise centroids")
        pos = centroid / float(roi_side_px)
        if pos.min() < -1e-3 or pos.max() > 1.0 + 1e-3:
            raise ValueError(f"{name}: normalised centroids outside [0, 1] "
                             f"(range {pos.min():.3f}..{pos.max():.3f}) - wrong ROI size?")
    elif pos_mode == "mm":
        pos = centroid * MM_PER_PIXEL
    else:
        raise ValueError(f"unknown pos_mode {pos_mode!r}")
    return Data(x=g.ndata["feat"].float(), edge_index=ei, pos=pos, morph=morph,
                y=torch.tensor([label]), num_nodes=n)


class BRACSDataset(Dataset):
    def __init__(self, folder: str, pos_mode: str = "normalised", area_mode: str = "max",
                 roi_sizes: dict = None, area_stats: dict = None):
        super().__init__()
        self.files = sorted(Path(folder).glob("BRACS_*.bin"))
        if not self.files:
            raise RuntimeError(f"No graphs in {folder}")
        self.pos_mode, self.area_mode = pos_mode, area_mode
        self.roi_sizes, self.area_stats = roi_sizes or {}, area_stats
        if pos_mode == "normalised":
            missing = [f.stem for f in self.files if f.stem not in self.roi_sizes]
            if missing:
                raise RuntimeError(f"ROI size unknown for {len(missing)} graphs in {folder}, e.g. {missing[:3]}")
        if area_mode == "zlog" and area_stats is None:
            raise ValueError("area_mode='zlog' needs training-set area statistics")
        self.labels = [BRACS_TOKEN_TO_IDX[f.stem.split("_")[2]] for f in self.files]
        cnt = Counter(self.labels)
        print(f"[BRACS] {folder}: {len(self.files)} graphs  " +
              "  ".join(f"{c}={cnt.get(i, 0)}" for i, c in enumerate(BRACS_CLASSES)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f, label = self.files[idx], self.labels[idx]
        g, ld = _load_dgl(f)
        stored = int(ld["label"].view(-1)[0]) if "label" in ld else label
        if stored != label:
            raise ValueError(f"{f.name}: label stored in file ({stored}) != label in file name ({label})")
        side = max(self.roi_sizes[f.stem]) if self.pos_mode == "normalised" else None
        return bracs_graph_to_data(g, label, f.name, self.pos_mode, self.area_mode, side, self.area_stats)

    def get_labels(self) -> List[int]:
        return list(self.labels)

    def get_ids(self) -> List[str]:
        return [f.stem for f in self.files]


def load_area_stats(train_dir: str) -> dict:
    """log-area mean/std of the TRAINING graphs (only for the pre-revision area_mode='zlog')."""
    stats_file = Path(train_dir).parent / "area_stats_train.json"
    if stats_file.exists():
        return json.load(open(stats_file))
    files = sorted(Path(train_dir).glob("BRACS_*.bin"))
    m, s = _log_area_stats(files)
    stats = {"mean": m, "std": s, "n_graphs": len(files)}
    json.dump(stats, open(stats_file, "w"), indent=2)
    return stats


def load_roi_sizes(graph_root: str, image_dir: str, stems: List[str]) -> dict:
    """{ROI name: [width, height]} in pixels, read from the image headers once and cached as JSON.

    Images are searched recursively in image_dir as <ROI name>.png (BRACS release layout:
    BRACS_RoI/latest_version/<split>/<class>/<ROI name>.png).
    """
    cache = Path(graph_root) / "roi_image_sizes.json"
    sizes = json.load(open(cache)) if cache.exists() else {}
    todo = [s for s in stems if s not in sizes]
    if todo:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None                            # BRACS ROIs can be very large
        found = {p.stem: p for ext in ("png", "jpg", "tif", "tiff") for p in Path(image_dir).rglob(f"BRACS_*.{ext}")}
        missing = [s for s in todo if s not in found]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ROI images not found under {image_dir} "
                                    f"(e.g. {missing[:3]}); set data.bracs_image_dir, or data.bracs_pos=mm")
        for s in todo:
            with Image.open(found[s]) as im:                     # reads the header only
                sizes[s] = list(im.size)                         # (width, height)
        json.dump(sizes, open(cache, "w"), indent=1)
        print(f"[BRACS] cached ROI sizes of {len(sizes)} images in {cache}")
    return sizes


# ═════════════════════════════════════════════════════════════════════════════
# ISIC 2018
# ═════════════════════════════════════════════════════════════════════════════

_FEATURE_EXTRACTOR = None
_FEATURE_DEVICE = None
_FIXED_PROJ: dict = {}


def _get_feature_extractor(device):
    """ImageNet-pretrained ResNet-50 without the classification layer (singleton)."""
    global _FEATURE_EXTRACTOR, _FEATURE_DEVICE
    if _FEATURE_EXTRACTOR is None or _FEATURE_DEVICE != device:
        import torchvision.models as models
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:                                  # torchvision < 0.13
            resnet = models.resnet50(pretrained=True)
        _FEATURE_EXTRACTOR = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
        _FEATURE_DEVICE = device
    return _FEATURE_EXTRACTOR


def _get_fixed_projection(feature_dim: int = 512) -> np.ndarray:
    """One random 2048 -> feature_dim matrix (seed 42) shared by ALL images."""
    if feature_dim not in _FIXED_PROJ:
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((2048, feature_dim)).astype(np.float32)
        _FIXED_PROJ[feature_dim] = proj / np.sqrt(feature_dim)
    return _FIXED_PROJ[feature_dim]


def image_to_graph(image_path: str, label: int, device, n_segments: int = 75, compactness: float = 10.0,
                   k_neighbours: int = 6, patch_size: int = 64, feature_dim: int = 512) -> Data:
    """
    1. read image, resize to 224x224
    2. SLIC superpixels -> N regions
    3. centroid, area and solidity (= area / convex-hull area) of every region
    4. 64x64 crop around every centroid, resized to 224x224 -> ResNet-50 -> 2048-d
    5. fixed shared random projection 2048 -> 512
    6. undirected, duplicate-free k-NN graph on the centroids
    7. morph = [degree / max degree, area / max area, convexity (= solidity)]
    """
    import cv2
    import torchvision.transforms as T
    from skimage.measure import regionprops
    from skimage.segmentation import slic
    from sklearn.neighbors import NearestNeighbors

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224))

    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    props = regionprops(segments + 1)
    N = len(props)
    if N < 2:
        raise ValueError(f"Only {N} superpixel(s) in {image_path}")
    centroids = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=np.float32)   # (col, row)
    areas = np.array([p.area for p in props], dtype=np.float32)
    solidity = np.array([p.solidity for p in props], dtype=np.float32)
    pos = centroids / 224.0

    prep = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    half, patches = patch_size // 2, []
    for col, row in centroids:
        r0, c0 = max(0, int(row) - half), max(0, int(col) - half)
        patch = img[r0:min(224, r0 + patch_size), c0:min(224, c0 + patch_size)]
        if patch.size == 0:
            patch = img[:patch_size, :patch_size]
        patches.append(prep(cv2.resize(patch, (224, 224))))
    with torch.no_grad():
        feats = _get_feature_extractor(device)(torch.stack(patches).to(device))
    feats = feats.squeeze(-1).squeeze(-1).cpu().numpy() @ _get_fixed_projection(feature_dim)

    k = min(k_neighbours, N - 1)
    _, idxs = NearestNeighbors(n_neighbors=k + 1).fit(pos).kneighbors(pos)
    src = np.repeat(np.arange(N), k)
    dst = idxs[:, 1:].reshape(-1)
    edge_index = to_undirected(torch.tensor(np.stack([src, dst]), dtype=torch.long), num_nodes=N)

    deg = torch.zeros(N).scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    area_t = torch.tensor(areas)
    morph = torch.stack([deg / deg.max().clamp(min=1.0), area_t / area_t.max().clamp(min=1.0),
                         torch.tensor(solidity)], dim=1)
    return Data(x=torch.tensor(feats, dtype=torch.float32), edge_index=edge_index,
                pos=torch.tensor(pos, dtype=torch.float32), morph=morph,
                y=torch.tensor([label], dtype=torch.long), num_nodes=N)


class ISICDataset(Dataset):
    """
    ISIC 2018 Task 3 with a lesion-level split read from `split_csv` (columns: image, split).
    Graphs are built on first use and cached; build them beforehand with preprocessing/preprocess_isic.py
    (DataLoader workers cannot use the GPU, so a missing cache entry inside a worker raises an error).
    """

    def __init__(self, image_dir, annotations_csv, split_csv, split, cache_dir,
                 device=torch.device("cpu"), n_segments=75, k_neighbours=6):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split, self.device = split, device
        self.n_segments, self.k_neighbours = n_segments, k_neighbours
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        with open(annotations_csv.strip(), newline="") as f:
            for row in csv.DictReader(f):
                img_id = row["image"].strip()
                pos = [c for c in ISIC_CLASSES if int(float(row[c])) == 1]
                if len(pos) != 1:
                    raise ValueError(f"{img_id}: expected exactly one positive class, got {pos}")
                for ext in (".jpg", ".JPG", ".jpeg", ".png"):
                    p = Path(image_dir) / f"{img_id}{ext}"
                    if p.exists():
                        samples.append((str(p), ISIC_CLASS_TO_IDX[pos[0]]))
                        break
                else:
                    raise FileNotFoundError(f"Image not found for {img_id}")

        with open(split_csv, newline="") as f:
            split_of = {r["image"].strip(): r["split"].strip() for r in csv.DictReader(f)}
        missing = [s for s in samples if Path(s[0]).stem not in split_of]
        if missing:
            raise ValueError(f"{len(missing)} images missing from {split_csv}")
        self.samples = [s for s in samples if split_of[Path(s[0]).stem] == split]

        n_missing = sum(not self._cache_path(p).exists() for p, _ in self.samples)
        if n_missing:
            print(f"  WARNING: {n_missing}/{len(self.samples)} graphs not cached. "
                  f"Run: python preprocessing/preprocess_isic.py")
        cnt = Counter(l for _, l in self.samples)
        print(f"[ISIC:{split}] {len(self.samples)} images  split={Path(split_csv).name}  graphs={GRAPH_VERSION}  " +
              "  ".join(f"{c}={cnt.get(i, 0)}" for i, c in enumerate(ISIC_CLASSES)))

    def _cache_path(self, img_path: str) -> Path:
        key = hashlib.md5(f"{img_path}_{self.n_segments}_{self.k_neighbours}_{GRAPH_VERSION}".encode()
                          ).hexdigest()[:12]
        return self.cache_dir / f"{Path(img_path).stem}_{key}.pt"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        cache = self._cache_path(img_path)
        if cache.exists():
            data = torch.load(str(cache), map_location="cpu", weights_only=False)
        else:
            from torch.utils.data import get_worker_info
            if get_worker_info() is not None:
                raise RuntimeError(f"Graph cache missing for {Path(img_path).name}. "
                                   f"Run preprocessing/preprocess_isic.py (cache dir: {self.cache_dir})")
            data = image_to_graph(img_path, label, self.device, self.n_segments, k_neighbours=self.k_neighbours)
            torch.save(data, str(cache))
        data.y = torch.tensor([label], dtype=torch.long)
        return data

    def get_labels(self) -> List[int]:
        return [l for _, l in self.samples]

    def get_ids(self) -> List[str]:
        return [Path(p).stem for p, _ in self.samples]


# ═════════════════════════════════════════════════════════════════════════════
# CAMELYON (precomputed patch graphs)
# ═════════════════════════════════════════════════════════════════════════════

_ID_FIELDS = ("sample_id", "name", "filename", "file_name", "patch_id", "id", "image_id", "path")


def _load_graph_list(path: Path):
    """List of graphs from a pickle: a list/tuple of Data, or a dict holding such a list."""
    import pickle
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:                                            # saved with torch.save instead of pickle
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("graphs", "data", "data_list", "dataset"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(f"{path.name}: dict without a graph list (keys: {list(obj)[:10]})")
    if not isinstance(obj, (list, tuple)) or not obj:
        raise ValueError(f"{path.name}: expected a non-empty list of graphs, got {type(obj).__name__}")
    return list(obj)


def _field_names(g):
    k = g.keys() if callable(getattr(g, "keys", None)) else getattr(g, "keys", [])
    return list(k)


def _sample_id(g, path: Path, i: int):
    for k in _ID_FIELDS:
        v = getattr(g, k, None) if not isinstance(g, dict) else g.get(k)
        if v is not None:
            v = v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
            return str(v)
    return None


class CAMELYONDataset(Dataset):
    """
    Precomputed CAMELYON patch graphs of one split, used exactly as stored.

    Every graph must carry x (N, 512), edge_index, pos (N, 2), morph (N, 3) and y (0 = Normal, 1 = Tumor).
    Missing fields raise an error instead of being filled in. sample_id is kept for the prediction export;
    graphs without an identifier field get "<split>_<index>" (index = position in the pickle) and a warning.
    """

    REQUIRED = ("x", "edge_index", "pos", "morph", "y")

    def __init__(self, root, split: str):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        path = Path(root) / f"{split}_graphs.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (data.camelyon_graph_dir = {root})")
        raw = _load_graph_list(path)
        self.graphs, self.ids, self.labels, generated = [], [], [], 0
        for i, g in enumerate(raw):
            if isinstance(g, dict):
                g = Data(**g)
            missing = [k for k in self.REQUIRED if getattr(g, k, None) is None]
            if missing:
                raise ValueError(f"{path.name} graph {i}: missing field(s) {missing}; stored fields: {_field_names(g)}")
            x, ei, pos, morph = (torch.as_tensor(getattr(g, k)) for k in ("x", "edge_index", "pos", "morph"))
            y = int(torch.as_tensor(g.y).view(-1)[0])
            if y not in (0, 1):
                raise ValueError(f"{path.name} graph {i}: label {y} is not 0 (Normal) or 1 (Tumor)")
            if x.size(1) != 512:
                raise ValueError(f"{path.name} graph {i}: node features have {x.size(1)} dims, expected 512")
            if morph.dim() != 2 or morph.size(1) != 3 or morph.size(0) != x.size(0):
                raise ValueError(f"{path.name} graph {i}: morph has shape {tuple(morph.shape)}, expected ({x.size(0)}, 3)")
            if pos.dim() != 2 or pos.size(0) != x.size(0):
                raise ValueError(f"{path.name} graph {i}: pos has shape {tuple(pos.shape)}, expected ({x.size(0)}, 2)")
            sid = _sample_id(g, path, i)
            if sid is None:
                sid, generated = f"{split}_{i:06d}", generated + 1
            # keep only the fields the model reads (other attributes, e.g. strings, would break batching)
            self.graphs.append(Data(x=x.float(), edge_index=ei.long(), pos=pos.float(), morph=morph.float(),
                                    y=torch.tensor([y], dtype=torch.long), num_nodes=x.size(0)))
            self.ids.append(sid)
            self.labels.append(y)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{path.name}: sample identifiers are not unique")
        if generated:
            print(f"  WARNING: {generated} graphs in {path.name} have no identifier field; "
                  f"using {split}_<index> (position in the pickle)")
        cnt = Counter(self.labels)
        nodes = [g.num_nodes for g in self.graphs]
        print(f"[CAMELYON:{split}] {len(self.graphs)} graphs  Normal={cnt.get(0, 0)}  Tumor={cnt.get(1, 0)}  "
              f"nodes/graph {min(nodes)}-{max(nodes)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_labels(self) -> List[int]:
        return list(self.labels)

    def get_ids(self) -> List[str]:
        return list(self.ids)


def check_split_overlap(datasets):
    """Raise if any sample identifier occurs in more than one split."""
    names = ("train", "val", "test")
    for a in range(3):
        for b in range(a + 1, 3):
            common = set(datasets[a].get_ids()) & set(datasets[b].get_ids())
            if common:
                raise ValueError(f"{len(common)} sample ids in both {names[a]} and {names[b]}, e.g. {sorted(common)[:3]}")


# ═════════════════════════════════════════════════════════════════════════════
# Loaders
# ═════════════════════════════════════════════════════════════════════════════

def build_datasets(cfg):
    d = cfg.data
    if cfg.dataset == "bracs":
        root = Path(d.bracs_dir)
        dirs = [root / f"{s}_official" for s in ("train", "val", "test")]
        stats = sizes = None
        if d.bracs_area == "zlog":
            stats = load_area_stats(str(dirs[0]))
            print(f"[BRACS] log-area mean={stats['mean']:.4f} std={stats['std']:.4f} (training graphs)")
        if d.bracs_pos == "normalised":
            stems = [f.stem for dd in dirs for f in dd.glob("BRACS_*.bin")]
            sizes = load_roi_sizes(str(root), d.bracs_image_dir, stems)
        print(f"[BRACS] pos={d.bracs_pos} area={d.bracs_area}")
        return tuple(BRACSDataset(str(dd), d.bracs_pos, d.bracs_area, sizes, stats) for dd in dirs)
    if cfg.dataset in ("camelyon", "camelyon16"):
        root = Path(d.camelyon_graph_dir)
        sets = tuple(CAMELYONDataset(root, split) for split in ("train", "val", "test"))
        check_split_overlap(sets)
        return sets
    feat_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tuple(ISICDataset(d.isic_image_dir, d.isic_annotations, d.isic_split_csv, s, d.isic_cache_dir,
                             feat_device, d.n_segments, d.k_neighbours) for s in ("train", "val", "test"))


def build_dataloaders(cfg):
    """Returns (train_loader, val_loader, test_loader). The training sampler is seeded with cfg.seed."""
    train_ds, val_ds, test_ds = build_datasets(cfg)
    t = cfg.train
    sampler = None
    if t.weighted_sampling:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        w = compute_sample_weights(train_ds.get_labels(), len(cfg.class_names))
        sampler = WeightedRandomSampler(w, len(w), replacement=True, generator=g)
    kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=t.num_workers > 0,
              persistent_workers=t.num_workers > 0)
    train = GeoDataLoader(train_ds, sampler=sampler, shuffle=sampler is None, **kw)
    val = GeoDataLoader(val_ds, shuffle=False, **kw)
    test = GeoDataLoader(test_ds, shuffle=False, **kw)
    print(f"[DataLoaders] {cfg.dataset} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train, val, test
