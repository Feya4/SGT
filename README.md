# SGT: Structural-Morphology-Aware Graph Transformer

Official PyTorch implementation of

> **Structural-Morphology-Aware Graph Transformer for Fine-Grained Digital Pathology Image Classification**

SGT classifies histopathology (and dermoscopy) images from their tissue graphs. It extends a sparse graph transformer with two components:

- **MASE (Morphology-Aware Structural Encoder).** It encodes the degree, area and convexity of each tissue region in a separate morphological token stream. This stream is updated by neighbourhood aggregation at every layer and fused into the visual stream through a learned per-dimension gate.
- **SRE (Spatial Relationship Encoder).** It adds a learnable distance bias, β·φᵢⱼ with φᵢⱼ = 1 / (1 + ‖pᵢ − pⱼ‖²), to the attention logits of adjacent regions. Among a region's neighbours, closer regions therefore receive relatively more attention.

---

## Repository layout

```
sgt-pathology/
├── train.py              # training + single final test evaluation
├── profile_graphs.py     # graph sizes, parameters, GPU memory, latency
├── config/config.py      # all hyper-parameters and paths (presets per dataset)
├── dataset/dataset.py    # BRACS and ISIC 2018 graph datasets, data loaders
├── model/model.py        # SGT: MASE, gated fusion, SRE attention, classifier
├── utils/utils.py        # seeding, losses, sampling weights, control experiments, metrics
└── data/                 # datasets only (not code), see below
```

Each code folder has an `__init__.py`, so imports such as `from config import get_config` work unchanged. Run all commands from the `sgt-pathology/` folder.

## Installation

```bash
conda create -n sgt python=3.10 -y && conda activate sgt
# PyTorch: pick the build for your CUDA version from https://pytorch.org
pip install torch torchvision
pip install torch_geometric torch_scatter          # torch_scatter must match your torch/CUDA build
pip install dgl                                    # reads the BRACS tissue graphs (.bin)
pip install numpy scikit-learn scikit-image opencv-python pillow
```

Experiments in the paper were run on NVIDIA A40 GPUs.

## Data

All paths are relative to the environment variable `SGT_ROOT`, which defaults to the current folder. Every path can be changed in `config/config.py` or overridden with `--set`.

### BRACS (7 classes: N, PB, UDH, ADH, FEA, DCIS, IC)

1. Download BRACS and its official HACT tissue graphs from <https://www.bracs.icar.cnr.it/>.
2. Add the node fields `area` and `solidity` (convexity) to every graph with `preprocessing/compute_morphology.py` (not yet included in this repository).
3. Arrange the folders as follows:

```
data/tissue_graphs_official_morph/{train,val,test}_official/BRACS_*.bin
data/BRACS_RoI/...  /BRACS_*.png       # ROI images; only width/height are read
```

The ROI sizes are needed to normalise the node centroids to [0, 1] by the longer ROI side. They are read once from the image headers and cached in `roi_image_sizes.json`.

We use the official patient-level split. Eight training ROIs (7 UDH, 1 IC) were unavailable in our copy, so training uses 3,155 ROIs; validation (602) and test (626) are complete.

### ISIC 2018 Task 3 / HAM10000 (7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC)

```
data/ISIC2018_Task3_Training_Input/*.jpg
data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv
splits/isic2018_lesion_split.csv           # columns: image, split (train/val/test)
```

The split is at the lesion level (70/15/15, stratified by diagnosis, seed 42), and near-duplicate images (perceptual hash, Hamming distance ≤ 2) are kept in the same partition.

Each image is turned into a graph as follows:

1. The image is resized to 224×224 and segmented into about 75 SLIC superpixels.
2. Each superpixel is described by ResNet-50 features of a 64×64 crop, reduced to 512 dimensions by a fixed random projection (seed 42).
3. Superpixels are connected in a 6-nearest-neighbour graph.

Graphs are cached in `data/graph_cache/isic/`. Build them beforehand with `preprocessing/preprocess_isic.py` (not yet included), because DataLoader workers cannot use the GPU.

### CAMELYON16 (2 classes: normal, tumor)

The training settings are in the `camelyon16` preset. **The WSI graph-building code and data loader are not yet included.**

## Training and evaluation

```bash
python train.py --dataset bracs --seed 0 --exp_name sgt_bracs_s0
python train.py --dataset isic  --seed 0 --exp_name sgt_isic_s0

# re-evaluate a trained model (best.pt) on the test split
python train.py --dataset bracs --exp_name sgt_bracs_s0 --test_only

# override any config field
python train.py --dataset bracs --exp_name x --set train.lr=1e-4 model.dropout=0.2
```

The paper reports mean ± SD over seeds 0–4:

```bash
for s in 0 1 2 3 4; do python train.py --dataset bracs --seed $s --exp_name sgt_bracs_s$s; done
```

**Protocol.**

- Training uses the training split only.
- Weighted F1 on the validation split is computed after every epoch, and the checkpoint with the highest value is kept.
- Training stops early after `patience` epochs without improvement.
- The test split is evaluated **once**, with that checkpoint.

Outputs:

| File | Content |
|---|---|
| `checkpoints/<exp>/best.pt` | best-validation checkpoint (`latest.pt` for `--resume`) |
| `outputs/<exp>/train.log` | per-epoch log |
| `outputs/<exp>/report/test_report.json` | metrics (accuracy, weighted/macro F1, macro OvR AUC, per class, confusion matrix) and the full config |
| `outputs/<exp>/report/{val,test}_predictions.npz` | sample id, label, prediction, softmax probabilities |

### Ablations and control experiments

```bash
python train.py --dataset bracs --exp_name abl_noMASE --set model.use_mase=false   # remove MASE
python train.py --dataset bracs --exp_name abl_noSRE  --set model.use_sre=false    # remove SRE
python train.py --dataset bracs --exp_name ctl_shuffle --morph_mode shuffle        # permute morphology within each graph
python train.py --dataset bracs --exp_name ctl_pos     --pos_mode shuffle          # permute centroids within each graph
```

`--morph_mode` accepts `real`, `shuffle`, `degree` (area and convexity set to 0) or `none` (all attributes 0, but the stream is kept). To remove the MASE module itself, use `model.use_mase=false`.

### Profiling

```bash
python profile_graphs.py --dataset bracs --checkpoint checkpoints/sgt_bracs_s0/best.pt
```

This reports the number of nodes and edges per graph, the parameter count, peak GPU memory, latency at batch size 1 and throughput. Data loading and feature extraction are not included. *Note: this script imports `load_model` from `eval_bach.py`, which is not yet included.*

## Hyper-parameters

| | BRACS | ISIC 2018 | CAMELYON16 |
|---|---|---|---|
| Layers / heads / hidden / FFN | 6 / 8 / 256 / 512 | same | same |
| Morph. MLP width, β init | 64, 1.0 | same | same |
| Dropout / classifier width | 0.1 / 64 | 0.5 / 128 | 0.1 / 128 |
| Stochastic depth | linear 0 → 0.15, per graph | same | same |
| Optimiser | AdamW, lr 5e-5, wd 1e-4 | AdamW, lr 3e-5, wd 5e-4 | AdamW, lr 1e-4, wd 1e-4 |
| Batch / epochs / warm-up / patience | 16 / 200 / 10 / 30 | 16 / 200 / 10 / 30 | 32 / 100 / 5 / 20 |
| Loss | focal (γ = 2) | CE + label smoothing 0.1 | focal (γ = 2) + class weights |
| Class imbalance | weighted sampling | weighted sampling | weighted sampling |
| Augmentation | none | feature noise σ 0.02, edge dropout 0.10 | none |
| Parameters | 4.90 M | 4.92 M | – |

All runs use cosine decay to 1e-6 after a linear warm-up and gradient clipping at 1.0.

## Reproducing runs made with the pre-revision code

The implementation was revised in October 2026 to match the manuscript exactly:

- the gate uses m^(ℓ−1);
- stochastic depth is applied per graph;
- BRACS centroids are normalised by the ROI side;
- BRACS area is normalised by the maximum area in the graph.

To reproduce or re-evaluate checkpoints trained with the earlier code, add:

```bash
--set model.fuse_prev_morph=false model.drop_path_per_graph=false data.bracs_pos=mm data.bracs_area=zlog
```

## Results (manuscript)

| Dataset | Metric | SGT | CGT (same protocol) |
|---|---|---|---|
| BRACS (7-class) | weighted F1 | 67.61 | 66.67 |
| CAMELYON16 | accuracy / AUC | 92.77 / 94.81 | – |
| ISIC 2018 (lesion split) | weighted F1 | 80.57 | 75.80 |

For BRACS, the SGT−CGT difference is +0.94 percentage points (95% CI 0.33–1.55, p = 0.008). Results are means over five seeds.

## Citation

```bibtex
@article{akmel2026sgt,
  title   = {Structural-Morphology-Aware Graph Transformer for Fine-Grained Digital Pathology Image Classification},
  author  = {Akmel, Feidu and Gong, Xun and Ibrahim, Ashraf Osman and Alemu, Gedamu and Hadabi, Abdalla and Naouri, Abdenacer},
  journal = {(under review)},
  year    = {2026}
}
```

## Acknowledgements

- Tissue graphs for BRACS come from the HACT preprocessing (Pati et al.).
- The CGT baseline uses its authors' released implementation (Wang et al.).
- Datasets: BRACS, CAMELYON16, ISIC 2018 / HAM10000.

## Contact

Feidu Akmel (feiduakmel4@gmail.com) or Xun Gong (xgong@swjtu.edu.cn).
