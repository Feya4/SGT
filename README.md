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
├── train.py                  # training + single final test evaluation
├── profile_graphs.py         # graph sizes, parameters, GPU memory, latency
├── config/config.py          # all hyper-parameters and paths (presets per dataset)
├── dataset/dataset.py        # BRACS and ISIC 2018 graph datasets, data loaders
├── model/model.py            # SGT: MASE, gated fusion, SRE attention, classifier
├── utils/utils.py            # seeding, losses, sampling weights, control experiments, metrics
├── utils/checkpoint.py       # rebuild model + config from a checkpoint
├── preprocessing/            # graph construction and data splits (run once, in the order below)
│   ├── organize_bracs_official.py
│   ├── build_bracs_graphs.py
│   ├── compute_morphology.py
│   ├── make_isic_lesion_split.py
│   ├── inspect_pairs.py
│   └── preprocess_isic.py
└── data/                     # datasets only (not code, not committed)
```

Each code folder has an `__init__.py`, so imports such as `from config import get_config` work unchanged. Run all commands from the `sgt-pathology/` folder.

## Installation

```bash
conda create -n sgt python=3.10 -y && conda activate sgt
# PyTorch: pick the build for your CUDA version from https://pytorch.org
pip install torch torchvision
pip install torch_geometric torch_scatter          # torch_scatter must match your torch/CUDA build
pip install dgl                                    # reads the BRACS tissue graphs (.bin)
pip install numpy pandas scikit-learn scikit-image opencv-python pillow imagehash
pip install histocartography                       # preprocessing of BRACS only
```

(or `pip install -r requirements.txt` after installing PyTorch).

Experiments in the paper were run on NVIDIA A40 GPUs.

## Data

All paths are relative to the environment variable `SGT_ROOT`, which defaults to the current folder. Every path can be changed in `config/config.py` or overridden with `--set`.

### BRACS (7 classes: N, PB, UDH, ADH, FEA, DCIS, IC)

Download the BRACS ROI images (<https://www.bracs.icar.cnr.it/>) into `data/{train,val,test}/<class>/`, keeping the official folder structure, which defines the patient-level split. Then run:

```bash
# 1. copy existing tissue graphs into folders that follow the official split; audit slide overlap
python preprocessing/organize_bracs_official.py --dry_run
python preprocessing/organize_bracs_official.py        # -> data/tissue_graphs_official/, missing_graphs_*.txt

# 2. build graphs for ROIs that have none (HACT pipeline: Vahadane + merged SLIC + ResNet-34 + RAG)
python preprocessing/build_bracs_graphs.py --verify 5  # rebuild 5 existing graphs, should be identical
python preprocessing/build_bracs_graphs.py --split test

# 3. add region area and convexity (solidity) to every node
python preprocessing/compute_morphology.py --check 5
python preprocessing/compute_morphology.py --workers 8 # -> data/tissue_graphs_official_morph/
```

The resulting layout is:

```
data/{train,val,test}/<class>/.../BRACS_*.png                  # ROI images
data/tissue_graphs_official_morph/{train,val,test}_official/   # graphs used for training
```

- Node features are 512-d ResNet-34 features; the two location features of the RAG builder are dropped.
- `compute_morphology.py` checks that every region matches a graph node (same count, centroids within 2 px) and that the features are 512-dimensional. It lists any graph that fails in `morphology_problems.txt`.
- The ROI sizes needed to normalise centroids to [0, 1] are read from the image headers once and cached in `roi_image_sizes.json`.
- Eight training ROIs (7 UDH, 1 IC) were unavailable in our copy, so training uses 3,155 ROIs; validation (602) and test (626) are complete.

### ISIC 2018 Task 3 / HAM10000 (7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC)

```
data/ISIC2018_Task3_Training_Input/*.jpg
data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv
```

```bash
# lesion-level 70/15/15 split, stratified by diagnosis, seed 42; near-duplicates (pHash Hamming <= 2) grouped
python preprocessing/make_isic_lesion_split.py \
    --gt     data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv \
    --groups data/ISIC2018_Task3_Training_LesionGroupings.csv \
    --img_dir data/ISIC2018_Task3_Training_Input          # -> splits/isic2018_lesion_split.csv

# optional: review candidate pairs to choose the threshold
python preprocessing/inspect_pairs.py --img_dir data/ISIC2018_Task3_Training_Input

# build and cache all graphs (GPU recommended) -> data/graph_cache/isic/
python preprocessing/preprocess_isic.py
```

Each image is turned into a graph as follows:

1. The image is resized to 224×224 and segmented into about 75 SLIC superpixels (compactness 10).
2. Each superpixel becomes a node, described by ImageNet ResNet-50 features of a 64×64 crop at its centroid, reduced to 512 dimensions by one fixed random projection (seed 42).
3. Nodes are connected in an undirected 6-nearest-neighbour graph.

Graphs must be cached before training, because DataLoader workers cannot use the GPU.

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

This reports the number of nodes and edges per graph, the parameter count, peak GPU memory, latency at batch size 1 and throughput. Data loading and feature extraction are not included.


## Reproducing runs made with the pre-revision code

The implementation was revised in October 2026 to match the manuscript exactly:

- the gate uses m^(ℓ−1);
- stochastic depth is applied per graph;
- BRACS centroids are normalised by the ROI side;
- BRACS area is normalised by the maximum area in the graph.

Checkpoints store the configuration they were trained with:

- `utils.load_model` and `profile_graphs.py` recognise checkpoints from the earlier code and rebuild them with the earlier settings automatically.
- `train.py --test_only` refuses to evaluate a checkpoint with different settings, and prints the `--set` values to use.
- To train or evaluate with the earlier behaviour, add:

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
