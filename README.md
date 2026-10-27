# SGT: Structural-Morphology-Aware Graph Transformer

Repository: https://github.com/Feya4/SGT · 
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
├── run_controls.py           # control experiments of the revision (5 seeds, Table 7 settings)
├── summarize_controls.py     # control table, paired t-test and bootstrap vs SGT, settings check
├── build_reproducibility.py  # builds reproducibility/ (split files + per-sample test predictions)
├── list_results.py           # mean ± SD of every run group in outputs/
├── reproducibility/          # split files and per-sample test predictions of every reported run
├── results/isic2018/       
├── preprocessing/            # graph construction and data splits (run once, in the order below)
│   ├── organize_bracs_official.py
│   ├── build_bracs_graphs.py
│   ├── compute_morphology.py
│   ├── make_isic_lesion_split.py
│   ├── inspect_pairs.py
│   └── preprocess_isic.py
├── extra/bach/               # optional BACH graphs (not part of the paper)
├── assets/target.png         # stain-normalisation target (add it, see assets/README.md)
├── splits/                   # isic2018_lesion_split.csv
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


**Protocol.**

- Training uses the training split only.
- Weighted F1 on the validation split is computed after every epoch, and the checkpoint with the highest value is kept.
- Training stops early after `patience` epochs without improvement.
- The test split is evaluated **once**, with that checkpoint.

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
| CAMELYON16 | weighted F1  |89.18 | 88.11 |
| ISIC 2018 (lesion split) | weighted F1 | 80.57 | 75.80 |


## Citation

```bibtex
@article{akmel2026sgt,
  title   = {Structural-Morphology-Aware Graph Transformer for Fine-Grained Digital Pathology Image Classification},
  author  = {Akmel, Feidu and Gong, Xun and Ibrahim, Ashraf Osman and Alemu, Gedamu and Hadabi, Abdalla and Naouri, Abdenacer},
  journal = {(under review)},
  year    = {2026}
}
```

Software: Akmel F. et al., *SGT code, splits and test predictions*, Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX (see `CITATION.cff`).

## License

MIT (see `LICENSE`).

## Acknowledgements

- Tissue graphs for BRACS come from the HACT preprocessing (Pati et al.).
- The CGT baseline uses its authors' released implementation (Wang et al.).
- Datasets: BRACS, CAMELYON16, ISIC 2018 / HAM10000.

## Contact

Feidu Akmel (feiduakmel@gmail.com) or Xun Gong (xgong@swjtu.edu.cn).
