# Comprehensive Road Scene Understanding for Autonomous Driving

Course project exploring the differences between the segmentation architectures and evaluating their perception capabilities in safety-sensitive applications, such as autonomous driving

Two tracks:
- **Semantic segmentation** — ERFNet baseline vs EoMT (Cityscapes pretrained, COCO pretrained, three finetuned variants)
- **Anomaly segmentation** — OoD detection on road-scene datasets (RoadObstacle21, LostFound, RoadAnomaly, …)

---

## Quick Start

### Local

```bash
git clone https://github.com/timmfy/MaskArchitectureAnomaly_CourseProject.git
cd MaskArchitectureAnomaly_CourseProject
pip install -r requirements.txt
```

Open `semantic_segmentation.ipynb` or `anomaly_segmentation.ipynb` in Jupyter, edit the **path config cell** at the top, then run all cells.

### Google Colab

Each notebook auto-detects Colab, clones the repo, installs dependencies, and mounts Google Drive. The only cell to edit is the path config block:

```python
_DRIVE         = '/content/drive/MyDrive/CourseProject'
DATASET_DIR    = f'{_DRIVE}/CityScapesDataset'
PRETRAINED_DIR = f'{_DRIVE}/weights'
FINETUNED_DIR  = f'{_DRIVE}/checkpoints'
```

Put model weights on Google Drive and point these variables at them.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `semantic_segmentation.ipynb`  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timmfy/MaskArchitectureAnomaly_CourseProject/blob/main/semantic_segmentation.ipynb)| Demo images + quantitative comparison (mIoU) for ERFNet and all EoMT variants → `results_semantic/comparison.csv` |
| `anomaly_segmentation.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timmfy/MaskArchitectureAnomaly_CourseProject/blob/main/anomaly_segmentation.ipynb)| Demo anomaly heatmaps + model comparison (AUPRC, FPR@TPR95) + temperature scaling → `results_anomaly/` |
---

## Weights

Weights are distributed via Google Drive. Place them as follows (or update the path config cell in each notebook):

```
eomt/weights/
    eomt_cityscapes.bin       # EoMT pretrained on Cityscapes (semantic)
    eomt_coco.bin             # EoMT pretrained on COCO (panoptic)
trained_models/
    erfnet_pretrained.pth     # ERFNet baseline
    backbone_frozen_24_epochs.ckpt
    backbone_unfrozen_10_50_epochs.ckpt
    backbone_unfrozen_interpolation_10_45_epochs.ckpt
```

---

## Datasets

```
datasets/
    cityscapes/               # Cityscapes (leftImg8bit/ + gtFine/)
    Validation_Dataset/
        RoadObsticle21/       # images/ + labels_masks/
        FS_LostFound_full/
        fs_static/
        RoadAnomaly/
        RoadAnomaly21/
```

---

## Repository Structure

```
.
├── semantic_segmentation.ipynb   # Semantic segmentation demo + comparison
├── anomaly_segmentation.ipynb    # Anomaly segmentation demo + comparison
├── eomt_finetune.ipynb           # EoMT fine-tuning
├── eomt_setup.py                 # Model loading utilities
├── eomt_inference.py             # Inference + visualisation utilities
├── coco_to_cityscape.py          # COCO → Cityscapes class ID mapping
├── requirements.txt
├── eomt/
│   ├── configs/                  # YAML configs (Cityscapes semantic, COCO panoptic, finetuning)
│   ├── datasets/                 # Data modules (CityscapesSemantic, COCOPanoptic, …)
│   ├── models/                   # EoMT architecture (ViT encoder + query-based decoder)
│   ├── training/                 # Lightning modules (semantic, instance, panoptic)
│   └── main.py                   # LightningCLI entry point for training
└── eval/
    ├── evalAnomaly.py            # ERFNet anomaly eval (writes results_anomaly/results_erfnet_<ds>.csv)
    ├── evalAnomalyForEomt.py     # EoMT anomaly eval  (writes results_anomaly/results_<model>_<ds>.csv + logits/)
    ├── collect_results.py        # Runs eval scripts for all model×dataset combos → comparison.csv
    ├── eval_iou.py               # ERFNet mIoU eval on Cityscapes val
    ├── erfnet.py                 # ERFNet architecture
    ├── iouEval.py                # IoU evaluator
    ├── pathGTComparison.py       # GT mask loader for OOD datasets
    └── dataset.py                # Cityscapes dataset for ERFNet eval
```

---

## Running Evaluations from the CLI

### Anomaly comparison CSV

```bash
python eval/run_eval_anomaly.py \
  --datasets RoadObsticle21 \
  --erfnet-weights trained_models/erfnet_pretrained.pth \
  --eomt-models eomt/weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --output-csv results_anomaly/comparison.csv
```

Add `--skip-run` to consolidate already-computed per-run CSVs without re-running inference.

### Semantic comparison CSV

```bash
python eval/run_eval_semantic.py --loadDir ../trained_models/ --loadWeights erfnet_pretrained.pth \
  --datadir ../datasets/cityscapes/
```

---

## Models

### ERFNet
Real-time semantic segmentation baseline. Predicts 19 Cityscapes train classes.
- Pretrained mIoU on Cityscapes val: **54.10%**

### EoMT (Early-to-Middle Token)
Transformer-based architecture using a ViT (DINOv2) backbone with 100 learned query tokens.

| Weights | Task | mIoU (Cityscapes val) |
|---|---|---|
| `eomt_cityscapes.bin` | Semantic | — |
| `eomt_coco.bin` | Panoptic (COCO) | — |
| `backbone_frozen_24_epochs.ckpt` | Semantic (finetuned) | — |
| `backbone_unfrozen_10_50_epochs.ckpt` | Semantic (finetuned) | — |
| `backbone_unfrozen_interpolation_10_45_epochs.ckpt` | Semantic (finetuned) | — |

Run `semantic_segmentation.ipynb` Section 3 to fill in the mIoU values.

---

## Anomaly Detection Methods

Each model is evaluated with three scoring functions:

| Method | Formula |
|---|---|
| **MSP** | `1 − max softmax probability` |
| **MaxLogit** | `1 − max logit` |
| **MaxEntropy** | `Σ −p log p` |
| **RbA** (EoMT only) | `−Σ tanh(logit)` |

Metrics: **AUPRC** (higher is better) and **FPR@TPR95** (lower is better).
