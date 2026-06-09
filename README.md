# Comprehensive Road Scene Understanding for Autonomous Driving

Course project exploring the differences between the segmentation architectures and evaluating their perception capabilities in safety-sensitive applications, such as autonomous driving

Two tracks:
- **Semantic segmentation** - ERFNet baseline vs EoMT (Cityscapes pretrained, COCO pretrained, three finetuned variants)
- **Anomaly segmentation** — OoD detection on road-scene datasets (RoadObstacle21, LostFound, RoadAnomaly, …)

---

## Quick Start

### Local

```bash
git clone https://github.com/timmfy/MaskArchitectureAnomaly_CourseProject.git
cd MaskArchitectureAnomaly_CourseProject
pip install -r requirements.txt
```

Open `semantic_segmentation.ipynb` or `anomaly_segmentation.ipynb` in Jupyter, edit the path config cells at the top, then run all cells.

### Google Colab

Each notebook auto-detects Colab, clones the repo, installs dependencies, downloads model weights and mounts the Google Drive. Since this repository does not provide the datasets, the only cell to edit is the block with the path to the dataset (Cityscapes validation set for semantic segmentation or a folder with validation datasets for anomaly segmentation)

`semantic_segmentation.ipynb`

```python
_DRIVE         = '/content/drive/MyDrive/CourseProject'
DATASET_DIR    = f'{_DRIVE}/datasets/CityScapesDataset'
```

`anomaly_segmentation.ipynb`

```python
_DRIVE                  = '/content/drive/MyDrive/CourseProject'
DATASETS_ANOMALY_DIR    = f'{_DRIVE}/datasets/Validation_Dataset'
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `semantic_segmentation.ipynb`  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timmfy/MaskArchitectureAnomaly_CourseProject/blob/main/semantic_segmentation.ipynb)| Qualitative comparison with demo images + quantitative comparison (mIoU) for ERFNet and all EoMT variants: outputs `results_semantic/comparison.csv` |
| `anomaly_segmentation.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timmfy/MaskArchitectureAnomaly_CourseProject/blob/main/anomaly_segmentation.ipynb)| Demo anomaly heatmaps + model comparison (AUPRC, FPR@TPR95) + temperature scaling: outputs `results_anomaly/` |

---

## Workflow

This repository supports two evaluation workflows:
1. **Interactive (Notebooks)**: Use `semantic_segmentation.ipynb` and `anomaly_segmentation.ipynb` for step-by-step exploration, qualitative visualizations (heatmaps), and experimentation.
2. **Batch (CLI)**: Use the scripts in `eval/` for batch evaluation across multiple models and datasets, producing CSV reports.

---

## Datasets

This repository does not provide the datasets. You must download them separately and organize them as follows:

```
datasets/
├── cityscapes/
│   ├── leftImg8bit/
│   │   └── val/
│   └── gtFine/
│       └── val/
└── Validation_Dataset/
    ├── RoadObsticle21/
    │   ├── images/
    │   └── labels_masks/
    ├── RoadAnomaly/
    └── ...
```

> **Cityscapes Format**: EoMT models expect the Cityscapes dataset to be provided as Zip archives (`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`) within the `datasets/cityscapes/` folder, while the ERFNet baseline expects the files to be extracted as shown in the tree above.

---

## Weights

Weights are distributed via [Hugging Face Hub](https://huggingface.co/timmfy/comprehensive-road-scene-understanding).

### Automatic Download
The notebooks will automatically download the weights. Alternatively, you can download them via the CLI:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='timmfy/comprehensive-road-scene-understanding', local_dir='./weights', allow_patterns=['*.pth', '*.bin', '*.ckpt'])"
```

### Weights Manifest
```
weights/
    eomt_cityscapes.bin       # EoMT pretrained on Cityscapes (semantic)
    eomt_coco.bin             # EoMT pretrained on COCO (panoptic)
    eomt_coco_finetuned.ckpt  # EoMT pretrained on COCO and finetuned on Cityscapes
    erfnet_pretrained.pth     # ERFNet baseline
```

---

## Repository Structure

```
.
├── semantic_segmentation.ipynb   # Semantic segmentation demo + comparison
├── anomaly_segmentation.ipynb    # Anomaly segmentation demo + comparison
├── eval/                         # Evaluation scripts and utilites + ERFNet code
├── eomt/                         # Code from the EoMT repository
├── eomt_tools/                   # Utilities for EoMT setup and inference + COCO to Cityscapes classes mapping
```

---

## Running Evaluations from the CLI

### Anomaly comparison CSV (single model and dataset)

```bash
python eval/run_eval_anomaly.py \
  --datasets RoadObsticle21 \
  --erfnet-weights weights/erfnet_pretrained.pth \
  --eomt-cityscapes weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --output-csv results_anomaly/comparison.csv
```

### Semantic comparison CSV (single model and dataset)
Outputs the mIoU as well as the per-class mIoU for each model

```bash
python eval/run_eval_semantic.py \
  --erfnet-weights weights/erfnet_pretrained.pth \
  --eomt-cityscapes weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --dataset-dir datasets/cityscapes/
```

### Multiple Models & Datasets

You can evaluate multiple datasets or all EoMT variants at once:

**Anomaly (All models, multiple datasets):**
```bash
python eval/run_eval_anomaly.py \
  --datasets RoadObsticle21 RoadAnomaly21 RoadAnomaly \
  --erfnet-weights weights/erfnet_pretrained.pth \
  --eomt-cityscapes weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --eomt-coco weights/eomt_coco.bin:eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --eomt-finetuned weights/eomt_coco_finetuned.ckpt:eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x_finetuning.yaml
```

**Semantic (All models comparison):**
```bash
python eval/run_eval_semantic.py \
  --erfnet-weights weights/erfnet_pretrained.pth \
  --eomt-cityscapes weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --eomt-coco weights/eomt_coco.bin:eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml \
  --eomt-finetuned weights/eomt_coco_finetuned.ckpt:eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x_finetuning.yaml \
  --dataset-dir datasets/cityscapes/
```
---

## Anomaly Detection Methods

Each model is evaluated with three scoring functions:

| Method | Formula |
|---|---|
| **MSP** | $1-\max_{c} P(c\|x)$ |
| **MaxLogit** | $-\max_{c} z_c$ |
| **MaxEntropy** | $-\sum_{c} P(c\|x) \log P(c\|x)$ |
| **RbA** (EoMT only) | $-\sum_{c} \tanh(z_c)$ |

Metrics: **AUPRC** and **FPR@TPR95**.
