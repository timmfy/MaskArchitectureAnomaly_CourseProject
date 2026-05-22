import os
import csv
import torch
from eomt_tools import eomt_setup, eomt_inference


# Define Cityscapes classes for shared use
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

def evaluate_model(label, weights, cfg_path, data_path, device, kind, limit, ignore_index=255):
    """
    Evaluates a single model on Cityscapes validation set.
    Returns: {"model", "weights", "mIoU", "per_class"}
    """
    num_classes_to_report = 19
    model = None

    from eval import eval_iou
    if kind == "erfnet":
        miou, ious = eval_iou.evaluate_erfnet(
            weightsPath=weights,
            datadir=data_path,
            limit=limit,
            ignore_index=ignore_index if ignore_index != 255 else 19,
        )
    else:
        cfg = eomt_setup.load_config(cfg_path)
        data = eomt_setup.setup_data(cfg, data_path=data_path)
        
        is_coco = (kind == "eomt_coco")
        
        if is_coco and device.type == 'mps':
            # Load on CPU first then move to GPU (MPS framework doesn't support float64)
            model = eomt_setup.load_model(cfg, data, torch.device('cpu'), weights_path=weights)
            model = model.to(device=device, dtype=torch.float32)
        else:
            model = eomt_setup.load_model(cfg, data, device, weights_path=weights)
            
        miou, ious = eomt_inference.evaluate_semantic(
            model, data.val_dataloader(), device, data.img_size,
            num_classes=num_classes_to_report, ignore_index=ignore_index, is_coco=is_coco, limit=limit
        )

    result = {
        'model': label,
        'weights': os.path.basename(weights),
        'mIoU': round(float(miou), 4) if miou is not None else 0.0,
        'per_class': [round(float(v), 4) for v in ious[:num_classes_to_report]] if ious is not None else []
    }
    
    # Clean up
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result

def write_results_csv(results, csv_path, class_names):
    """Writes evaluation results to a CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Weights', 'mIoU'] + class_names)
        for r in results:
            writer.writerow([r['model'], r['weights'], r['mIoU']] + r.get('per_class', []))
    print(f'Saved: {csv_path}\n')

def print_results_summary(results, class_names):
    """Prints a summary table and per-class breakdown of results."""
    # Summary table
    print(f'{"Model":<35} {"mIoU":>6}')
    print('-' * 45)
    for r in results:
        print(f"{r['model']:<35} {r['mIoU']:>6.4f}")

    # Per-class breakdown
    print(f'\n{"":35}  ' + '  '.join(f'{c[:5]:>5}' for c in class_names))
    for r in results:
        vals = '  '.join(f'{v:5.3f}' for v in (r.get('per_class') or []))
        print(f"{r['model']:<35}  {vals}")
