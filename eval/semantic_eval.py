import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from eomt_tools import eomt_setup, eomt_inference

# Define Cityscapes classes for shared use
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

def load_my_state_dict(model, state_dict):
    """Custom function to load ERFNet weights, handling 'module.' prefix."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                key = name.split("module.")[-1]
                if key in own_state:
                    own_state[key].copy_(param)
                else:
                    continue
            else:
                continue
        else:
            own_state[name].copy_(param)
    return model

def evaluate_model(label, weights, cfg_path, data_path, device, kind, limit, ignore_index=255):
    """
    Evaluates a single model on Cityscapes validation set.
    Returns: {"model", "weights", "mIoU", "per_class"}
    """
    print(f"Evaluating {label} ({kind}) ...")
    
    # Import here to avoid circular dependencies or unnecessary loads
    from eval import erfnet, iouEval, eval_cityscapes_color
    from eval.dataset import cityscapes
    from eval.eval_cityscapes_color import input_transform_cityscapes, target_transform_cityscapes, evaluate_erfnet

    if kind == "erfnet":
        num_classes = 20 # As in eval_cityscapes_color.py
        model = erfnet.ERFNet(num_classes)
        # Load weights
        state_dict = torch.load(weights, map_location='cpu', weights_only=False)
        model = load_my_state_dict(model, state_dict)
        model = model.to(device)
        
        loader = torch.utils.data.DataLoader(
            cityscapes(data_path, input_transform_cityscapes, target_transform_cityscapes, subset='val'),
            batch_size=1, shuffle=False, num_workers=4
        )
        
        miou, ious = evaluate_erfnet(model, loader, device, limit=limit, save_dir=None)
        num_classes_to_report = 19
    else:
        num_classes_to_report = 19
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
