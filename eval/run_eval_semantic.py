#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# Ensure repo root and eval/ are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval.semantic_eval import evaluate_model, write_results_csv, print_results_summary, CITYSCAPES_CLASSES

def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation models on Cityscapes.')
    parser.add_argument('--dataset-dir', type=str, default='datasets/cityscapes', help='Path to Cityscapes dataset.')
    parser.add_argument('--erfnet-weights', type=str, help='Path to ERFNet weights.')
    parser.add_argument('--eomt-cityscapes', type=str, help='weights:config pair for EoMT Cityscapes.')
    parser.add_argument('--eomt-coco', type=str, help='weights:config pair for EoMT COCO.')
    parser.add_argument('--eomt-finetuned', type=str, help='weights:config pair for EoMT finetuned.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images for evaluation.')
    parser.add_argument('--ignore-index', type=int, default=255, help='Ignore index for evaluation.')
    parser.add_argument('--output-csv', type=str, default='results_semantic/comparison.csv', help='Output CSV path.')
    parser.add_argument('--device', type=str, help='Device to use (cuda, mps, cpu).')

    args = parser.parse_args()
    
    if not args.device:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    dataset_dir = _abs(args.dataset_dir)
    output_csv = _abs(args.output_csv)
    results = []
    
    # 1. ERFNet
    if args.erfnet_weights:
        weights = _abs(args.erfnet_weights)
        res = evaluate_model("ERFNet", weights, None, dataset_dir, device, "erfnet", args.limit, args.ignore_index)
        results.append(res)
        
    # 2. EoMT Cityscapes
    if args.eomt_cityscapes:
        weights_str, config_str = args.eomt_cityscapes.split(':')
        weights, config = _abs(weights_str), _abs(config_str)
        res = evaluate_model("EoMT Cityscapes Pretrained", weights, config, dataset_dir, device, "eomt_cityscapes", args.limit, args.ignore_index)
        results.append(res)
        
    # 3. EoMT COCO
    if args.eomt_coco:
        weights_str, config_str = args.eomt_coco.split(':')
        weights, config = _abs(weights_str), _abs(config_str)
        res = evaluate_model("EoMT COCO Pretrained", weights, config, dataset_dir, device, "eomt_coco", args.limit, args.ignore_index)
        results.append(res)
        
    # 4. EoMT Finetuned
    if args.eomt_finetuned:
        weights_str, config_str = args.eomt_finetuned.split(':')
        weights, config = _abs(weights_str), _abs(config_str)
        label = f"EoMT Finetuned ({os.path.basename(weights)})"
        res = evaluate_model(label, weights, config, dataset_dir, device, "eomt_finetuned", args.limit, args.ignore_index)
        results.append(res)
            
    if results:
        write_results_csv(results, output_csv, CITYSCAPES_CLASSES)
        print_results_summary(results, CITYSCAPES_CLASSES)
    else:
        print("No models evaluated. Please provide at least one model weights argument.")

if __name__ == "__main__":
    main()
