#!/usr/bin/env python3
"""
Run anomaly eval scripts for configured model/dataset combos,
then consolidate all per-run CSVs into one comparison CSV.

Usage (from repo root):
  python eval/run_eval_anomaly.py \
    --datasets RoadObsticle21 \
    --erfnet-weights trained_models/erfnet_pretrained.pth \
        --eomt-cityscapes eomt/weights/eomt_cityscapes.bin:eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --output-csv results_anomaly/comparison.csv
"""
import os
import sys
import csv
import subprocess
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def _parse_weight_config(value):
    if not value:
        return None
    if ":" not in value:
        raise ValueError("Expected 'weight_path:config_path'")
    weight_path, conf_path = value.split(":", 1)
    return _abs(weight_path), _abs(conf_path)


def run_eval(script_name, extra_args):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path] + list(extra_args)
    label = f"{script_name} {' '.join(str(a) for a in extra_args[:4])} ..."
    print(f"\n[run] {label}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"  WARNING: exited with code {result.returncode}")
        return False
    return True


def read_results_csv(csv_path):
    """Read a results CSV produced by the eval scripts, skipping duplicate headers."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    seen = set()
    with open(csv_path, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("Model,Dataset"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            row = {
                "Model": parts[0],
                "Dataset": parts[1],
                "Method": parts[2],
                "AUPRC": parts[3],
                "FPR@TPR95": parts[4],
            }
            key = (row["Model"], row["Dataset"], row["Method"])
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def print_table(rows, fieldnames):
    if not rows:
        return
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in fieldnames}
    header = " | ".join(f"{k:<{widths[k]}}" for k in fieldnames)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r.get(k, '')):<{widths[k]}}" for k in fieldnames))


def main():
    parser = ArgumentParser(description="Collect anomaly segmentation results.")
    parser.add_argument(
        "--datasets-dir",
        default=os.path.join(ROOT_DIR, "datasets/Validation_Dataset"),
    )
    parser.add_argument("--datasets", nargs="+", default=["RoadObsticle21"])
    parser.add_argument(
        "--erfnet-weights",
        default="trained_models/erfnet_pretrained.pth",
        help="ERFNet weights path (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--eomt-cityscapes",
        type=str,
        help="weights:config pair for EoMT Cityscapes (paths relative to repo root or absolute)",
    )
    parser.add_argument(
        "--eomt-coco",
        type=str,
        help="weights:config pair for EoMT COCO (paths relative to repo root or absolute)",
    )
    parser.add_argument(
        "--eomt-finetuned",
        type=str,
        help="weights:config pair for EoMT finetuned (paths relative to repo root or absolute)",
    )
    parser.add_argument(
        "--output-csv",
        default="results_anomaly/comparison.csv",
        help="Output comparison CSV path (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--skip-erfnet", action="store_true", help="Skip ERFNet evaluation"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running eval scripts and only merge existing CSVs",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT_DIR, "results_anomaly"), exist_ok=True)
    output_csv = _abs(args.output_csv)

    # Run eval scripts for each configured model/dataset combo, unless --skip-run is set
    if not args.skip_run:
        if not args.skip_erfnet:
            w = _abs(args.erfnet_weights)
            if os.path.exists(w):
                for ds in args.datasets:
                    run_eval("evalAnomaly.py", [
                        "--input", os.path.join(args.datasets_dir, ds, "images", "*.*"),
                        "--loadDataset", ds,
                        "--loadDir", os.path.dirname(w) + os.sep,
                        "--loadWeights", os.path.basename(w),
                    ])
            else:
                print(f"ERFNet weights not found, skipping: {w}")

        if args.eomt_cityscapes:
            try:
                weight_path, conf_path = _parse_weight_config(args.eomt_cityscapes)
            except ValueError as exc:
                print(f"Skipping malformed entry for EoMT Cityscapes Pretrained: {exc}")
            else:
                if os.path.exists(weight_path):
                    for ds in args.datasets:
                        run_eval("evalAnomalyForEomt.py", [
                            "--input", os.path.join(args.datasets_dir, ds, "images", "*.*"),
                            "--loadDataset", ds,
                            "--loadWeights", weight_path,
                            "--loadConf", conf_path,
                        ])
                else:
                    print(f"Weights not found, skipping: {weight_path}")

        if args.eomt_coco:
            try:
                weight_path, conf_path = _parse_weight_config(args.eomt_coco)
            except ValueError as exc:
                print(f"Skipping malformed entry for EoMT COCO Pretrained: {exc}")
            else:
                if os.path.exists(weight_path):
                    for ds in args.datasets:
                        run_eval("evalAnomalyForEomt.py", [
                            "--input", os.path.join(args.datasets_dir, ds, "images", "*.*"),
                            "--loadDataset", ds,
                            "--loadWeights", weight_path,
                            "--coco",
                            "--loadConf", conf_path,
                        ])
                else:
                    print(f"Weights not found, skipping: {weight_path}")

        if args.eomt_finetuned:
            try:
                weight_path, conf_path = _parse_weight_config(args.eomt_finetuned)
            except ValueError as exc:
                print(f"Skipping malformed entry for EoMT Finetuned: {exc}")
            else:
                if os.path.exists(weight_path):
                    for ds in args.datasets:
                        run_eval("evalAnomalyForEomt.py", [
                            "--input", os.path.join(args.datasets_dir, ds, "images", "*.*"),
                            "--loadDataset", ds,
                            "--loadWeights", weight_path,
                            "--loadConf", conf_path,
                        ])
                else:
                    print(f"Weights not found, skipping: {weight_path}")

    # Consolidate all per-run CSVs
    results_dir = os.path.join(ROOT_DIR, "results_anomaly")
    all_rows = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".csv") and os.path.join(results_dir, fname) != output_csv and not fname.__contains__("temperature"):
            all_rows.extend(read_results_csv(os.path.join(results_dir, fname)))

    fieldnames = ["Model", "Dataset", "Method", "AUPRC", "FPR@TPR95"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nComparison saved to: {output_csv}")
    if all_rows:
        print_table(all_rows, fieldnames)
    else:
        print("No results found. Run eval scripts first or check results_anomaly/.")


if __name__ == "__main__":
    main()
