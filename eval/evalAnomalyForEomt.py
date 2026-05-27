import os
import sys
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
root_path = os.path.abspath(".")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from eomt_tools import eomt_setup, eomt_inference
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathGTComparison import maskGt

def save_prediction(image_tensor, anomaly_score, ground_truth, save_filename, output_dir, metric_name="Score"):
    """
    Save an image composed of the original image, the heatmap, and the colored ground truth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    os.makedirs(output_dir, exist_ok=True)
    img_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.clip(img_np, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    heatmap_im = axes[1].imshow(anomaly_score, cmap='jet')
    axes[1].set_title(f"Prediction ({metric_name})")
    axes[1].axis('off')
    fig.colorbar(heatmap_im, ax=axes[1], fraction=0.046, pad=0.04)

    h, w = ground_truth.shape
    gt_color = np.zeros((h, w, 3), dtype=np.float32)

    gt_color[ground_truth == 0] = [0.2, 0.2, 0.2]
    gt_color[ground_truth == 1] = [1.0, 0.0, 0.0]
    gt_color[ground_truth == 255] = [1.0, 1.0, 1.0]

    axes[2].imshow(gt_color)
    axes[2].set_title("Ground Truth (Colored)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_filename), bbox_inches='tight')
    plt.close(fig)

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IMG_SIZE = (1024, 1024)
input_transform = Compose(
    [
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize(IMG_SIZE, Image.NEAREST),
    ]
)

NUM_CLASSES = 19 

class DataInfo:
    def __init__(self):
        self.img_size = IMG_SIZE
        self.num_classes = NUM_CLASSES

def main():
    parser = ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(os.path.join(script_dir, "../datasets/Validation_Dataset/RoadObsticle21/images/*.webp"))
    default_weights = os.path.abspath(os.path.join(script_dir, "../eomt/weights/eomt_cityscapes.bin"))
    default_conf = os.path.abspath(os.path.join(script_dir, "../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"))

    parser.add_argument("--input", default=default_input, nargs="+")
    parser.add_argument('--loadDataset', default="RoadObsticle21")
    parser.add_argument('--loadWeights', default=default_weights)
    parser.add_argument('--loadConf', default=default_conf)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Lists for metrics
    anomaly_score_MSP_list = []
    anomaly_score_MaxLogit_list = []
    anomaly_score_MaxEntropy_list = []
    anomaly_score_RbA_list = [] # Added RbA
    ood_gts_list = []

    os.makedirs("results_anomaly", exist_ok=True)
    file = open(f'results_anomaly/results_{args.loadWeights.split("/")[-1].split(".")[0]}_{args.loadDataset}.csv', 'w')

    # --- MODEL LOADING ---

    device = eomt_setup.setup_environment(eomt_path="eomt")

    config_path = args.loadConf
    weights_path = args.loadWeights

    conf = eomt_setup.load_config(config_path)
    data_info = DataInfo()
    model = eomt_setup.load_model(conf, data_info, torch.device("cpu"),weights_path = weights_path)
    

    if not args.cpu:
        model = model.to(device)

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model_name = weights_path.split("/")[-1].split(".")[0]
        output_dir = os.path.join("images", dataset_name, model_name)
        save_logit = os.path.join("logits", dataset_name, model_name)
        os.makedirs(save_logit, exist_ok=True)



        images = input_transform((Image.open(path).convert('RGB'))).float().to(device)
        logits = eomt_inference.get_pixel_logits(model, images, IMG_SIZE, device)
        torch.save(logits.cpu(), os.path.join(save_logit, f"logits_{os.path.basename(path).split('.')[0]}.pt"))

        probs = F.softmax(logits, dim=0)
        
        logits_np = logits.squeeze(0).cpu().numpy()
        probs_np = probs.squeeze(0).cpu().numpy()

        epsilon = 1e-7

        # MSP (Maximum Softmax Probability)
        anomaly_score_MSP = 1.0 - np.max(probs_np, axis=0)
        
        # Max Logit
        anomaly_score_MaxLogit = 1.0 -(np.max(logits_np, axis=0))
        
        # Max Entropy
        anomaly_score_MaxEntropy =np.sum(-probs_np * np.log(probs_np + epsilon), axis=0)

        # RbA
        anomaly_score_RbA = -torch.tanh(logits).sum(dim=0).cpu().numpy()
        
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        result = maskGt(pathGT, target_transform)

        if result is None:
            continue
        else:
            ood_gts, _ = result
            ood_gts_list.append(ood_gts)          
            ood_gts_list.append(ood_gts)
            anomaly_score_MSP_list.append(anomaly_score_MSP)
            anomaly_score_MaxLogit_list.append(anomaly_score_MaxLogit)
            anomaly_score_MaxEntropy_list.append(anomaly_score_MaxEntropy)
            anomaly_score_RbA_list.append(anomaly_score_RbA)
            save_prediction(images, anomaly_score_MSP, ood_gts, f"out_{os.path.basename(path)}", output_dir, "MSP")

    anomaly_scores = {
        "MSP": np.array(anomaly_score_MSP_list),
        "MaxLogit": np.array(anomaly_score_MaxLogit_list),
        "MaxEntropy": np.array(anomaly_score_MaxEntropy_list),
        "RbA": np.array(anomaly_score_RbA_list)
    }

    ood_gts = np.array(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    file.write(f"Model,Dataset,Method,AUPRC,FPR@TPR95\n")
    for method_name, scores in anomaly_scores.items():
        ood_out = scores[ood_mask]
        ind_out = scores[ind_mask]

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        model_name = args.loadWeights.split("/")[-1].split(".")[0]
        file.write(f"{model_name},{dataset_name},{method_name},{prc_auc*100.0:.2f},{fpr*100.0:.2f}")
        file.write("\n")

    file.close()

if __name__ == '__main__':
    main()

