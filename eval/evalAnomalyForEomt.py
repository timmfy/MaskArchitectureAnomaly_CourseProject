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

import eomt_setup
import eomt_inference
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torch.nn.functional as F
import matplotlib.pyplot as plt

def salva_predizione(immagine_tensor, score_anomalia, ground_truth, nome_file_salvataggio, output_dir, nome_metrica="Score"):
    """
    Salva un'immagine con l'originale, la heatmap e la ground truth a colori.
    """
    fig, assi = plt.subplots(1, 3, figsize=(18, 5))
    
    os.makedirs(output_dir, exist_ok=True)
    # --- 1. IMMAGINE ORIGINALE ---
    img_np = immagine_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.clip(img_np, 0, 1)
    
    assi[0].imshow(img_np)
    assi[0].set_title("Immagine Originale")
    assi[0].axis('off')
    
    # --- 2. PREDIZIONE (HEATMAP) ---
    mappa = assi[1].imshow(score_anomalia, cmap='jet')
    assi[1].set_title(f"Predizione ({nome_metrica})")
    assi[1].axis('off')
    fig.colorbar(mappa, ax=assi[1], fraction=0.046, pad=0.04)
    
    # --- 3. GROUND TRUTH (A COLORI) ---
    h, w = ground_truth.shape
    # Creiamo un'immagine vuota a 3 canali (RGB)
    gt_color = np.zeros((h, w, 3), dtype=np.float32)
    
    # Assegniamo i colori in base ai valori
    gt_color[ground_truth == 0] = [0.2, 0.2, 0.2]   # Strada (Valore 0) -> Grigio scuro
    gt_color[ground_truth == 1] = [1.0, 0.0, 0.0]   # Anomalia (Valore 1) -> ROSSO
    gt_color[ground_truth == 255] = [1.0, 1.0, 1.0] # Ignora (Valore 255) -> Bianco
    
    assi[2].imshow(gt_color)
    assi[2].set_title("Ground Truth (Valuation)")
    assi[2].axis('off')
    
    # --- SALVATAGGIO ---
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, nome_file_salvataggio), bbox_inches='tight')
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
    if not os.path.exists(f'results_anomaly/results_{args.loadWeights.split("/")[-1].split(".")[0]}_{args.loadDataset}.csv'):
        open(f'results_anomaly/results_{args.loadWeights.split("/")[-1].split(".")[0]}_{args.loadDataset}.csv', 'w').close()
    file = open(f'results_anomaly/results_{args.loadWeights.split("/")[-1].split(".")[0]}_{args.loadDataset}.csv', 'a')

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

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_MSP_list.append(anomaly_score_MSP)
             anomaly_score_MaxLogit_list.append(anomaly_score_MaxLogit)
             anomaly_score_MaxEntropy_list.append(anomaly_score_MaxEntropy)
             anomaly_score_RbA_list.append(anomaly_score_RbA)
             salva_predizione(images, anomaly_score_MSP, ood_gts, f"out_{os.path.basename(path)}", output_dir, "MSP")

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

