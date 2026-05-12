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

def salva_predizione(immagine_tensor, score_anomalia, ground_truth, nome_file_salvataggio, nome_metrica="Score"):
    """
    Salva un'immagine con l'originale, la heatmap e la ground truth a colori.
    """
    fig, assi = plt.subplots(1, 3, figsize=(18, 5))
    
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
    plt.savefig(nome_file_salvataggio, bbox_inches='tight')
    plt.close(fig)

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((640, 640), Image.BILINEAR),
        ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((640, 640), Image.NEAREST),
    ]
)

NUM_CLASSES = 19 
IMG_SIZE = (640, 640)

class DataInfo:
    def __init__(self):
        self.img_size = IMG_SIZE
        self.num_classes = NUM_CLASSES

def main():
    parser = ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(os.path.join(script_dir, "../datasets/Anomaly_Validation_Datasets/Validation_Dataset/RoadObsticle21/images/*.webp"))
    default_weights = os.path.abspath(os.path.join(script_dir, "../weights/eomt_cityscapes.bin"))
    default_conf = os.path.abspath(os.path.join(script_dir, "../eomt/configs/dinov2/anomaly/eomt_640_RoadObsticle21.yaml"))

    parser.add_argument("--input", default=default_input, nargs="+")
    parser.add_argument('--loadDir', default=os.path.abspath(os.path.join(script_dir, "../weights/")))
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

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    # --- MODEL LOADING ---

    device = eomt_setup.setup_environment(eomt_path="eomt")

    config_path = args.loadConf
    weights_path = args.loadWeights

    conf = eomt_setup.load_config(config_path)
    data_info = DataInfo()
    model = eomt_setup.load_model(conf, data_info, torch.device("cpu"),weights_path = weights_path)



    print(f"Loading EoMT model from {args.loadWeights}")
    

    if not args.cpu:
        model = model.cuda()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        
        logits = eomt_inference.get_pixel_logits(model, images, device)
    
            
        # --- ANOMALY SCORING ---
        # 1. Probabilities
        probs = F.softmax(logits, dim=1)
        
        # Move to CPU for numpy operations
        logits_np = logits.squeeze(0).cpu().numpy()
        probs_np = probs.squeeze(0).cpu().numpy()

        epsilon = 1e-7

        # MSP (Maximum Softmax Probability)
        anomaly_score_MSP = 1.0 - np.max(probs_np, axis=0)
        
        # Max Logit
        anomaly_score_MaxLogit = -(np.max(logits_np, axis=0))
        
        # Max Entropy
        anomaly_score_MaxEntropy = -np.sum(probs_np * np.log(probs_np + epsilon), axis=0)

        # RbA to implement see the repo in git

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

             # Visualizing MSP as sample
             salva_predizione(images, anomaly_score_MSP, ood_gts, f"out_{os.path.basename(path)}", "MSP")

    # --- EVALUATION LOOP ---
    anomaly_scores = {
        "MSP": np.array(anomaly_score_MSP_list),
        "MaxLogit": np.array(anomaly_score_MaxLogit_list),
        "MaxEntropy": np.array(anomaly_score_MaxEntropy_list),
        "RbA": np.array(anomaly_score_RbA_list)
    }

    ood_gts = np.array(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    for method_name, scores in anomaly_scores.items():
        ood_out = scores[ood_mask]
        ind_out = scores[ind_mask]

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        result_str = f"Method: {method_name} | AUPRC: {prc_auc*100.0:.2f} | FPR95: {fpr*100.0:.2f}\n"
        print(result_str)
        file.write(result_str)

    file.close()

if __name__ == '__main__':
    main()

