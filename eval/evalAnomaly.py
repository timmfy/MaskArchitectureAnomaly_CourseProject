# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
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

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadDataset', default="RoadObsticle21")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    anomaly_score_MSP_list = []
    anomaly_score_MaxLogit_list = []
    anomaly_score_MaxEntropy_list = []

    ood_gts_list = []

    if not os.path.exists(f'results_erfnet_{args.loadDataset}.txt'):
        open(f'results_erfnet_{args.loadDataset}.txt', 'w').close()
    file = open(f'results_erfnet_{args.loadDataset}.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    if (not args.cpu):
        model = torch.nn.DataParallel(model).to(device)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        output_dir = os.path.join("images", dataset_name, "erfnet")

        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)

        # Calcolo degli Score per ERFnet

        # Calcolo Score intermedi : Logits e Probabilities

        logits = result.squeeze(0).data.cpu().numpy()
        probabilities = F.softmax(result, dim =1)
        probabilities = probabilities.squeeze(0).data.cpu().numpy()

    
        #Calcolo Score finali: MSP, MaxLogit, MaxEntropy

        epsilon = 1e-7

        anomaly_score_MSP = 1.0 - np.max(probabilities, axis = 0)
        anomaly_score_MaxLogit = 1.0 - np.max(logits, axis = 0)
        anomaly_score_MaxEntropy = np.sum(-probabilities * np.log(probabilities+epsilon), axis = 0)



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

             nome_base = os.path.basename(path)
             nome_file_out = f"heatmap_{nome_base}.png"
             salva_predizione(
                immagine_tensor=images, 
                score_anomalia=anomaly_score_MSP, 
                ground_truth=ood_gts,
                nome_file_salvataggio=nome_file_out,
                     output_dir=output_dir,
                nome_metrica="MSP"
        )


        del result, anomaly_score_MaxEntropy, anomaly_score_MaxLogit, anomaly_score_MSP, ood_gts, mask

    file.write( "\n")

    anomaly_scores = {
        "MSP": np.array(anomaly_score_MSP_list),
        "MaxLogit": np.array(anomaly_score_MaxLogit_list),
        "MaxEntropy": np.array(anomaly_score_MaxEntropy_list)
    }

    ood_gts = np.array(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    for method_name, anomaly_score in anomaly_scores.items():

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f'AUPRC score: {prc_auc*100.0}')
        print(f'FPR@TPR95: {fpr*100.0}')

        file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))

    file.close()

if __name__ == '__main__':
    main()