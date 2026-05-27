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
from pathGTComparison import maskGt

import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_prediction(image_tensor, anomaly_score, ground_truth, save_filename, output_dir, metric_name="Score"):
    """
    Save an image composed of the original image, the heatmap, and the colored ground truth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    os.makedirs(output_dir, exist_ok=True)

    # Original Image
    img_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.clip(img_np, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Prediction
    heatmap_im = axes[1].imshow(anomaly_score, cmap='jet')
    axes[1].set_title(f"Prediction ({metric_name})")
    axes[1].axis('off')
    fig.colorbar(heatmap_im, ax=axes[1], fraction=0.046, pad=0.04)

    # GT
    h, w = ground_truth.shape
    # Create an empty 3-channel (RGB) image
    gt_color = np.zeros((h, w, 3), dtype=np.float32)

    # Assign colors based on the label values
    gt_color[ground_truth == 0] = [0.2, 0.2, 0.2]   # Road (Value 0) -> Dark gray
    gt_color[ground_truth == 1] = [1.0, 0.0, 0.0]   # Anomaly (Value 1) -> Red
    gt_color[ground_truth == 255] = [1.0, 1.0, 1.0] # Ignore (Value 255) -> White

    axes[2].imshow(gt_color)
    axes[2].set_title("Ground Truth (Colored)")
    axes[2].axis('off')

    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_filename), bbox_inches='tight')
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
        default="datasets/Validation_Dataset/RoadAnomaly21/images/*.*",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadDataset', default="RoadAnomaly21")
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

    os.makedirs("results_anomaly", exist_ok=True)
    file = open(f'results_anomaly/results_erfnet_{args.loadDataset}.csv', 'w')

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
    print ("Model and weights loaded successfully")
    model.eval()
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        output_dir = os.path.join("images", dataset_name, "erfnet")
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        # images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images)

        logits = result.squeeze(0).data.cpu().numpy()
        probabilities = F.softmax(result, dim =1)
        probabilities = probabilities.squeeze(0).data.cpu().numpy()

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

        result = maskGt(pathGT, target_transform)
        if result is None:
            continue
        else:
            ood_gts, _ = result
            ood_gts_list.append(ood_gts)
            anomaly_score_MSP_list.append(anomaly_score_MSP)
            anomaly_score_MaxLogit_list.append(anomaly_score_MaxLogit)
            anomaly_score_MaxEntropy_list.append(anomaly_score_MaxEntropy)
            base_name = os.path.basename(path).split('.')[0]
            heatmap_filename = f"heatmap_{base_name}.png"
            save_prediction(
                image_tensor=images,
                anomaly_score=anomaly_score_MSP,
                ground_truth=ood_gts,
                save_filename=heatmap_filename,
                output_dir=output_dir,
                metric_name="MSP", 
            )
            
    anomaly_scores = {
        "MSP": np.array(anomaly_score_MSP_list),
        "MaxLogit": np.array(anomaly_score_MaxLogit_list),
        "MaxEntropy": np.array(anomaly_score_MaxEntropy_list)
    }

    ood_gts = np.array(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    file.write(f"Model,Dataset,Method,AUPRC,FPR@TPR95\n")
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

        file.write(f"ERFNET,{dataset_name},{method_name},{prc_auc*100.0:.2f},{fpr*100.0:.2f}")
        file.write( "\n")

    file.close()

if __name__ == '__main__':
    main()