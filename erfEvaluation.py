import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
import numpy as np
import os
import sys

eval_path = os.path.abspath("eval")
if eval_path not in sys.path:
    sys.path.insert(0, eval_path)

try:
    from iouEval import iouEval
except ImportError:
    print("Error: iouEval module not found.")
    sys.exit(1)

def evaluate_erfnet_cityscapes(model, dataloader, device, num_classes=19, ignore_index=255, limit=None):
    model.eval()
    evaluator = iouEval(num_classes + 1, ignoreIndex=num_classes)
    count = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            if limit and count >= limit:
                break
                
            imgs = imgs.to(device)
            targets = targets.squeeze(1).to(device) if targets.dim() == 4 else targets.to(device)

            with autocast(dtype=torch.float16, device_type=device.type):
                logits = model(imgs) 
                
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                
                preds = logits.argmax(dim=1)

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            for i in range(preds.shape[0]):
                pred_eval = preds[i].copy()
                target_eval = targets[i].copy()
                
                pred_eval[pred_eval == ignore_index] = num_classes
                target_eval[target_eval == ignore_index] = num_classes
                
                pred_tensor = torch.from_numpy(pred_eval).unsqueeze(0).unsqueeze(0).long()
                target_tensor = torch.from_numpy(target_eval).unsqueeze(0).unsqueeze(0).long()
                
                evaluator.addBatch(pred_tensor, target_tensor)
                
                count += 1
                if limit and count >= limit:
                    break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {count} images...")

    mIoU, ious = evaluator.getIoU()
    return mIoU, ious

def get_cityscapes_dataloader(data_root, batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.squeeze().long())
    ])

    dataset = Cityscapes(
        data_root, 
        split='val', 
        mode='fine', 
        target_type='semantic',
        transform=transform,
        target_transform=target_transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
