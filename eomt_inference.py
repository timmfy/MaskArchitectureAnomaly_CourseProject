import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add eval path to sys.path to import iouEval
eval_path = os.path.abspath("eval")
if eval_path not in sys.path:
    sys.path.insert(0, eval_path)

try:
    from iouEval import iouEval
except ImportError:
    iouEval = None

# ... (rest of the imports and mappings)

# Import mapping if available
try:
    from coco_to_cityscape import coco_to_cityscape
except ImportError:
    coco_to_cityscape = {}

CLASS_MAPPING_COCO = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 92: 80, 93: 81, 95: 82, 100: 83, 107: 84, 109: 85, 112: 86, 118: 87, 119: 88, 122: 89, 125: 90, 128: 91, 130: 92, 133: 93, 138: 94, 141: 95, 144: 96, 145: 97, 147: 98, 148: 99, 149: 100, 151: 101, 154: 102, 155: 103, 156: 104, 159: 105, 161: 106, 166: 107, 168: 108, 171: 109, 175: 110, 176: 111, 177: 112, 178: 113, 180: 114, 181: 115, 184: 116, 185: 117, 186: 118, 187: 119, 188: 120, 189: 121, 190: 122, 191: 123, 192: 124, 193: 125, 194: 126, 195: 127, 196: 128, 197: 129, 198: 130, 199: 131, 200: 132,
}

IDX_TO_COCO = {v: k for k, v in CLASS_MAPPING_COCO.items()}

def get_idx_to_cityscapes():
    idx_to_cs = {}
    for idx, coco_id in IDX_TO_COCO.items():
        if coco_id in coco_to_cityscape:
            idx_to_cs[idx] = coco_to_cityscape[coco_id]
        else:
            idx_to_cs[idx] = 255
    return idx_to_cs

IDX_TO_CITYSCAPES = get_idx_to_cityscapes()

def map_coco_preds_to_cityscapes(preds):
    """Map COCO predicted indices to Cityscapes train IDs."""
    cs_preds = np.full(preds.shape, 255, dtype=np.uint8)
    for coco_idx, cs_id in IDX_TO_CITYSCAPES.items():
        cs_preds[preds == coco_idx] = cs_id
    return cs_preds

def create_mapping(images, ignore_index):
    """Create a color mapping for different classes/IDs."""
    unique_ids = np.unique(np.concatenate([np.unique(img) for img in images]))
    valid_ids = unique_ids[unique_ids != ignore_index]
    colors = np.array(
        [plt.cm.hsv(i / len(valid_ids))[:3] for i in range(len(valid_ids))]
    )
    mapping = {cid: colors[i] for i, cid in enumerate(valid_ids)}
    mapping[ignore_index] = np.array([0, 0, 0])
    return mapping

def apply_colormap(image, mapping):
    """Apply a color mapping to a single-channel image."""
    colored_image = np.zeros((*image.shape, 3))
    for cid in np.unique(image):
        colored_image[image == cid] = mapping.get(cid, [0, 0, 0])
    return colored_image

def infer_semantic(model, img, target, device, img_size, ignore_index=255):
    """Perform semantic inference on a single image."""
    with torch.no_grad(), autocast(dtype=torch.float16, device_type=device.type if hasattr(device, 'type') else str(device)):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear"
        )

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        preds = logits[0].argmax(0).cpu()

    pred_array = preds.numpy()
    target_array = model.to_per_pixel_targets_semantic([target], ignore_index)[0].numpy()
    return pred_array, target_array

def plot_semantic_results(img, pred_array, target_array, ignore_index=255):
    """Plot original image, prediction, and target for semantic segmentation."""
    mapping = create_mapping([pred_array, target_array], ignore_index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if img.dim() == 3:
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    else:
        axes[0].imshow(img.cpu().numpy())
    axes[0].set_title("Image")
    axes[1].imshow(apply_colormap(pred_array, mapping))
    axes[1].set_title("Prediction")
    axes[2].imshow(apply_colormap(target_array, mapping))
    axes[2].set_title("Target")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def infer_panoptic(model, img, target, device):
    """Perform panoptic inference on a single image."""
    with torch.no_grad(), autocast(dtype=torch.float16, device_type=device.type if hasattr(device, 'type') else str(device)):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds = model.to_per_pixel_preds_panoptic(
            mask_logits,
            class_logits_per_layer[-1],
            model.stuff_classes,
            model.mask_thresh,
            model.overlap_thresh,
        )[0].cpu()

    pred = preds.numpy()
    sem_pred, inst_pred = pred[..., 0], pred[..., 1]

    target_seg = model.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
    sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

    return sem_pred, inst_pred, sem_target, inst_target

def draw_black_border(sem, inst, mapping):
    """Draw black borders around instances in panoptic segmentation."""
    h, w = sem.shape
    out = np.zeros((h, w, 3))
    for s in np.unique(sem):
        out[sem == s] = mapping.get(s, [0, 0, 0])

    combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out

def plot_panoptic_results(model, img, sem_pred, inst_pred, sem_target, inst_target):
    """Plot original image, prediction, and target for panoptic segmentation."""
    all_ids = np.union1d(np.unique(sem_pred), np.unique(sem_target))
    mapping = {
        s: (
            [0, 0, 0]
            if s == -1 or s == model.num_classes
            else plt.cm.hsv(i / len(all_ids))[:3]
        )
        for i, s in enumerate(all_ids)
    }

    vis_pred = draw_black_border(sem_pred, inst_pred, mapping)
    vis_target = draw_black_border(sem_target, inst_target, mapping)

    img_np = (
        img.cpu().numpy().transpose(1, 2, 0) if img.dim() == 3 else img.cpu().numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(vis_pred)
    axes[1].set_title("Prediction")
    axes[2].imshow(vis_target)
    axes[2].set_title("Target")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def evaluate_semantic(model, dataloader, device, img_size, num_classes=19, ignore_index=255, is_coco=False, limit=None):
    """Evaluate semantic segmentation on a dataset."""
    if iouEval is None:
        print("iouEval not found. Skipping evaluation.")
        return None, None

    # Cityscapes has 19 classes. iouEval expects ignoreIndex to be nClasses-1 if provided.
    # We map 255 to num_classes (e.g. 19) and use nClasses=num_classes+1 (e.g. 20).
    evaluator = iouEval(num_classes + 1, ignoreIndex=num_classes)
    
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if limit and count >= limit:
                break
            
            imgs, targets = batch
            for i in range(len(imgs)):
                img = imgs[i]
                target = targets[i]
                
                # Perform inference
                with autocast(dtype=torch.float16, device_type=device.type if hasattr(device, 'type') else str(device)):
                    imgs_in = [img.to(device)]
                    img_sizes = [img.shape[-2:] for img in imgs_in]
                    crops, origins = model.window_imgs_semantic(imgs_in)

                    mask_logits_per_layer, class_logits_per_layer = model(crops)
                    mask_logits = F.interpolate(
                        mask_logits_per_layer[-1], img_size, mode="bilinear"
                    )

                    crop_logits = model.to_per_pixel_logits_semantic(
                        mask_logits, class_logits_per_layer[-1]
                    )
                    logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
                    preds = logits[0].argmax(0).cpu().numpy()

                target_array = model.to_per_pixel_targets_semantic([target], ignore_index)[0].numpy()
                
                if is_coco:
                    preds = map_coco_preds_to_cityscapes(preds)
                
                # Map target and preds to [0, num_classes] range for iouEval
                # 255 -> num_classes
                preds_eval = preds.copy()
                preds_eval[preds_eval == 255] = num_classes
                target_eval = target_array.copy()
                target_eval[target_eval == 255] = num_classes
                
                evaluator.addBatch(torch.from_numpy(preds_eval).unsqueeze(0).unsqueeze(0).long(), 
                                   torch.from_numpy(target_eval).unsqueeze(0).unsqueeze(0).long())
                
                count += 1
                if limit and count >= limit:
                    break
    
    mIoU, ious = evaluator.getIoU()
    return mIoU, ious
