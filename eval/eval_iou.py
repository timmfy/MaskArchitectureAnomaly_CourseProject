# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import torch
import os
import time

from PIL import Image

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20


class LabelIdsToTrainIds:

    def __init__(self):
        lut = torch.full((256,), 255, dtype=torch.long)

        # Pass-through for existing trainIds and ignore.
        for train_id in range(19):
            lut[train_id] = train_id
        lut[255] = 255

        # Cityscapes labelId -> trainId mapping.
        lut[7] = 0
        lut[8] = 1
        lut[11] = 2
        lut[12] = 3
        lut[13] = 4
        lut[17] = 5
        lut[19] = 6
        lut[20] = 7
        lut[21] = 8
        lut[22] = 9
        lut[23] = 10
        lut[24] = 11
        lut[25] = 12
        lut[26] = 13
        lut[27] = 14
        lut[28] = 15
        lut[31] = 16
        lut[32] = 17
        lut[33] = 18

        self.lut = lut

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor), 'tensor needs to be LongTensor'
        return self.lut[tensor]

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    LabelIdsToTrainIds(),
    Relabel(255, 19),   #ignore label to 19
])

def evaluate_erfnet(
    weightsPath="weights/erfnet_pretrained.pth",
    modelPath="erfnet.py",
    subset="val",  # can be val or train (must have labels)
    datadir="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/",
    num_workers=4,
    batch_size=1,
    cpu=False,
    state=None,
    limit=None,
    ignore_index=19
):

    print("Loading model: " + modelPath)
    print("Loading weights: " + weightsPath)

    model = ERFNet(NUM_CLASSES)

    if not cpu:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model).to(device)

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(f"{name} not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightsPath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")

    model.eval()

    if not os.path.exists(datadir):
        print(f"Error: datadir could not be loaded: {datadir}")

    loader = DataLoader(
        cityscapes(datadir, input_transform_cityscapes, target_transform_cityscapes, subset=subset), 
        num_workers=num_workers, 
        batch_size=batch_size, 
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=ignore_index)

    start = time.time()

    for step, (images, labels, _, _) in enumerate(loader):
        if limit is not None and step >= limit:
            break

        if not cpu:
            images = images.to(device)
            labels = labels.to(device)

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_float = [float(v) for v in iou_classes]

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print(f"Took {time.time()-start} seconds")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal*100) + '\033[0m'
    print(f"MEAN IoU: {iouStr}%")

    return iouVal, iou_classes

if __name__ == '__main__':

    evaluate_erfnet()