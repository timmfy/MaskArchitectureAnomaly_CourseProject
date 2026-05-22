# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png") or filename.endswith("_labelIds.png")


def cityscapes_key_from_image(filename):
    base = os.path.basename(filename)
    if base.endswith("_leftImg8bit.png"):
        return base[:-len("_leftImg8bit.png")]
    if base.endswith("_leftImg8bit.jpg"):
        return base[:-len("_leftImg8bit.jpg")]
    return os.path.splitext(base)[0]


def cityscapes_key_from_label(filename):
    base = os.path.basename(filename)
    if base.endswith("_gtFine_labelTrainIds.png"):
        return base[:-len("_gtFine_labelTrainIds.png")]
    if base.endswith("_gtFine_labelIds.png"):
        return base[:-len("_gtFine_labelIds.png")]
    return os.path.splitext(base)[0]

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    if os.path.isabs(name):
        return name
    root_norm = os.path.normpath(root)
    name_norm = os.path.normpath(name)
    if name_norm.startswith(root_norm + os.sep):
        return name
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='val'):

        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)
        image_files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(os.path.expanduser(self.images_root))
            for f in fn if is_image(f)
        ]
        image_files.sort()

        label_files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(os.path.expanduser(self.labels_root))
            for f in fn if is_label(f)
        ]
        label_files.sort()

        labels_by_key = {}
        for f in label_files:
            key = cityscapes_key_from_label(f)
            # Prefer trainIds masks when both are present.
            if key in labels_by_key and labels_by_key[key].endswith("_labelTrainIds.png"):
                continue
            labels_by_key[key] = f

        self.samples = []
        for image_file in image_files:
            key = cityscapes_key_from_image(image_file)
            label_file = labels_by_key.get(key)
            if label_file is not None:
                self.samples.append((image_file, label_file))

        self.filenames = [sample[0] for sample in self.samples]
        self.filenamesGt = [sample[1] for sample in self.samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No Cityscapes pairs found in {self.images_root} and {self.labels_root}. "
                "Expected *_labelTrainIds.png or *_labelIds.png labels."
            )

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename, filenameGt = self.samples[index]

        #print(filename)

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.samples)

