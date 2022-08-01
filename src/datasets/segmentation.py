import os
import pandas as pd
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import albumentations as A
import random
import cv2


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        phase="train",
        transform=None,
        albumentation=False,
        bgroot=None,
        bg_names=None,
    ):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.phase = phase
        df = pd.read_csv(os.path.join(root, f"{phase}.csv"))
        self.image_name_list = [os.path.join(root, p) for p in df.image]
        self.label_name_list = [os.path.join(root, p) for p in df.image_w_alpha]
        self.transform = transform
        self.albumentation = albumentation
        self.bgs = []
        if bgroot is not None:
            bg_names = os.listdir(bgroot) if bg_names is None else bg_names
            for name in bg_names:
                self.bgs.extend(glob.glob(os.path.join(bgroot, name, "*jpg")))
            print(f"found {len(self.bgs)} background images")

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        label = io.imread(self.label_name_list[idx])
        # get random background
        bg_sample = {}
        if len(self.bgs) > 0:
            bg_path = random.sample(self.bgs, 1)[0]
            bg = io.imread(bg_path)
            bg = cv2.resize(bg, (288, 288))
            if np.ndim(bg) == 2 or bg.shape[2] == 1:
                bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_sample["bg"] = bg
            bg_sample["bg_path"] = bg_path

        if self.albumentation:
            albumentation_transform = get_albumentations()
            transform_output = albumentation_transform(image=image, label=label)
            image = transform_output["image"]
            label = transform_output["label"]

        sample = {
            "imidx": imidx,
            "image": image,
            "label": label,
            "path": self.image_name_list[idx],
            **bg_sample,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_albumentations():
    transforms = A.Compose(
        [
            A.Resize(320, 320),
            A.ShiftScaleRotate(p=0.4),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=0,
                border_mode=0,
                value=0,
                p=0.2,
            ),
            A.PiecewiseAffine(
                scale=(0.03, 0.04),
                nb_rows=5,
                nb_cols=5,
                always_apply=True,
                keypoints_threshold=0.01,
                p=0.4,
            ),
            A.HorizontalFlip(p=0.5),
        ],
        additional_targets={"label": "image"},
    )

    return transforms
