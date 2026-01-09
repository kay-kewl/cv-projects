import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class BirdsDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=256):
        self.root_dir = os.path.join(root_dir, split)
        self.images_folder = os.path.join(self.root_dir, "images")
        self.gt_folder = os.path.join(self.root_dir, "gt")

        self.image_paths = []
        self.mask_paths = []
        self._load_filepaths()

        self.transform = self._get_transforms(split, image_size)

    def _load_filepaths(self):
        if not os.path.exists(self.images_folder):
            raise FileNotFoundError(f"Directory not found: {self.images_folder}")

        for class_name in sorted(os.listdir(self.images_folder)):
            class_dir = os.path.join(self.images_folder, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, fname)
                    mask_name = os.path.splitext(fname)[0] + ".png"
                    mask_path = os.path.join(self.gt_folder, class_name, mask_name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

    def _get_transforms(self, split, size):
        if split == "train":
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=size),
                    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
                    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.3),
                    A.ColorJitter(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=size),
                    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
    
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask
