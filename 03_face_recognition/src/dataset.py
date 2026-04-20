import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebAIdentityDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=112):
        """
        CelebA Dataset with Identity Labels for Metric Learning.

        Args:
            root_dir: Path to folder containing 'img_align_celeba' and 'identity_CelebA.txt'
        """
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, "img_align_celeba")
        self.identity_file = os.path.join(root_dir, "identity_CelebA.txt")

        self.image_files, self.labels = self._load_identities()
        self.transform = self._get_transforms(split, image_size)

    def _load_identities(self):
        with open(self.identity_file, "r") as f:
            lines = f.readlines()

        split_ranges = {
            "train": (0, 162770),
            "val": (162770, 182637),
            "test": (182637, 202599),
        }
        start, end = split_ranges.get(self.split, (0, 0))
        lines_in_split = lines[start:end]

        unique_ids = sorted(list(set([int(line.split()[1]) for line in lines_in_split])))
        id_map = {original: new for new, original in enumerate(unique_ids)}

        images, labels = [],[]
        for line in lines_in_split:
            fname, original_id = line.split()
            images.append(os.path.join(self.image_dir, fname))
            labels.append(id_map[int(original_id)])

        return images, labels

    def _get_transforms(self, split, size):
        if split == "train":
            return transforms.Compose(
                [
                    transforms.CenterCrop(178),
                    transforms.Resize((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
