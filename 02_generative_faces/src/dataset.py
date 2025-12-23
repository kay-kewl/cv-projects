import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=64):
        """
        Custom Dataset for CelebA to handle face cropping and resizing.

        Args:
            root_dir (str): Path containing the 'img_align_celeba' folder.
            split (str): 'train', 'val', or 'test'.
            image_size (int): Target output size (default 64).
        """
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, "img_align_celeba")
        self.attr_path = os.path.join(root_dir, "list_eval_partition.txt")

        self.image_files = self._load_split()

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def _load_split(self):
        """Parses the CelebA partition file to separate train/val/test."""
        if not os.path.exists(self.attr_path):
            print(
                f"Warning: Partition file not found at {self.attr_path}. Loading all images."
            )
            return [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]

        split_map = {"train": 0, "val": 1, "test": 2}
        target_split = split_map.get(self.split, 0)

        files = []
        with open(self.attr_path, "r") as f:
            for line in f:
                filename, partition = line.split()
                if int(partition) == target_split:
                    files.append(filename)
        return files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0
