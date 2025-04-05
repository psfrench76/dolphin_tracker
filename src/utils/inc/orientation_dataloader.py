import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .settings import settings

# TODO: annotations should be either input from previous network, or loaded from dataset from __init__

class DolphinOrientationDataset(Dataset):
    def __init__(self, dataset_root_dir, transform=None):
        self.dataset_root_dir = Path(dataset_root_dir)
        self.image_dir = self.dataset_root_dir / settings['images_dir']
        self.label_dir = self.dataset_root_dir / settings['labels_dir']

        # TODO: determine format and convert from previous network or load from dataset
        annotations = [{'image': 'dolphin1.jpg', 'bbox': [50, 50, 100, 100], 'orientation': 45},
            {'image': 'dolphin2.jpg', 'bbox': [30, 30, 150, 150], 'orientation': 90}, ]

        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.annotations[idx]['image']
        image = Image.open(img_path).convert("RGB")
        bbox = self.annotations[idx]['bbox']
        orientation = self.annotations[idx]['orientation']

        # Crop the image using the bounding box
        x, y, w, h = bbox
        image = image.crop((x, y, x + w, y + h))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(orientation, dtype=torch.float32)