import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .settings import settings

class DolphinOrientationDataset(Dataset):

    """
    Annotations can be passed in as a YOLO output file. If it is left blank, it will load annotations from the dataset.
    images_index_file must also be passed.
    """
    def __init__(self, dataset_root_dir, annotations=None, images_index_file=None, transform=None):
        self.dataset_root_path = Path(dataset_root_dir)
        if annotations:
            annotations_path = Path(annotations)
            images_index_path = Path(images_index_file)
            if not annotations_path.exists() or not images_index_path.exists():
                raise FileNotFoundError(f"Annotations or images index file not found: {annotations_path}, {images_index_path}")
            self.annotations = self._convert_annotations_from_yolo(annotations_path, images_index_path)
        else:
            self.annotations = self._load_annotations_from_dataset_dir(dataset_root_dir)


        self.image_dir = self.dataset_root_path / settings['images_dir']
        image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix in settings['image_file_extensions']])
        self.images_index = {}
        for image_file in image_files:
            self.images_index[image_file.stem] = image_file

        self.orientations_dir = self.dataset_root_path / settings['orientations_dir']
        orientations_files = sorted(self.orientations_dir.iterdir())
        self.orientations_index = {}
        for orientations_file in orientations_files:
            self.orientations_index[orientations_file.stem] = orientations_file

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['image']
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        bbox = self.annotations[idx]['bbox']
        orientation = self.annotations[idx]['orientation']

        # Crop the image using the bounding box
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox

        x1 = int(x_top_left * width)
        y1 = int(y_top_left * height)
        x2 = int(x_bottom_right * width)
        y2 = int(y_bottom_right * height)

        image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(orientation, dtype=torch.float32)


    def _load_annotations_from_dataset_dir(self, dataset_root_path):
        annotations = []
        labels_dir_path = dataset_root_path / settings['labels_dir']
        for label_file_path in sorted(labels_dir_path.glob('*.txt')):
            if label_file_path.stem not in self.orientations_index:
                raise FileNotFoundError(f"Orientation file for {label_file_path.stem} not found.")
            else:
                orientation_file_path = self.orientations_index[label_file_path.stem]

            image_path = self.images_index[label_file_path.stem]

            for line, orientation in zip(label_file_path.read_text().splitlines(),
                                         orientation_file_path.read_text().splitlines()):

                # The following is still in the YOLO coordinate system (normalized from 0-1)
                _, x_center, y_center, width, height = map(float, line.split())

                x_top_left = x_center - width / 2
                y_top_left = y_center - height / 2
                x_bottom_right = x_center + width / 2
                y_bottom_right = y_center + height / 2

                bbox = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
                orientation_x, orientation_y = map(float, orientation.split())

                annotations.append({'image': image_path, 'bbox': bbox, 'orientation': [orientation_x, orientation_y]})

        return annotations


    def _convert_annotations_from_yolo(self, yolo_file, images_index_file):
        annotations = []
        with open(yolo_file, 'r') as file:
            with open(images_index_file, 'r') as index_file:
                for line, image_file_name in zip(file, index_file):
                    x_center, y_center, width, height = map(float, line.strip().split(',')[2:6])

                    x_top_left = x_center - width / 2
                    y_top_left = y_center - height / 2
                    x_bottom_right = x_center + width / 2
                    y_bottom_right = y_center + height / 2

                    image_path = self.dataset_root_path / settings['images_dir'] / image_file_name.strip()
                    bbox = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
                    orientation = [None, None]

                    annotations.append({'image': image_path, 'bbox': bbox, 'orientation': orientation})

        return annotations