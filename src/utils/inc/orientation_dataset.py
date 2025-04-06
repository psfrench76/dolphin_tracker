import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .settings import settings
from .orientation_io import crop_image_with_bbox

class DolphinOrientationDataset(Dataset):

    """
    Annotations can be passed in as a YOLO output file. If it is left blank, it will load annotations from the dataset.
    images_index_file must also be passed.
    """
    def __init__(self, dataset_root_dir, annotations=None, images_index_file=None, transform=None):
        self.dataset_root_path = Path(dataset_root_dir)

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

        self.tracks_dir = self.dataset_root_path / settings['tracks_dir']
        tracks_files = sorted(self.tracks_dir.iterdir())
        self.tracks_index = {}
        for tracks_file in tracks_files:
            self.tracks_index[tracks_file.stem] = tracks_file

        if annotations:
            annotations_path = Path(annotations)
            images_index_path = Path(images_index_file)
            if not annotations_path.exists() or not images_index_path.exists():
                raise FileNotFoundError(f"Annotations or images index file not found: {annotations_path}, {images_index_path}")
            self.annotations = self._convert_annotations_from_yolo(annotations_path, images_index_path)
        else:
            self.annotations = self._load_annotations_from_dataset_dir(self.dataset_root_path)

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        orientation = self.annotations[idx]['orientation']
        track = self.annotations[idx]['track']
        img_path = self.annotations[idx]['image']
        bbox = self.annotations[idx]['bbox']
        image = crop_image_with_bbox(img_path, bbox, 'yolo')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(orientation, dtype=torch.float32), track, idx

    def get_image_path(self, idx):
        return self.annotations[idx]['image']

    # TODO: Modularize this and the following function to share functionality
    def _load_annotations_from_dataset_dir(self, dataset_root_path):
        annotations = []
        labels_dir_path = dataset_root_path / settings['labels_dir']
        for label_file_path in sorted(labels_dir_path.glob('*.txt')):
            if label_file_path.stem not in self.orientations_index:
                raise FileNotFoundError(f"Orientation file for {label_file_path.stem} not found.")
            else:
                orientation_file_path = self.orientations_index[label_file_path.stem]

            if label_file_path.stem not in self.tracks_index:
                raise FileNotFoundError(f"Track file for {label_file_path.stem} not found.")
            else:
                track_file_path = self.tracks_index[label_file_path.stem]

            image_path = self.images_index[label_file_path.stem]

            orientations = {int(label_index): [float(x), float(y)] for label_index, x, y in
                            [val.split() for val in orientation_file_path.read_text().splitlines()]}

            for label_index, (line, track) in enumerate(zip(label_file_path.read_text().splitlines(),
                                                          track_file_path.read_text().splitlines())):
                if label_index not in orientations:
                    continue

                track = int(track)
                # The following is still in the YOLO coordinate system (normalized from 0-1)
                _, x_center, y_center, width, height = map(float, line.split())

                bbox = [x_center, y_center, width, height]

                annotations.append({'image': image_path, 'bbox': bbox, 'orientation': orientations[label_index], 'track': track})

        return annotations


    def _convert_annotations_from_yolo(self, yolo_file, images_index_file):
        annotations = []
        with open(yolo_file, 'r') as file:
            with open(images_index_file, 'r') as index_file:
                for line, image_file_name in zip(file, index_file):
                    track, x_center, y_center, width, height = map(float, line.strip().split(',')[1:6])

                    image_path = self.dataset_root_path / settings['images_dir'] / image_file_name.strip()

                    bbox = [x_center, y_center, width, height]
                    orientation = [None, None]

                    annotations.append({'image': image_path, 'bbox': bbox, 'orientation': orientation, 'track': track})

        return annotations