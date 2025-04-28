import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.amp import GradScaler, autocast
from .reporting import OrientationMetrics

class OrientationResNet(nn.Module):
    def __init__(self, weights=None, device=None):
        super(OrientationResNet, self).__init__()
        self.device = device
        load_args = {'weights_only': True}

        if self.device is not None:
            self.to(self.device)
            load_args['map_location'] = device

        if weights is not None:
            self.resnet = models.resnet18()
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
            self.load_state_dict(torch.load(weights, **load_args))
        else:
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.resnet(x)
        return x

    def set_device(self, device):
        self.device = device
        self.to(device)

    def freeze_layers(self, num_layers_to_freeze):
        """
        Freezes the first `num_layers_to_freeze` layers of the ResNet model.
        """
        layers = list(self.resnet.children())
        for i, layer in enumerate(layers[:num_layers_to_freeze]):
            for param in layer.parameters():
                param.requires_grad = False

    def train_model(self, dataloader, optimizer):
        self.train()
        running_loss = 0.0
        for images, targets, _, _ in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self(images)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(dataloader.dataset)

    def train_model_scaled(self, train_dataloader, optimizer):
        self.train()
        scaler = GradScaler(str(self.device))
        total_loss = 0.0

        for images, orientations, tracks, indices in train_dataloader:
            images, orientations = images.to(self.device), orientations.to(self.device)

            optimizer.zero_grad()

            with autocast(str(self.device)):
                outputs = self(images)
                loss = self.compute_loss(outputs, orientations)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_dataloader)

    def validate_model(self, dataloader):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, targets, _, _ in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self(images)
                loss = self.compute_loss(outputs, targets)
                running_loss += loss.item() * images.size(0)
        return running_loss / len(dataloader.dataset)

    def predict(self, dataloader, outfile_path):
        all_outputs, _, all_indices, all_tracks = self._get_predictions(dataloader)
        all_filenames = [str(dataloader.dataset.get_image_path(idx).stem) for idx in all_indices]
        data = {'dataloader_index': all_indices, 'filename': all_filenames, 'object_id': all_tracks, }
        other_df = pd.DataFrame(data)

        pred_df = self._consolidate_data(all_outputs, other_df)
        pred_df.to_csv(outfile_path, index=False)
        print(f"Final angles saved to {outfile_path}")

        return pred_df

    def evaluate(self, dataloader, outfile_path):
        all_outputs, all_gt_orientations, all_indices, all_tracks = self._get_predictions(dataloader)
        all_filenames = [str(dataloader.dataset.get_image_path(idx).stem) for idx in all_indices]
        data = {'dataloader_index': all_indices, 'filename': all_filenames, 'object_id': all_tracks, }
        other_df = pd.DataFrame(data)

        pred_df = self._consolidate_data(all_outputs, other_df)
        pred_df.to_csv(outfile_path, index=False)
        print(f"Final angles saved to {outfile_path}")

        gt_data = {'dataloader_index': all_indices, 'filename': all_filenames, 'object_id': all_tracks,
                   'x_val': all_gt_orientations[:, 0], 'y_val': all_gt_orientations[:, 1]}
        gt_df = pd.DataFrame(gt_data)
        gt_df['angle'] = self.calculate_angles_pd(gt_df['x_val'], gt_df['y_val'])

        om = OrientationMetrics(pred_df, gt_df)
        om.calculate_metrics()
        return om

    def _get_predictions(self, dataloader):
        self.eval()
        all_outputs = []
        all_indices = []
        all_tracks = []
        all_gt_orientations = []
        with torch.no_grad():
            for images, orientations, tracks, idxs in tqdm(dataloader, desc="Predicting orientations", unit="batch"):
                images = images.to(self.device)
                outputs = self(images)
                all_outputs.append(outputs.cpu())
                all_indices.append(idxs)
                all_tracks.append(tracks)
                all_gt_orientations.append(orientations)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_gt_orientations = torch.cat(all_gt_orientations, dim=0).cpu().numpy()
        all_indices = torch.cat(all_indices, dim=0).cpu().numpy()
        all_tracks = torch.cat(all_tracks, dim=0).cpu().numpy()
        return all_outputs, all_gt_orientations, all_indices, all_tracks

    def calculate_angles(self, outputs):
        x_val, y_val = outputs[:, 0], outputs[:, 1]
        angle = torch.atan2(y_val, x_val) * (180 / torch.pi)  # Convert radians to degrees
        return angle

    def calculate_angles_pd(self, x_col, y_col):
        return np.arctan2(y_col, x_col) * (180 / np.pi)

    def compute_loss(self, predictions, targets):
        # print(f"Predictions: {predictions}")
        # print(f"Targets: {targets}")
        return self.loss_fn(predictions, targets)

    # TODO: do column sorting
    # other_data should already be a pandas dataframe, with column names
    def _consolidate_data(self, outputs, other_data=None):
        angles = self.calculate_angles(outputs)
        outputs = outputs.cpu()
        angles = angles.unsqueeze(1).cpu()
        # print(outputs.shape, angles.shape, other_data.shape)
        all_output = torch.cat([angles, outputs], dim=1)
        df = pd.DataFrame(all_output.numpy())
        df.columns = ['angle', 'x_val', 'y_val']
        if other_data is not None:
            df = pd.concat([df, other_data], axis=1)
        df.sort_values(by=['filename', 'object_id'], inplace=True)

        return df