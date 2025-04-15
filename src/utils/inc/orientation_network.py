import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pandas as pd
from tqdm import tqdm

class OrientationResNet(nn.Module):
    def __init__(self):
        super(OrientationResNet, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.loss_fn = nn.MSELoss()
        self.device = None

    def forward(self, x):
        x = self.resnet(x)
        return x

    def set_device(self, device):
        self.device = device
        self.to(device)

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

    def predict(self, dataloader):
        self.eval()
        all_outputs = []
        all_indices = []
        all_tracks = []
        with torch.no_grad():
            for images, _, tracks, idxs in tqdm(dataloader, desc="Predicting orientations", unit="batch"):
                images = images.to(self.device)
                outputs = self(images)
                all_outputs.append(outputs.cpu())
                all_indices.append(idxs)
                all_tracks.append(tracks)
        return torch.cat(all_outputs, dim=0), torch.cat(all_indices, dim=0).cpu().numpy(), torch.cat(all_tracks, dim=0).cpu().numpy()

    def calculate_angles(self, outputs):
        x_val, y_val = outputs[:, 0], outputs[:, 1]
        angle = torch.atan2(y_val, x_val) * (180 / torch.pi)  # Convert radians to degrees
        return angle

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)

    # TODO: make this prettier, probably have first two columns filename and object ID. May need to build this into a separate IO module for the sake of sanity
    # other_data should already be a pandas dataframe, with column names
    def write_outputs(self, outputs, other_data, file_path):
        angles = self.calculate_angles(outputs)
        outputs = outputs.cpu()
        angles = angles.unsqueeze(1).cpu()
        # print(outputs.shape, angles.shape, other_data.shape)
        all_output = torch.cat([angles, outputs], dim=1)
        df = pd.DataFrame(all_output.numpy())
        df.columns = ['angle', 'x_val', 'y_val']
        df = pd.concat([df, other_data], axis=1)
        df.sort_values(by=['filename', 'object_id'], inplace=True)

        df.to_csv(file_path, index=False, header=True)
