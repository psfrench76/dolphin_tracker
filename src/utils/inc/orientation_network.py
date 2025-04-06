import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pandas as pd

class OrientationResNet(nn.Module):
    def __init__(self):
        super(OrientationResNet, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.resnet(x)
        return x

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
        print(df)
        df.to_csv(file_path, index=False, header=True)
