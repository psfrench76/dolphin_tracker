import torch
import torch.nn as nn
import torchvision.models as models

class OrientationResNet(nn.Module):
    def __init__(self):
        super(OrientationResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.resnet(x)
        return x

    def predict_orientation(self, x):
        outputs = self.forward(x)
        x_val, y_val = outputs[:, 0], outputs[:, 1]
        angle = torch.atan2(y_val, x_val) * (180 / torch.pi)  # Convert radians to degrees
        return angle

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)