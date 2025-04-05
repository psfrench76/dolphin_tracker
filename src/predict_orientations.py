from .utils.inc.orientation_network import OrientationResNet
from .utils.inc.orientation_dataloader import DolphinOrientationDataset

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize to mean and standard deviations of RGB values for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TODO: parameterize this, etc
dummy_dataset = "../data/toy_100"
dataset = DolphinOrientationDataset(dataset_root_dir=dummy_dataset, transform=transform)

model = OrientationResNet()
model.to(device)

for images, targets in dataset:
    images.to(device)
    targets.to(device)
    outputs = model(images)