from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataloader import DolphinOrientationDataset
from utils.inc.settings import set_seed

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(0)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize to mean and standard deviations of RGB values for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TODO: parameterize this, etc
dummy_dataset = "../data/toy_orientations"
dataset = DolphinOrientationDataset(dataset_root_dir=dummy_dataset, transform=transform)

# # Inspect a few samples
# for i in range(min(5, len(dataset))):
#     images, targets = dataset[i]
#     print(f"Sample {i}:")
#     print(f"Image shape: {images.shape}")
#     print(f"Targets: {targets}")

model = OrientationResNet()
model.to(device)

for images, _ in dataset:
    images = images.unsqueeze(0).to(device)
    outputs = model(images)

    print(outputs)