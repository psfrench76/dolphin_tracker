import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataloader import DolphinOrientationDataset
from utils.inc.settings import set_seed

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_root_dir = "../data/toy_orientations"
    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = OrientationResNet().to(device)
    criterion = model.compute_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "orientation_resnet.pth")
    print("Model saved to orientation_resnet.pth")

    # Save the final angles to a file
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    model.save_angles(all_outputs, "final_angles.txt")
    print("Final angles saved to final_angles.txt")

if __name__ == "__main__":
    main()