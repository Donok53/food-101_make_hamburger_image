import torch
import os
import argparse
from torchvision import transforms, datasets
from stylegan3 import StyleGAN3

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('../dataset/food-101/food-101/images', type=str, default='./data/', help='Path to high-resolution dataset')
parser.add_argument('./food-101_StyleGan3/outputs', type=str, default='./outputs/', help='Output directory')
args = parser.parse_args()

# Data Preprocessing
transform = transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load Dataset
dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StyleGAN3().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to(device)
        
        # Generate noise
        noise = torch.randn(images.size(0), 512).to(device)

        # Forward Pass
        generated_images = model(noise)
        loss = torch.nn.MSELoss()(generated_images, images)

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
            torch.save(model.state_dict(), os.path.join(args.output_path, f'anyres_epoch_{epoch+1}.pth'))
