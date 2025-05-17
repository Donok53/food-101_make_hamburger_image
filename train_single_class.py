import sys
sys.path.append('/stylegan3')

import torch
import os
import argparse
from torchvision import transforms, datasets
from training.networks_stylegan3 import Generator
from torchvision.utils import save_image

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../dataset/food-101/food-101/images/', help='Path to dataset')
parser.add_argument('--output_path', type=str, default='./food-101_StyleGan3/outputs/', help='Output directory')
args = parser.parse_args()

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load Dataset
dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize StyleGAN3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator(
    z_dim=512,
    c_dim=0,
    w_dim=512,
    img_resolution=256,
    img_channels=3
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("사용 가능한 GPU 개수:", torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    print(f"GPUs 사용: {torch.cuda.device_count()}개")
    model = torch.nn.DataParallel(model)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to(device)
        
        # Generate noise
        z = torch.randn(images.size(0), 512).to(device)
        c = torch.zeros(images.size(0), 0).to(device)  # c_dim=0이므로 shape=(batch, 0)

        # Forward Pass
        generated_images = model(z, c)
        loss = torch.nn.MSELoss()(generated_images, images)
        
        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    # epoch 끝날 때마다 이미지 생성 및 저장
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, 512).to(device)
        c = torch.zeros(16, 0).to(device)
        fake_images = model(z, c)
        save_image(fake_images, os.path.join(args.output_path, f'generated_epoch_{epoch+1}.png'), normalize=True)
    model.train()
    # 모델 파라미터 저장
    torch.save(model.state_dict(), os.path.join(args.output_path, f'model_epoch_{epoch+1}.pth'))
