import torch
from stylegan3 import StyleGAN3
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StyleGAN3().to(device)
model.load_state_dict(torch.load('./outputs/model_epoch_50.pth'))

# Generate synthetic images
model.eval()
with torch.no_grad():
    noise = torch.randn(16, 512).to(device)
    c = torch.zeros(noise.size(0), 0).to(device)  # c_dim=0이므로 shape=(batch, 0)
    fake_images = model(noise, c)
    save_image(fake_images, './outputs/generated.png', normalize=True)
