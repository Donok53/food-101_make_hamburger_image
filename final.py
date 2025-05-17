#!/usr/bin/env python3
import sys
sys.path.append('/stylegan3')   # stylegan3 레포가 위치한 경로

import os
import random
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

from training.networks_stylegan3 import Generator, Discriminator  # stylegan3 코드 기준

# ─────────────────────────────────────────────────────────────────────────────
#            Helper: 배치 단위로 랜덤 패치 자르기 (B×C×H×W → B×C×P×P)
def random_crop_batch(batch: torch.Tensor, patch_size: int):
    B, C, H, W = batch.shape
    patches = []
    for i in range(B):
        top   = random.randint(0, H - patch_size)
        left  = random.randint(0, W - patch_size)
        patch = batch[i:i+1, :, top:top+patch_size, left:left+patch_size]
        patches.append(patch)
    return torch.cat(patches, dim=0)
# ─────────────────────────────────────────────────────────────────────────────

def hinge_d_loss(real_pred, fake_pred):
    # D 손실 (Hinge)
    loss_real = torch.mean(F.relu(1.0 - real_pred))
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    return loss_real + loss_fake

def hinge_g_loss(fake_pred):
    # G 손실 (Hinge)
    return -torch.mean(fake_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    type=str,
                        default='../dataset/food-101/food-101/images/hamburger',
                        help='단일 클래스(예: hamburger) 이미지 폴더 경로')
    parser.add_argument('--anyres_path',  type=str,
                        default='../dataset/food-101/food-101/highres_images/hamburger/',
                        help='고해상도 이미지 폴더 경로')
    parser.add_argument('--output_path',  type=str, default='./food-101_StyleGan3/outputs_100/',
                        help='모델/샘플 저장 폴더')
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--epochs',       type=int,   default=2000)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--lambda_patch', type=float, default=1.0,
                        help='Patch Loss 가중치')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ───────────────── DataLoader ─────────────────
    transform_global = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset_global = datasets.ImageFolder(root=args.data_path,
                                          transform=transform_global)
    loader_global = DataLoader(dataset_global,
                               batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    # 고해상도 Any-resolution 이미지는 transform 없이 로드
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    # dataset_hr = datasets.ImageFolder(root=args.anyres_path,
    #                                   transform=transform_hr)

    if os.path.isdir(args.anyres_path):
        dataset_hr = datasets.ImageFolder(root=args.anyres_path,
                                          transform=transform_hr)
    else:
        print(f'Warning: anyres_path "{args.anyres_path}" not found. Falling back to data_path.')
        # fallback 시 global 전처리(Resize 256×256) 사용
        dataset_hr = datasets.ImageFolder(root=args.data_path,
                                         transform=transform_global)


    loader_hr = DataLoader(dataset_hr,
                           batch_size=args.batch_size,
                           shuffle=True, drop_last=True)
    # ────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────
    #    완전 Scratch 상태로 Generator 초기화
    G = Generator(
        z_dim=512, c_dim=0, w_dim=512,
        img_resolution=256, img_channels=3
    ).to(device)
    # 멀티-GPU 지원
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Generator")
        G = nn.DataParallel(G)
 
    # 만약 256×256으로 샘플을 보고 싶다면 추가로 리사이즈하세요.

    D_global = Discriminator(
        c_dim=0, img_resolution=256, img_channels=3
    ).to(device)

    # 멀티-GPU 지원
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Global Discriminator")
        D_global = nn.DataParallel(D_global)

    D_patch  = Discriminator(
        c_dim=0, img_resolution=256, img_channels=3
    ).to(device)

    # 멀티-GPU 지원
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Patch Discriminator")
        D_patch = nn.DataParallel(D_patch)

    optim_G  = torch.optim.Adam(G.parameters(),        lr=args.lr, betas=(0.0,0.99))
    optim_Dg = torch.optim.Adam(D_global.parameters(), lr=args.lr, betas=(0.0,0.99))
    optim_Dp = torch.optim.Adam(D_patch.parameters(),   lr=args.lr, betas=(0.0,0.99))
    # ────────────────────────────────────────────────

    z_dim = 512

    for epoch in range(1, args.epochs+1):
        G.train(); D_global.train(); D_patch.train()

        for (real_imgs, _), (hr_imgs, _) in zip(loader_global, loader_hr):
            real_imgs = real_imgs.to(device)       # 256×256 global images
            hr_imgs   = hr_imgs.to(device)         # high-res original

            # ---- 1) Discriminator 업데이트 ----
            # 1.1) Fake 생성 (512→256 리사이즈)
            z = torch.randn(real_imgs.size(0), z_dim, device=device)
            c = None
            fake_full = G(z, c)  # [B, 3, 512, 512]
            fake_global = F.interpolate(
                fake_full, size=(256,256),
                mode='bilinear', align_corners=False
            )                 # [B, 3, 256, 256]

            # 1.2) Global D 손실
            d_real = D_global(real_imgs, None)
            d_fake = D_global(fake_global.detach(), None)
            loss_Dg = hinge_d_loss(d_real, d_fake)

            # 1.3) Patch D 손실 (256×256 패치)
            #    real global 이미지에서 크롭 vs fake에서도 동일 크롭
            real_patches = random_crop_batch(hr_imgs if random.random()<0.5 else real_imgs, 256)
            fake_patches = random_crop_batch(fake_global.detach(), 256)

            dp_real = D_patch(real_patches, None)
            dp_fake = D_patch(fake_patches.detach(), None)
            loss_Dp = hinge_d_loss(dp_real, dp_fake)

            # 1.4) 역전파 및 스텝
            optim_Dg.zero_grad(); loss_Dg.backward(); optim_Dg.step()
            optim_Dp.zero_grad(); loss_Dp.backward(); optim_Dp.step()

            # ---- 2) Generator 업데이트 ----
            fake_full = G(z, c)
            fake_global = F.interpolate(
                fake_full, size=(256,256),
                mode='bilinear', align_corners=False
            )
            dg_fake_for_g = D_global(fake_global, None)
            loss_Gg = hinge_g_loss(dg_fake_for_g)

            fake_patches_G = random_crop_batch(fake_global, 256)
            dp_fake_for_g  = D_patch(fake_patches_G, None)
            loss_Gp = hinge_g_loss(dp_fake_for_g)

            loss_G = loss_Gg + args.lambda_patch * loss_Gp
            optim_G.zero_grad(); loss_G.backward(); optim_G.step()

        # ─── Logging & 샘플 저장 ─────────────────────
        print(f'[Epoch {epoch}/{args.epochs}] '
              f'Loss_Dg: {loss_Dg.item():.4f}, '
              f'Loss_Dp: {loss_Dp.item():.4f}, '
              f'Loss_Gg: {loss_Gg.item():.4f}, '
              f'Loss_Gp: {loss_Gp.item():.4f}')

        # 매 epoch마다 샘플 이미지 저장
        G.eval()
        with torch.no_grad():
            z = torch.randn(16, z_dim, device=device)
            fake_full = G(z, None)
            fake_sample = F.interpolate(
                fake_full, size=(256,256),
                mode='bilinear', align_corners=False
            )
            save_image(fake_sample,
                       os.path.join(args.output_path, f'fake_epoch_{epoch:03d}.png'),
                       nrow=4, normalize=True, value_range=(-1,1))
            # DataParallel 로 감쌌다면 .module 로 실제 Generator를 꺼내서 저장
            state_dict = G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict()
            torch.save(state_dict, 
                        os.path.join(args.output_path, f'G_epoch_{epoch:03d}.pth'))
        G.train()

    print('Training finished.')

if __name__ == '__main__':
    main()
