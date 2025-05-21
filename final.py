#!/usr/bin/env python3
import sys
sys.path.append('/content/stylegan3')   # stylegan3 레포가 위치한 경로

import os
import random
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

from training.networks_stylegan3 import Generator  # stylegan3 코드 기준
from training.networks_stylegan2 import Discriminator  # stylegan3 코드 기준

def random_crop_batch(batch: torch.Tensor, patch_size: int):
    B, C, H, W = batch.shape
    patches = []
    for i in range(B):
        top   = random.randint(0, H - patch_size)
        left  = random.randint(0, W - patch_size)
        patch = batch[i:i+1, :, top:top+patch_size, left:left+patch_size]
        patches.append(patch)
    return torch.cat(patches, dim=0)

def hinge_d_loss(real_pred, fake_pred):
    loss_real = torch.mean(F.relu(1.0 - real_pred))
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    return loss_real + loss_fake

def hinge_g_loss(fake_pred):
    return -torch.mean(fake_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus',    type=int,   default=2, choices=[0,1],
                        help='사용할 GPU 개수 (1 또는 2)')
    parser.add_argument('--data_path',   type=str,
                        default='/dataset/food-101/food-101/images/hamburger',
                        help='단일 클래스 이미지 폴더 경로')
    parser.add_argument('--anyres_path', type=str,
                        default='/dataset/food-101/food-101/highres_images/hamburger/',
                        help='고해상도 이미지 폴더 경로')
    parser.add_argument('--output_path', type=str, default='./outputs_resume/',
                        help='모델/샘플 저장 폴더')
    parser.add_argument('--batch_size',  type=int,   default=8,
                        help='배치 크기 (기본: 8)')
    parser.add_argument('--epochs',      type=int,   default=2000)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--lambda_patch',type=float, default=1.0,
                        help='Patch 손실 가중치')
    args = parser.parse_args()

    # GPU 환경 설정
    gpu_ids = list(range(args.num_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_path, exist_ok=True)

    # DataLoader 설정
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset_global = datasets.ImageFolder(root=args.data_path, transform=transform)
    loader_global = DataLoader(dataset_global,
                               batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    if os.path.isdir(args.anyres_path):
        dataset_hr = datasets.ImageFolder(root=args.anyres_path, transform=transform)
    else:
        print(f'Warning: anyres_path "{args.anyres_path}" not found. Falling back.')
        dataset_hr = datasets.ImageFolder(root=args.data_path, transform=transform)
    loader_hr = DataLoader(dataset_hr,
                           batch_size=args.batch_size,
                           shuffle=True, drop_last=True)

    # 모델 초기화
    G        = Generator(512, 0, 512, img_resolution=256, img_channels=3)
    D_global = Discriminator(0, 256, 3)
    D_patch  = Discriminator(0, 256, 3)

    # 멀티 GPU 설정
    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        G        = nn.DataParallel(G)
        D_global = nn.DataParallel(D_global)
        D_patch  = nn.DataParallel(D_patch)

    # 장치로 이동
    G.to(device)
    D_global.to(device)
    D_patch.to(device)

    # 옵티마이저
    optim_G  = torch.optim.Adam(G.parameters(),        lr=args.lr, betas=(0.0, 0.99))
    optim_Dg = torch.optim.Adam(D_global.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optim_Dp = torch.optim.Adam(D_patch.parameters(),  lr=args.lr, betas=(0.0, 0.99))

    # 체크포인트 로드
    start_epoch = 1
    ckpt_path = os.path.join(args.output_path, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        print('==> 이전 체크포인트 로드:', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt['G'])
        D_global.load_state_dict(ckpt['Dg'])
        D_patch.load_state_dict(ckpt['Dp'])
        optim_G.load_state_dict(ckpt['optG'])
        optim_Dg.load_state_dict(ckpt['optDg'])
        optim_Dp.load_state_dict(ckpt['optDp'])
        start_epoch = ckpt['epoch'] + 1
        print(f'==> 에폭 {start_epoch}부터 재개')

    z_dim = 512
    for epoch in range(start_epoch, args.epochs + 1):
        G.train(); D_global.train(); D_patch.train()
        for (real_imgs, _), (hr_imgs, _) in zip(loader_global, loader_hr):
            real_imgs = real_imgs.to(device)
            hr_imgs   = hr_imgs.to(device)

            # Discriminator 업데이트
            z = torch.randn(real_imgs.size(0), z_dim, device=device)
            fake_full    = G(z, None)
            fake_global  = F.interpolate(fake_full, size=(256,256),
                                         mode='bilinear', align_corners=False)

            d_real = D_global(real_imgs, None)
            d_fake = D_global(fake_global.detach(), None)
            loss_Dg = hinge_d_loss(d_real, d_fake)

            real_patches = random_crop_batch(hr_imgs if random.random()<0.5 else real_imgs, 256)
            fake_patches = random_crop_batch(fake_global.detach(), 256)
            dp_real = D_patch(real_patches, None)
            dp_fake = D_patch(fake_patches.detach(), None)
            loss_Dp = hinge_d_loss(dp_real, dp_fake)

            optim_Dg.zero_grad(); loss_Dg.backward(); optim_Dg.step()
            optim_Dp.zero_grad(); loss_Dp.backward(); optim_Dp.step()

            # Generator 업데이트
            fake_full    = G(z, None)
            fake_global  = F.interpolate(fake_full, size=(256,256),
                                         mode='bilinear', align_corners=False)
            loss_Gg = hinge_g_loss(D_global(fake_global, None))
            fake_patches_G = random_crop_batch(fake_global, 256)
            loss_Gp = hinge_g_loss(D_patch(fake_patches_G, None))

            loss_G = loss_Gg + args.lambda_patch * loss_Gp
            optim_G.zero_grad(); loss_G.backward(); optim_G.step()

        # 로그 & 샘플 저장
        print(f'[Epoch {epoch}/{args.epochs}] '
            f'Loss_Dg: {loss_Dg.item():.4f}, '
            f'Loss_Dp: {loss_Dp.item():.4f}, '
            f'Loss_Gg: {loss_Gg.item():.4f}, '
            f'Loss_Gp: {loss_Gp.item():.4f}')
        G.eval()
        with torch.no_grad():
            z = torch.randn(16, z_dim, device=device)
            fake_full   = G(z, None)
            fake_sample = F.interpolate(fake_full, size=(256,256),
                                        mode='bilinear', align_corners=False)
            save_image(fake_sample,
                       os.path.join(args.output_path, f'fake_epoch_{epoch:03d}.png'),
                       nrow=4, normalize=True, value_range=(-1,1))

        # 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'G':   G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict(),
            'Dg':  D_global.module.state_dict() if isinstance(D_global, nn.DataParallel) else D_global.state_dict(),
            'Dp':  D_patch.module.state_dict() if isinstance(D_patch, nn.DataParallel) else D_patch.state_dict(),
            'optG': optim_G.state_dict(),
            'optDg': optim_Dg.state_dict(),
            'optDp': optim_Dp.state_dict(),
        }, ckpt_path)

        torch.save(G.module.state_dict() if isinstance(G, nn.DataParallel) else G.state_dict(),
                   os.path.join(args.output_path, f'G_epoch_{epoch:03d}.pth'))
        G.train()

    print('Training finished.')

if __name__ == '__main__':
    main()

