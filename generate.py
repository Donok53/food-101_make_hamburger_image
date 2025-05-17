#!/usr/bin/env python3
import sys
import os
import torch
from torchvision.utils import save_image

# stylegan3 레포지토리 위치
sys.path.append('/stylegan3')
import legacy     # stylegan3 제공 로더

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 사전 학습된 Generator 불러오기
    pkl = '/stylegan3/pretrained/stylegan3-afhqv2-512x512.pkl'  # 실제 경로로 수정
    with open(pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()

    # 2) 원하는 개수만큼 랜덤 샘플 생성
    n_samples = 16
    z = torch.randn(n_samples, G.z_dim, device=device)
    c = None  # 조건(label) 없을 때는 None

    with torch.no_grad():
        # 완전 생성 (mapping + synthesis)
        imgs = G(z, c)  # shape: [16, 3, 512, 512]

    # 3) 출력 디렉터리
    out_dir = './gen_samples'
    os.makedirs(out_dir, exist_ok=True)

    # 4) 이미지 저장 (512→256 리사이즈해서 저장하고 싶으면 아래 주석 해제)
    # imgs = torch.nn.functional.interpolate(imgs, size=(256,256), mode='bilinear', align_corners=False)

    save_image(
        imgs,
        os.path.join(out_dir, 'sample_%04d.png'),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f'Saved {n_samples} samples to {out_dir}')

if __name__ == '__main__':
    main()
