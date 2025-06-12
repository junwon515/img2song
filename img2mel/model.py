import torch
import torch.nn as nn


class Img2MelNet(nn.Module):
    def __init__(self):
        """
        간단한 이미지-멜 스펙트로그램 변환 모델 정의
        - Encoder: 5개의 Conv2d 레이어로 구성
        - Decoder: 5개의 ConvTranspose2d 레이어로 구성
        - 입력: RGB 이미지 (3채널, 224x224)
        - 출력: 멜 스펙트로그램 (1채널, 224x224)
        - 특징: 각 Conv 레이어 후 ReLU 활성화 함수 사용
        - Conv 레이어는 stride=2로 다운샘플링, ConvTranspose 레이어는 stride=2로 업샘플링 
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),   # 224 → 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # 112 → 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 56 → 28
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 28 → 14
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 14 → 7
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 14 → 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 28 → 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   # 56 → 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),    # 112 → 224
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파 정의

        Args:
            x (torch.Tensor): 입력 이미지 텐서, shape=[batch_size, 3, 224, 224]
        
        Returns:
            torch.Tensor: 출력 멜 스펙트로그램 텐서, shape=[batch_size, 1, 224, 224]
        """
        z = self.encoder(x)
        return self.decoder(z)
