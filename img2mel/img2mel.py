import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import soundfile as sf
import numpy as np
from img2mel.preprocessor import preprocess_image, preprocess_audio
from tqdm import tqdm

class Img2MelNet(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Img2MelNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 파일 매칭
    image_files = [f for f in os.listdir('data') if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    epochs = 6

    for epoch in range(epochs):
        total_loss = 0
        for img_file in tqdm(image_files):
            base_name = os.path.splitext(img_file)[0]
            audio_file = base_name + '.wav'

            img_pt_path = f'data/{base_name}_img.pt'
            mel_pt_path = f'data/{base_name}_mel.pt'

            # .pt 파일이 있으면 전처리 스킵
            if os.path.exists(img_pt_path) and os.path.exists(mel_pt_path):
                img = torch.load(img_pt_path).unsqueeze(0).to(device)  # [1, 3, 224, 224]
                mel = torch.load(mel_pt_path).unsqueeze(0).to(device)  # [1, 1, 224, 224]
            else:
                if not os.path.exists(f'data/{audio_file}'):
                    continue  # 오디오도 없으면 스킵
                img = preprocess_image(f'data/{img_file}').unsqueeze(0).to(device)  # [1, 3, 224, 224]
                mel = preprocess_audio(f'data/{audio_file}').unsqueeze(0).to(device)  # [1, 1, 224, 224]

            # 모델 학습
            optimizer.zero_grad()
            output = model(img)  # output shape: [1, 1, 224, 224]
            loss = criterion(output, mel)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

    torch.save(model.state_dict(), 'img2mel.pth')
    print('✅ 모델 저장 완료: img2mel.pth')

def generate():
    # mel → waveform 변환 함수
    def mel_to_audio(mel_tensor, sr=22050, n_fft=2048, hop_length=512):
        mel_db = mel_tensor.squeeze().numpy()  # [128, T]
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return audio

    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Img2MelNet().to(device)
    model.load_state_dict(torch.load('img2mel.pth', map_location=device))
    model.eval()

    # 테스트 이미지 경로
    TEST_IMG = input('테스트할 이미지 경로를 입력하세요: ')
    OUTPUT_WAV = TEST_IMG.split('.')[0] + '.wav'
    SR = 22050

    # 이미지 전처리
    img_tensor = preprocess_image(TEST_IMG).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 추론
    with torch.no_grad():
        mel_pred = model(img_tensor).cpu()  # [1, 1, 224, 224]

    # 시간 축 크기 128 → 431으로 리사이즈 (10초 기준)
    mel_pred_resized = F.interpolate(mel_pred, size=(128, 431), mode='bilinear', align_corners=False)
    mel_pred_resized = mel_pred_resized.squeeze(0).squeeze(0)  # [128, original_time_frames]

    # mel → wav
    audio = mel_to_audio(mel_pred_resized)  # [128, original_time_frames] → waveform
    sf.write(OUTPUT_WAV, audio, SR)

    print(f'✅ 변환 완료: {OUTPUT_WAV}')

if __name__ == '__main__':
    input_mode = input('모드 선택 (train/generate): ')
    if input_mode == 'train':
        train_model()
    elif input_mode == 'generate':
        generate()
    else:
        print('잘못된 모드입니다. train 또는 generate를 선택하세요.')