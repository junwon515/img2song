import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import librosa
import soundfile as sf
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    
class Img2Mel:
    def __init__(self, model_path='img2mel.pth', device='cpu'):
        """
        Img2Mel 클래스 초기화

        Args:
            model_path (str): 모델 파일 경로, 기본값은 'img2mel.pth'
            device (str): 'cpu' 또는 'cuda' (GPU 사용 시)
        """
        self.model_path = model_path
        self.device = device
        self.model = Img2MelNet().to(self.device)
        if not os.path.exists(model_path):
            print(f'Model file not found at {model_path}. Please train the model first.')
            return
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f'Model loaded successfully from {model_path}.')
        except Exception as e:
            print(f'Failed to load model from {model_path}: {e}')

    def train(self, epochs=10, data_dir='data'):
        """
        모델 학습 메서드
        
        Args:
            epochs (int): 학습 에폭 수, 기본값은 10
            data_dir (str): 학습 데이터 디렉토리 경로, 기본값은 'data'
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        data_samples = self.preprocess_all(data_dir)
        if not data_samples:
            print(f'No valid data samples found in {data_dir}. Please check your data.')
            return
        
        for epoch in range(epochs):
            total_loss = 0
            for img_tensor, mel_tensor in tqdm(data_samples, desc=f'Epoch {epoch+1}/{epochs}'):
                img_tensor = img_tensor.to(self.device)
                mel_tensor = mel_tensor.to(self.device)

                optimizer.zero_grad()
                mel_pred = self.model(img_tensor)
                loss = criterion(mel_pred, mel_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_samples)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model trained and saved to {self.model_path}.')

    def generate(self, image_path, output_wav='output.wav', sr=22050, duration=10):
        """
        이미지로부터 멜 스펙트로그램을 생성하고 오디오 파일로 변환

        Args:
            image_path (str): 입력 이미지 파일 경로
            output_wav (str): 출력 오디오 파일 경로, 기본값은 'output.wav'
            sr (int): 샘플링 레이트, 기본값은 22050
            duration (int): 오디오 길이(초), 기본값은 10초
        """
        if not os.path.exists(self.model_path):
            print(f'Model file not found at {self.model_path}. Please train the model first.')
            return
        if not os.path.exists(image_path):
            print(f'Image file does not exist at path: {image_path}')
            return
        
        img_tensor = self.preprocess_image(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mel_pred = self.model(img_tensor).cpu()

        time_frame = ((sr * duration) // 512) + 1
        mel_pred_resized = F.interpolate(mel_pred, size=(128, time_frame), mode='bilinear', align_corners=False)
        mel_pred_resized = mel_pred_resized.squeeze(0).squeeze(0) # [128, time_frame]
        mel_db = mel_pred_resized.numpy()
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=2048, hop_length=512)
        sf.write(output_wav, audio, sr)
        print(f'Generated audio saved to {output_wav}.')

    def preprocess_all(self, data_dir='data'):
        img_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

        data_samples = []
        for img_file in tqdm(img_files):
            base_id = os.path.splitext(img_file)[0]
            audio_file = base_id + '.wav'
            if audio_file in audio_files:
                img_path = os.path.join(data_dir, img_file)
                audio_path = os.path.join(data_dir, audio_file)
                try:
                    img_tensor = self.preprocess_image(img_path).unsqueeze(0) # [1, 3, 224, 224]
                    mel_tensor = self.preprocess_audio(audio_path).unsqueeze(0) # [1, 1, 224, 224]
                    data_samples.append((img_tensor, mel_tensor))
                except Exception as e:
                    print(f'Error processing {img_file} or {audio_file}: {e}')
            else:
                print(f'No matching audio file for {img_file}. Skipping.')
        return data_samples

    def preprocess_image(self, image_path) -> torch.Tensor:
        """
        이미지 파일을 텐서로 변환하는 메서드

        Args:
            image_path (str): 이미지 파일 경로

        Returns:
            torch.Tensor: 변환된 이미지 텐서, shape=[3, 224, 224]
        """
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        return transform(image)  # [3, 224, 224]

    def preprocess_audio(self, audio_path, sr=22050, duration=10) -> torch.Tensor:
        """
        오디오 파일을 멜 스펙트로그램 텐서로 변환하는 메서드

        Args:
            audio_path (str): 오디오 파일 경로
            sr (int): 샘플링 레이트, 기본값은 22050
            duration (int): 오디오 길이(초), 기본값은 10초

        Returns:
            torch.Tensor: 변환된 멜 스펙트로그램 텐서, shape=[1, 224, 224]
        """
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)  # [1,1,128,time]
        mel_resized = F.interpolate(mel_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return mel_resized.squeeze(0)  # [1, 224, 224]
    
if __name__ == '__main__':
    try:
        from util import get_device
        img2mel = Img2Mel(device=get_device())
        # img2mel.train()
        user_input = input('Enter image path: ')
        img2mel.generate(user_input)
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    except Exception as e:
        print(f'Error: {e}')