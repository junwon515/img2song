import os
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
AUDIO_EXTENSION = '.wav'
DATA_DIR = 'data'

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return transform(image)  # [3, 224, 224]

# 오디오 전처리 함수
def preprocess_audio(audio_path, sr=22050, duration=10):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)  # [1,1,224,time]
    mel_resized = F.interpolate(mel_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return mel_resized.squeeze(0)  # [1, 224, 224]

# 전체 자동 처리
def preprocess_all():
    files = os.listdir(DATA_DIR)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

    for img_file in tqdm(image_files):
        base_id = os.path.splitext(img_file)[0]
        audio_file = base_id + AUDIO_EXTENSION

        img_path = os.path.join(DATA_DIR, img_file)
        audio_path = os.path.join(DATA_DIR, audio_file)

        if not os.path.exists(audio_path):
            print(f'❌ {audio_file} not found. Skipping.')
            continue

        # 전처리 및 저장
        try:
            img_tensor = preprocess_image(img_path)
            mel_tensor = preprocess_audio(audio_path)

            torch.save(img_tensor, os.path.join(DATA_DIR, f'{base_id}_img.pt'))
            torch.save(mel_tensor, os.path.join(DATA_DIR, f'{base_id}_mel.pt'))

        except Exception as e:
            print(f'⚠️ Failed to process {base_id}: {e}')

if __name__ == '__main__':
    preprocess_all()
    print('\n✅ 전처리 완료! .pt 파일이 data 폴더에 저장되었습니다.')
