import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import librosa
from tqdm import tqdm
import numpy as np

from core.utils import load_image_from_source


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    이미지 파일을 텐서로 변환하는 메서드

    Args:
        image (Image.Image): PIL 이미지 객체

    Returns:
        torch.Tensor: 변환된 이미지 텐서, shape=[3, 224, 224]
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return transform(image)  # [3, 224, 224]


def preprocess_audio(audio_path, sr=22050, duration=10) -> torch.Tensor:
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


def preprocess_all(data_dir):
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
                img = load_image_from_source(img_path)
                img_tensor = preprocess_image(img).unsqueeze(0) # [1, 3, 224, 224]
                mel_tensor = preprocess_audio(audio_path).unsqueeze(0) # [1, 1, 224, 224]
                data_samples.append((img_tensor, mel_tensor))
            except Exception as e:
                print(f'Error processing {img_file} or {audio_file}: {e}')
        else:
            print(f'No matching audio file for {img_file}. Skipping.')
    return data_samples
