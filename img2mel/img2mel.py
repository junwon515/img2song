import os
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

from core.utils import load_image_from_source
from core.config import DEVICE
from img2mel.config import MODEL_PATH
from img2mel.model import Img2MelNet
from img2mel.preprocessor import preprocess_image


class Img2Mel:
    def __init__(self):
        self.model = Img2MelNet().to(DEVICE)
        self._load_model(MODEL_PATH)

    def _load_model(self, path):
        """
        모델 파일을 로드하고 상태를 설정
        모델 파일은 'latest'로 끝나는 파일을 우선적으로 찾고, 없으면 가장 최근의 파일을 선택

        Args:
            path (str): 모델 파일 경로

        Raises:
            FileNotFoundError: 지정된 경로에 모델 파일이 없을 경우
        """
        dir, filename = os.path.split(path)
        files = { 
            f.split('_')[-1]: os.path.join(dir, f) for f in os.listdir(dir)
            if f.startswith(filename) and f.endswith('.pth')
        }
        if 'latest' in files:
            model_path = files['latest']
        else:
            sorted_files = sorted(files.keys())
            model_path = files[sorted_files[-1]] if sorted_files else None

        if model_path is None:
            raise FileNotFoundError(f'Model file not found in {dir}. Please check the directory.')
        
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def generate(self, image_source, output_wav='output.wav', sr=22050, duration=10):
        """
        이미지로부터 멜 스펙트로그램을 생성하고 오디오 파일로 변환

        Args:
            image_path (str): 입력 이미지 파일 경로
            output_wav (str): 출력 오디오 파일 경로, 기본값은 'output.wav'
            sr (int): 샘플링 레이트, 기본값은 22050
            duration (int): 오디오 길이(초), 기본값은 10초
        """
        try:
            image = load_image_from_source(image_source)
        except Exception as e:
            raise ValueError(e)
        
        img_tensor = preprocess_image(image).unsqueeze(0).to(self.device)
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
