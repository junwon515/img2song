import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from img2mel.model import Img2MelNet
from img2mel.preprocessor import preprocess_all
from img2mel.config import MODEL_PATH, LEARNING_RATE, EPOCHS
from core.config import DEVICE, DATA_DIR

def train():
    """
    모델 학습 메서드
    
    Args:
        data_dir (str): 학습 데이터 디렉토리 경로, 기본값은 'data'
    """
    model = Img2MelNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    data_samples = preprocess_all(DATA_DIR)
    if not data_samples:
        print(f'No valid data samples found in {DATA_DIR}. Please check your data.')
        return
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for img_tensor, mel_tensor in tqdm(data_samples, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            img_tensor = img_tensor.to(DEVICE)
            mel_tensor = mel_tensor.to(DEVICE)

            optimizer.zero_grad()
            mel_pred = model(img_tensor)
            loss = criterion(mel_pred, mel_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_samples)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')

    model_name = f'{MODEL_PATH}_{EPOCHS}epochs_{datetime.now().strftime('%Y%m%d')}.pth'
    torch.save(model.state_dict(model_name), )
    print(f'Model trained and saved to {model_name}.')
