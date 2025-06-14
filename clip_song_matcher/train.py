import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from core.config import DEVICE, NPZ_PATH, CHECKPOINTS_DIR
from clip_song_matcher.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, INPUT_DIM, PROJ_DIM, PROJ_HEADS_NAME
)
from clip_song_matcher.dataset import EmbeddingPairDataset
from clip_song_matcher.model import ProjectionHead


def train(npz_path: str = NPZ_PATH,
          input_dim: int = INPUT_DIM,
          proj_dim: int = PROJ_DIM,
          lr: float = LEARNING_RATE,
          batch_size: int = BATCH_SIZE,
          epochs: int = EPOCHS,
          save_path: str = None):
    dataset = EmbeddingPairDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    proj_img = ProjectionHead(in_dim=input_dim, out_dim=proj_dim).to(DEVICE)
    proj_txt = ProjectionHead(in_dim=input_dim, out_dim=proj_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(proj_img.parameters()) + list(proj_txt.parameters()),
        lr=lr
    )
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        total_loss = 0
        for img_emb, txt_emb in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            img_emb, txt_emb = img_emb.to(DEVICE), txt_emb.to(DEVICE)
            out_img = proj_img(img_emb)
            out_txt = proj_txt(txt_emb)

            targets = torch.ones(img_emb.size(0)).to(DEVICE)
            loss = loss_fn(out_img, out_txt, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}')

    if save_path is None:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        proj_filename = f'{PROJ_HEADS_NAME}_{epochs}_epochs_{datetime.now().strftime('%Y%m%d')}.pth'
        proj_path = os.path.join(CHECKPOINTS_DIR, proj_filename)
    else:
        proj_path = save_path
    
    torch.save({
        'proj_img': proj_img.state_dict(),
        'proj_txt': proj_txt.state_dict()
    }, proj_path)

    print(f'Projection heads saved to {proj_path}')


if __name__ == '__main__':
    train()