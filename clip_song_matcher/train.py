import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from core.config import DEVICE, NPZ_PATH
from clip_song_matcher.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PROJ_DIM, PROJ_HEADS_PATH
)
from clip_song_matcher.dataset import EmbeddingPairDataset
from clip_song_matcher.model import ProjectionHead


def train():
    dataset = EmbeddingPairDataset(NPZ_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    proj_img = ProjectionHead(in_dim=512, out_dim=PROJ_DIM).to(DEVICE)
    proj_txt = ProjectionHead(in_dim=512, out_dim=PROJ_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(proj_img.parameters()) + list(proj_txt.parameters()),
        lr=LEARNING_RATE
    )
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for img_emb, txt_emb in tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)
            out_img = proj_img(img_emb)
            out_txt = proj_txt(txt_emb)

            targets = torch.ones(img_emb.size(0)).to(DEVICE)
            loss = loss_fn(out_img, out_txt, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}')

    proj_name = f'{PROJ_HEADS_PATH}_{EPOCHS}_epochs_{datetime.now().strftime('%Y%m%d')}.pth'
    
    torch.save({
        'proj_img': proj_img.state_dict(),
        'proj_txt': proj_txt.state_dict()
    }, proj_name)
