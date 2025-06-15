import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from core.config import DEVICE, NPZ_PATH, CHECKPOINTS_DIR
from clip_song_matcher.config import (
    INPUT_DIM, PROJ_DIM, BATCH_SIZE, TEMPERATURE, EPOCHS, LEARNING_RATE,
    PROJ_HEADS_NAME
)
from clip_song_matcher.dataset import EmbeddingPairDataset
from clip_song_matcher.model import ProjectionHead


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor):
        logits = F.cosine_similarity(img_features.unsqueeze(1), txt_features.unsqueeze(0), dim=2)
        logits /= self.temperature

        labels_img = torch.arange(img_features.size(0)).to(img_features.device)
        loss_img = F.cross_entropy(logits, labels_img)

        labels_txt = torch.arange(txt_features.size(0)).to(txt_features.device)
        loss_txt = F.cross_entropy(logits.T, labels_txt)

        total_loss = (loss_img + loss_txt) / 2
        return total_loss


def train(npz_path: str = NPZ_PATH,
          input_dim: int = INPUT_DIM,
          proj_dim: int = PROJ_DIM,
          lr: float = LEARNING_RATE,
          batch_size: int = BATCH_SIZE,
          temperature: float = TEMPERATURE,
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
    loss_fn = InfoNCELoss(temperature=temperature)

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for img_emb, txt_emb in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            img_emb, txt_emb = img_emb.to(DEVICE), txt_emb.to(DEVICE)
            out_img = F.normalize(proj_img(img_emb), dim=-1)
            out_txt = F.normalize(proj_txt(txt_emb), dim=-1)

            loss = loss_fn(out_img, out_txt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    if save_path is None:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        proj_filename = f'{PROJ_HEADS_NAME}_{epochs}_epochs_{datetime.now().strftime('%Y%m%d')}.pth'
        proj_path = os.path.join(CHECKPOINTS_DIR, proj_filename)
    else:
        proj_path = save_path

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(proj_path.replace('.pth', '_loss_plot.png'))
    plt.show()
    
    torch.save({
        'proj_img': proj_img.state_dict(),
        'proj_txt': proj_txt.state_dict()
    }, proj_path)

    print(f'Projection heads saved to {proj_path}')


if __name__ == '__main__':
    train()