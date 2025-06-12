import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingPairDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.image_embs = data['image_embeddings']
        self.text_embs = data['text_embeddings']

    def __len__(self):
        return len(self.image_embs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.image_embs[idx], dtype=torch.float32),
            torch.tensor(self.text_embs[idx], dtype=torch.float32)
        )
