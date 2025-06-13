import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from core.utils import load_image_from_source
from core.config import DEVICE, CLIP_MODEL_NAME, NPZ_PATH
from clip_song_matcher.config import PROJ_HEADS_PATH
from clip_song_matcher.model import ProjectionHead


class MusicRecommender:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.proj_img = ProjectionHead().to(DEVICE)
        self.proj_txt = ProjectionHead().to(DEVICE)
        self._load_projection(PROJ_HEADS_PATH)

        db = np.load(NPZ_PATH)
        self.text_embs = torch.tensor(db['text_embeddings']).to(DEVICE)
        self.urls = db['urls']
        self.ids = db['ids']

        self.text_embs_proj = self.proj_txt(self.text_embs)

    def _load_projection(self, path):
        dir, filename = os.path.split(path)
        files = {
            f.split('_')[-1]: os.path.join(dir, f) for f in os.listdir(dir)
            if f.startswith(filename) and f.endswith('.pth')
        }
        if 'latest' in files:
            proj_path = files['latest']
        else:
            sorted_files = sorted(files.keys())
            proj_path = files[sorted_files[-1]] if sorted_files else None

        if proj_path is None:
            raise FileNotFoundError(f'Projection head file not found in {dir}. Please check the directory.')

        state = torch.load(proj_path, map_location=DEVICE)
        self.proj_img.load_state_dict(state['proj_img'])
        self.proj_txt.load_state_dict(state['proj_txt'])

    def recommend(self, image_source, top_k=5):
        try:
            image = load_image_from_source(image_source)
        except Exception as e:
            raise ValueError(e)
        inputs = self.processor(images=image, return_tensors='pt').to(DEVICE)
        img_emb = self.model.get_image_features(**inputs)
        img_emb_proj = self.proj_img(img_emb)

        sims = F.cosine_similarity(img_emb_proj, self.text_embs_proj)
        top_indices = torch.topk(sims, top_k).indices

        return [(self.ids[i], self.urls[i], sims[i].item()) for i in top_indices]


if __name__ == '__main__':
    recommender = MusicRecommender()
    while True:
        user_input = input('>> ').split()
        if user_input[0].lower() == 'exit':
            break
        
        try:
            top_k = int(user_input[1]) if len(user_input) > 1 else 5
            if top_k <= 0:
                raise ValueError('Top K must be a positive integer.')
            recommendations = recommender.recommend(user_input[0], top_k=top_k)
            
            print(f'Top {top_k} recommendations:')
            for rec in recommendations:
                print(f'ID: {rec[0]}, URL: {rec[1]}, Similarity: {rec[2]:.4f}')

        except Exception as e:
            print(f'Error: {e}')
            continue