import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO

from core.utils import load_image_from_source, translate_to_english, load_embeddings
from core.config import DEVICE, CLIP_MODEL_NAME, CHECKPOINTS_DIR, NPZ_PATH
from clip_song_matcher.config import PROJ_HEADS_NAME
from clip_song_matcher.model import ProjectionHead


class MusicRecommender:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.proj_img = ProjectionHead().to(DEVICE)
        self.proj_txt = ProjectionHead().to(DEVICE)
        self._load_projection(CHECKPOINTS_DIR, PROJ_HEADS_NAME)

        image_embs, text_embs, ids, urls = load_embeddings(NPZ_PATH)
        self.image_embs = torch.tensor(image_embs, dtype=torch.float32).to(DEVICE)
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32).to(DEVICE)
        self.urls = urls
        self.ids = ids

        self.image_embs_proj = self.proj_img(self.image_embs)
        self.text_embs_proj = self.proj_txt(self.text_embs)

    def _load_projection(self, dir: str, base_name: str) -> None:
        matching_files = [
            os.path.join(dir, f) for f in os.listdir(dir)
            if f.startswith(base_name) and f.endswith('.pth')
        ]

        if not matching_files:
            raise FileNotFoundError(f"No projection heads found with prefix '{base_name}' in {dir}")

        latest_file = max(matching_files, key=os.path.getctime)
        state = torch.load(latest_file, map_location=DEVICE)

        self.proj_img.load_state_dict(state['proj_img'])
        self.proj_txt.load_state_dict(state['proj_txt'])

    def recommend_image(self,
                        image_source: str | Image.Image | BytesIO,
                        top_k: int = 5
                        ) -> list[tuple[str, str, float]]:
        image = load_image_from_source(image_source)
        inputs = self.processor(images=image, return_tensors='pt').to(DEVICE)
        img_emb = self.model.get_image_features(**inputs)
        img_emb_proj = self.proj_img(img_emb)

        sims = F.cosine_similarity(img_emb_proj, self.text_embs_proj)
        top_indices = torch.topk(sims, top_k).indices

        return [(self.ids[i], self.urls[i], sims[i].item()) for i in top_indices]
    
    def recommend_text(self, query_text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        if not query_text.strip():
            raise ValueError('Query text cannot be empty.')
        
        translated_query = translate_to_english(query_text)
        inputs = self.processor(text=[translated_query], return_tensors='pt', truncation=True).to(DEVICE)
        txt_emb = self.model.get_text_features(**inputs)
        txt_emb_proj = self.proj_txt(txt_emb)

        sims = F.cosine_similarity(txt_emb_proj, self.image_embs_proj)
        top_indices = torch.topk(sims, top_k).indices

        return [(self.ids[i], self.urls[i], sims[i].item()) for i in top_indices]
