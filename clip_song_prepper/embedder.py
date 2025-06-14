import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

from core.utils import load_json, load_image_from_source
from core.config import DEVICE, CLIP_MODEL_NAME, METADATA_PATH, NPZ_PATH


def generate_embeddings():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    metadata = load_json(METADATA_PATH)

    ids, urls, image_embs, text_embs = [], [], [], []

    for entry in tqdm(metadata):
        if not entry.get('summary'):
            continue
        
        try:
            image = load_image_from_source(entry['thumbnail'])
        except Exception as e:
            print(f'Error loading image from {entry.get("thumbnail")}: {e}')
            continue

        text = ' '.join(entry.get('summary', []))
        image_inputs = processor(images=image, return_tensors='pt').to(DEVICE)
        text_inputs = processor(text=[text], return_tensors='pt', truncation=True).to(DEVICE)


        with torch.no_grad():
            img_emb = model.get_image_features(**image_inputs)
            txt_emb = model.get_text_features(**text_inputs)

        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

        image_embs.append(img_emb.squeeze().cpu().numpy())
        text_embs.append(txt_emb.squeeze().cpu().numpy())
        ids.append(entry.get('id'))
        urls.append(entry.get('webpage_url'))

    np.savez_compressed(
        NPZ_PATH,
        image_embeddings=np.stack(image_embs),
        text_embeddings=np.stack(text_embs),
        ids=np.array(ids),
        urls=np.array(urls)
    )


if __name__ == '__main__':
    generate_embeddings()