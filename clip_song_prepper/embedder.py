import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

from core.utils import load_json, load_image_from_source, load_embeddings, save_embeddings
from core.config import DEVICE, CLIP_MODEL_NAME, METADATA_PATH, NPZ_PATH


def generate_embeddings():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    image_embs, text_embs, ids, urls = load_embeddings(NPZ_PATH)
    existing_data = {
        id_: {
            'image_embedding': image_emb,
            'text_embedding': text_emb,
            'url': url
        }
        for id_, image_emb, text_emb, url in zip(ids, image_embs, text_embs, urls)
    }
    existing_ids = set(existing_data.keys()) if existing_data else set()

    metadata = load_json(METADATA_PATH)
    metadata_ids = set()
    new_data = {}

    for entry in tqdm(metadata):
        entry_id = entry.get('id')
        metadata_ids.add(entry_id)

        if entry_id in existing_ids:
            new_data[entry_id] = existing_data[entry_id]
            continue

        if not entry.get('summary'):
            continue

        print(f'Processing entry {entry_id}...')
        
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

        new_data[entry_id] = {
            'image_embedding': img_emb.squeeze().cpu().numpy(),
            'text_embedding': txt_emb.squeeze().cpu().numpy(),
            'url': entry.get('webpage_url')
        }

    filtered_data = {
        id_: data for id_, data in new_data.items() if id_ in metadata_ids
    }

    save_embeddings(
        NPZ_PATH,
        [data['image_embedding'] for data in filtered_data.values()],
        [data['text_embedding'] for data in filtered_data.values()],
        list(filtered_data.keys()),
        [data['url'] for data in filtered_data.values()]
    )


if __name__ == '__main__':
    generate_embeddings()