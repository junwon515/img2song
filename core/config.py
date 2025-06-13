import os

DEVICE = 'cuda'
DATA_DIR = 'data'
YOUTUBE_URLS_PATH = os.path.join(DATA_DIR, 'youtube_urls.json')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
NPZ_PATH = os.path.join(DATA_DIR, 'embeddings.npz')

CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CAPTIONER_MODEL_NAME = 'unsloth/llava-1.5-7b-hf-bnb-4bit'
