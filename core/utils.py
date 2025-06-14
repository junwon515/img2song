import os
import json
import requests
from typing import List, Tuple
from io import BytesIO
from PIL import Image
from googletrans import Translator
import numpy as np


translator = Translator()

def translate_to_english(text: str) -> str:
    try:
        result = translator.translate(text, dest='en')
        return result.text
    except Exception as e:
        print(f'Error translating text: {e}')
        return text


def load_image_from_source(image_source: str | Image.Image | BytesIO) -> Image.Image:
    try:
        if isinstance(image_source, Image.Image):
            return image_source.convert('RGB')
        
        if isinstance(image_source, BytesIO):
            return Image.open(image_source).convert('RGB')

        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')

        if os.path.exists(image_source):
            return Image.open(image_source).convert('RGB')

        raise ValueError(f'Invalid image source: {image_source}. Must be a valid URL or file path.')

    except requests.exceptions.RequestException as e:
        raise ValueError(f'Error loading image from URL: {e}')
    except (FileNotFoundError, Image.UnidentifiedImageError) as e:
        raise ValueError(f'Error loading image from file: {e}')


def load_embeddings(npz_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    try:
        npz = np.load(npz_path, allow_pickle=True)
        image_embs = list(npz['image_embeddings'])
        text_embs = list(npz['text_embeddings'])
        ids = list(npz['ids'])
        urls = list(npz['urls'])
        return image_embs, text_embs, ids, urls
    except FileNotFoundError:
        return [], [], [], []


def save_embeddings(npz_path: str, 
                    image_embeddings: List[np.ndarray], 
                    text_embeddings: List[np.ndarray], 
                    ids: List[str], 
                    urls: List[str]) -> None:
    try:
        np.savez_compressed(
            npz_path,
            image_embeddings=np.stack(image_embeddings),
            text_embeddings=np.stack(text_embeddings),
            ids=np.array(ids),
            urls=np.array(urls)
        )
    except IOError as e:
        print(f'Error saving embeddings to {npz_path}: {e}')


def load_json(file_path: str) -> List[dict]:
    if not os.path.exists(file_path):
        save_json([], file_path)
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f'Error decoding JSON from {file_path}. Returning empty list.')
        return []


def save_json(data: List[dict], file_path: str) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f'Error saving JSON to {file_path}: {e}')
