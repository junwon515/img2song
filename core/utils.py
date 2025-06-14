import os
import json
import requests
from typing import List
from io import BytesIO
from PIL import Image
from googletrans import Translator


translator = Translator()

def translate_to_english(text: str) -> str:
    try:
        result = translator.translate(text, dest='en')
        return result.text
    except Exception as e:
        print(f'Error translating text: {e}')
        return text


def load_image_from_source(image_source: str) -> Image.Image:
    try:
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
