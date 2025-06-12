import os
import json
import requests
from io import BytesIO
from typing import List
from PIL import Image


def load_image_from_source(image_source: str) -> Image.Image:
    """
    다양한 소스(URL, 파일 경로, 이미지 객체)에서 이미지를 로드

    Args:
        image_source (str 또는 PIL.Image.Image):
            - 이미지 URL (str)
            - 로컬 파일 경로 (str)

    Returns:
        PIL.Image.Image: 로드된 PIL 이미지 객체

    Raises:
        ValueError: 지원되지 않는 입력 타입이거나 이미지를 로드할 수 없을 때 발생
    """
    if image_source.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_source)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except requests.exceptions.RequestException as e:
            raise ValueError(f'Error loading image from URL: {e}')
        except Image.UnidentifiedImageError:
            raise ValueError(f'Error identifying image from URL: {image_source}')
    elif os.path.exists(image_source):
        try:
            image = Image.open(image_source).convert('RGB')
            return image
        except FileNotFoundError:
            raise ValueError(f'File not found: {image_source}')
        except Image.UnidentifiedImageError:
            raise ValueError(f'Error identifying image from file: {image_source}')
    else:
        raise ValueError(f'Invalid image source: {image_source}. Must be a valid URL or file path.')

def load_json(file_path: str) -> List[dict]:
    """
    JSON 파일을 로드하여 리스트로 반환

    Args:
        file_path (str): JSON 파일 경로

    Returns:
        List[dict]: JSON 데이터가 담긴 리스트
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f'Error decoding JSON from {file_path}')
            return []
        

def save_json(data: List[dict], file_path: str) -> None:
    """
    리스트 데이터를 JSON 파일로 저장

    Args:
        data (List[dict]): 저장할 데이터
        file_path (str): 저장할 JSON 파일 경로
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        try:
            json.dump(data, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f'Error saving JSON to {file_path}: {e}')
