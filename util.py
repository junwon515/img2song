import os
import json
import torch
from typing import List


def _load_json(file_path: str) -> List[dict]:
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
        

def _save_json(data: List[dict], file_path: str) -> None:
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


def get_device(vram_threshold_gb : int = 4) -> str:
    """
    사용 가능한 디바이스(cuda 또는 cpu)를 확인하고 반환합니다.
    VRAM 임계값을 기준으로 GPU 사용 여부를 결정합니다.

    Args:
        vram_threshold_gb (int): VRAM 임계값 (GB). 기본값은 4GB입니다.

    Returns:
        str: 'cuda' 또는 'cpu'. 사용 가능한 디바이스를 반환합니다.
    """
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram >= vram_threshold_gb:
            return 'cuda'
        else:
            print(f'Insufficient VRAM ({vram:.2f} GB). Using CPU instead.')
    return 'cpu'


if __name__ == '__main__':
    device = get_device()
    print(f'Using device: {device}')