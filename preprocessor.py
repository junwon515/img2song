import requests
import tqdm
from PIL import Image
from io import BytesIO

from util import load_json, save_json, get_device
from model.img2story import Img2Story


def preprocess_metadata(metadata_file: str = 'metadata.json') -> None:
    """
    메타데이터 파일을 로드하고 Img2Story 모델을 사용하여 스토리 생성

    Args:
        metadata_file (str): 메타데이터 JSON 파일 경로
    """
    metadata = load_json(metadata_file)
    img2story = Img2Story(device=get_device())

    for item in tqdm.tqdm(metadata, desc='Preprocessing metadata'):
        if item['story'] or item['found_lyrics']:
            continue

        try:
            thumbnail_url = item['thumbnail']
            response = requests.get(thumbnail_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            item['story'] = img2story.generate(image=image)
            save_json(metadata, metadata_file)
        except requests.RequestException as e:
            print(f'Error downloading image for video {item["id"]}: {e}')
        except Exception as e:
            print(f'Error processing video {item["id"]}: {e}')


if __name__ == '__main__':
    print('Preprocessing metadata...')
    preprocess_metadata()
    print('Preprocessing complete.')