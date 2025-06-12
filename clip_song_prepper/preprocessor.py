import requests
import tqdm

from core.utils import load_json, save_json, load_image_from_source
from core.config import METADATA_PATH
from clip_song_prepper.image_captioner import ImageCaptioner


def preprocess_metadata() -> None:
    """
    메타데이터 파일을 로드하고 LlaVA 모델을 사용하여 스토리 생성
    """
    metadata = load_json(METADATA_PATH)
    img2story = ImageCaptioner()

    for item in tqdm.tqdm(metadata, desc='Preprocessing metadata'):
        if item['story'] or item['found_lyrics']:
            continue

        try:
            thumbnail_url = item['thumbnail']
            image = load_image_from_source(thumbnail_url)
            item['story'] = img2story.generate(image=image)
            save_json(metadata, METADATA_PATH)
        except requests.RequestException as e:
            print(f'Error downloading image for video {item["id"]}: {e}')
        except Exception as e:
            print(f'Error processing video {item["id"]}: {e}')


if __name__ == '__main__':
    print('Preprocessing metadata...')
    preprocess_metadata()
    print('Preprocessing complete.')