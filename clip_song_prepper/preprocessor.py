import re
import requests
import tqdm

from core.utils import load_json, save_json
from core.config import METADATA_PATH
from clip_song_prepper.image_captioner import ImageCaptioner


def preprocess_title(title):
    allowed_pattern = r'[^a-zA-Z0-9\s.]'
    cleaned_title = re.sub(allowed_pattern, '', title)
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
    return cleaned_title.strip()


def preprocess_lyrics(lyrics):
    pass


def preprocess_story(story):
    pass


def caption_images():
    metadata = load_json(METADATA_PATH)
    captioner = ImageCaptioner()

    for item in tqdm.tqdm(metadata, desc='Preprocessing metadata'):
        if item['story'] or item['found_lyrics']:
            continue

        try:
            thumbnail_url = item['thumbnail']
            item['story'] = captioner.generate(image_source=thumbnail_url)
            save_json(metadata, METADATA_PATH)
        except requests.RequestException as e:
            print(f'Error downloading image for video {item["id"]}: {e}')
        except Exception as e:
            print(f'Error processing video {item["id"]}: {e}')

def preprocess():
    pass

if __name__ == '__main__':
    caption_images()
    print('Image captioning completed')