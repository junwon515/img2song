import re
import requests
import tqdm
from typing import List
from collections import Counter

from core.utils import load_json, save_json
from core.config import METADATA_PATH
from clip_song_prepper.image_captioner import ImageCaptioner
from clip_song_prepper.token_checker import TokenChecker


def preprocess_title(title: str) -> str:
    allowed_pattern = r'[^a-zA-Z0-9\s.]'
    cleaned_title = re.sub(allowed_pattern, '', title)
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
    unique_list = list(dict.fromkeys(cleaned_title.strip().split()))
    return ' '.join(unique_list)


def segment_text(text):
    end_markers = '….,!?~'
    pattern = r'(\([^)]*\))|([^….!?~()]+\s*[….!?~]+?)|([^\n]+)'

    current_text_segments = []
    for match in re.finditer(pattern, text):
        for part in match.groups():
            if part:
                part = re.sub(r'\s+', ' ', part.strip())
                current_text_segments.append(part)
    
    for segment in current_text_segments:
        if segment.startswith('(') and segment.endswith(')'):
            yield segment
        elif segment[-1] in end_markers:
            yield segment
        else:
            yield segment + '.'


def preprocess_lyrics(checker: TokenChecker, title: str, lyrics: List[str]) -> List[str]:
    parts = []
    for line in lyrics:
        parts.extend(segment_text(line))

    if checker.check(f'{title}. {" ".join(parts)}'):
        return list(dict.fromkeys(parts))

    parts_count = Counter(parts)
    unique_parts = list(dict.fromkeys(parts))
    while not checker.check(f'{title}. {" ".join(unique_parts)}'):
        temp_parts = []
        common_parts = [item for item, _ in parts_count.most_common(len(unique_parts)//2)]
        for part in unique_parts:
            if part in common_parts:
                temp_parts.append(part)

        unique_parts = temp_parts

    return unique_parts


def preprocess_story(checker: TokenChecker, title: str, story: str) -> List[str]:
    allowed_pattern = r"[^a-zA-Z0-9\s\n'\"\-–—….,!?~\(\)]"
    cleaned_story = re.sub(allowed_pattern, '', story)

    story_lines = cleaned_story.split('\n')
    parts = []
    for line in story_lines:
        line = re.sub(r'\s+', ' ', line.strip())
        if line:
            parts.extend(segment_text(line))

    while not checker.check(f'{title}. {" ".join(parts)}'):
        del parts[len(parts)//2]

    return parts

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
    metadata = load_json(METADATA_PATH)
    checker = TokenChecker()

    for item in tqdm.tqdm(metadata, desc='Preprocessing metadata'):
        if item['summary'] or not (item['found_lyrics'] or item['story']):
            continue

        temp_list: List[str] = []
        cleaned_title = preprocess_title(item['title'])
        temp_list.append(cleaned_title)
        if item['found_lyrics']:
            temp_list.extend(preprocess_lyrics(checker, cleaned_title, item['lyrics']))
        elif item['story']:
            temp_list.extend(preprocess_story(checker, cleaned_title, item['story']))

        item['summary'] = temp_list
        save_json(metadata, METADATA_PATH)

if __name__ == '__main__':
    preprocess()