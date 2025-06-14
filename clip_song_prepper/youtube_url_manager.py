import re
import json
from typing import Optional

from core.utils import load_json, save_json
from core.config import YOUTUBE_URLS_PATH


def extract_youtube_id(url: str) -> Optional[str]:
    playlist_match = re.match(r'.*?list=([a-zA-Z0-9_-]+)', url)
    video_match = re.match(r'.*?v=([a-zA-Z0-9_-]+)', url)

    if playlist_match:
        return playlist_match.group(1)
    elif video_match:
        return video_match.group(1)
    return None


def add_youtube_entry(url: str, title: str = None, description: str = None):
    if not 'youtube.com' in url:
        raise ValueError('The URL must be a YouTube link.')
    
    yt_id = extract_youtube_id(url)
    if yt_id is None:
        raise ValueError('The URL does not contain a valid YouTube ID.')

    data = load_json(YOUTUBE_URLS_PATH)

    if any(entry['id'] == yt_id for entry in data):
        print(f'Entry with ID {yt_id} already exists.')
        return

    data.append({
        'id': yt_id,
        'url': url,
        'title': title if title else '',
        'description': description if description else ''
    })
    save_json(data, YOUTUBE_URLS_PATH)


def remove_youtube_entry(yt_id: str):
    data = load_json(YOUTUBE_URLS_PATH)
    new_data = [entry for entry in data if entry['id'] != yt_id]
    if len(new_data) == len(data):
        print(f'No entry found with ID {yt_id}.')
        return
    save_json(new_data, YOUTUBE_URLS_PATH)


def list_youtube_entries():
    data = load_json(YOUTUBE_URLS_PATH)
    if not data:
        print('No YouTube entries found.')
        return
    for entry in data:
        print(json.dumps(entry, indent=2, ensure_ascii=False))


def find_youtube_entry(yt_id: str) -> Optional[dict]:
    data = load_json(YOUTUBE_URLS_PATH)
    for entry in data:
        if entry['id'] == yt_id:
            return entry
    return None
