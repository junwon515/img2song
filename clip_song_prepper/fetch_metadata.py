import os
import re
import tempfile
from typing import List, Tuple
import yt_dlp
import webvtt
import tqdm

from core.utils import load_json, save_json
from core.config import METADATA_PATH, YOUTUBE_URLS_PATH


def _clean_text(text: str) -> str:
    cleaned_text = text.replace('&nbsp;', '') \
                         .replace('\u200b', ' ') \
                         .replace('“', '"').replace('”', '"') \
                         .replace('‘', "'").replace('’', "'")
    allowed_pattern = r"[^a-zA-Z0-9\s'\"\-–—….,!?~\[\]\(\)]"
    cleaned_text = re.sub(allowed_pattern, '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()


def _remove_bracketed_captions(text: str) -> str:
    def replacer(match):
        content = match.group(1)
        filtered = ''.join(ch for ch in content if ch == '/')
        return filtered
    return re.sub(r'\[(.*?)\]', replacer, text)


def get_youtube_english_lyrics(video_id: str) -> Tuple[List[str], bool]:
    lyrics: List[str] = []
    found_lyrics = False
    downloaded_subtitle_path = ''

    vtt_path = os.path.join(tempfile.gettempdir(), f'{video_id}.en.vtt')
    srt_path = os.path.join(tempfile.gettempdir(), f'{video_id}.en.srt')

    try:
        if os.path.exists(vtt_path):
            downloaded_subtitle_path = vtt_path
            captions = webvtt.read(vtt_path)
        elif os.path.exists(srt_path):
            downloaded_subtitle_path = srt_path
            captions = webvtt.from_srt(srt_path)
        else:
            return [], False

        cleaned_lines = []
        last_cleaned_line = ''
        for caption in captions:
            if not caption.text.strip():
                continue
            lines = caption.text.split('\n')
            for line in lines:
                cleaned_line = _clean_text(line)
                if not cleaned_line:
                    continue
                if cleaned_line == last_cleaned_line: 
                    continue
                cleaned_lines.append(cleaned_line)
                last_cleaned_line = cleaned_line

        lyrics = _remove_bracketed_captions('/'.join(cleaned_lines)).split('/')
        lyrics = [lyric.strip() for lyric in lyrics if lyric.strip()]
        found_lyrics = bool(lyrics)

        if found_lyrics and len(' '.join(lyrics)) < 250:
            lyrics = []
            found_lyrics = False

    except Exception as e:
        print(f'Error processing lyrics for video {video_id}: {e}')
        lyrics = []
        found_lyrics = False
    finally:
        if downloaded_subtitle_path and os.path.exists(downloaded_subtitle_path):
            os.remove(downloaded_subtitle_path)

    return lyrics, found_lyrics


def fetch_metatadata(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)

            if info is None:
                return None

            video_id = info.get('id')
            lyrics, found_lyrics = get_youtube_english_lyrics(video_id)

            return {
                'id': video_id,
                'title': info.get('title'),
                'webpage_url': info.get('webpage_url'),
                'thumbnail': info.get('thumbnail'),
                'lyrics': lyrics,
                'found_lyrics': found_lyrics,
                'story': '',
                'summary': [],
            }
    except Exception as e:
        print(f'Error fetching metadata for {url}: {e}')
    return None


def update_metadata(url: str = None) -> None:
    ydl_flat_opts = {
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'extract_flat': True
    }

    if url is None:
        urls = [data['url'] for data in load_json(YOUTUBE_URLS_PATH)]
    else:
        urls = [url]
    total_entries: List[dict] = []
    for u in tqdm.tqdm(urls, desc='Fetching metadata from URLs'):
        with yt_dlp.YoutubeDL(ydl_flat_opts) as ydl_flat:
            try:
                info_dict = ydl_flat.extract_info(u, download=False)
                entries = info_dict.get('entries', []) if 'entries' in info_dict else [info_dict]
                total_entries.extend(entries)
            except yt_dlp.utils.DownloadError as e:
                print(f'Error fetching info for {u}: {e}')
      

    metadata = load_json(METADATA_PATH)
    existing_data = {item['id']: item for item in metadata if 'id' in item}
    existing_ids = set(existing_data.keys())

    new_metadata: List[dict] = [] if url is None else metadata.copy()
    for entry in tqdm.tqdm(total_entries, desc='Updating metadata'):
        if entry is None:
            continue

        video_id = entry.get('id')
        if video_id in existing_ids:
            if url is None:
                new_metadata.append(existing_data[video_id])
        else:
            new_item = fetch_metatadata(entry.get('url') or entry.get('webpage_url'))
            if new_item is not None:
                new_metadata.append(new_item)

    save_json(new_metadata, METADATA_PATH)


if __name__ == '__main__':
    update_metadata()