import os
import re
import tempfile
from typing import List, Tuple
import webvtt
import yt_dlp
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

    except Exception as e:
        print(f'Error processing lyrics for video {video_id}: {e}')
        lyrics = []
        found_lyrics = False
    finally:
        if downloaded_subtitle_path and os.path.exists(downloaded_subtitle_path):
            os.remove(downloaded_subtitle_path)

    return lyrics, found_lyrics


def fetch_youtube_metadata(url: str) -> None:
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

    metadata: List[dict] = load_json(METADATA_PATH)
    existing_video_ids = {item['id'] for item in metadata if 'id' in item}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            all_info = ydl.extract_info(url, download=False)
            entries = all_info.get('entries', []) if 'entries' in all_info else [all_info]
        except yt_dlp.utils.DownloadError as e:
            print(f'Error fetching info for {url}: {e}')
            return

        playlist_title = all_info.get('title', 'Unknown Playlist')
        print(f'{playlist_title} ({len(entries)} videos)')

        for entry in tqdm.tqdm(entries, desc=f'Processing videos from "{playlist_title}"'):
            if entry is None:
                continue

            video_id = entry.get('id')
            if not video_id or video_id in existing_video_ids:
                continue

            title = entry.get('title')
            webpage_url = entry.get('webpage_url')
            thumbnail = entry.get('thumbnail')

            try:
                ydl.download([webpage_url])
            except yt_dlp.utils.DownloadError as e:
                print(f'Error downloading video {video_id}: {e}')
                continue

            lyrics, found_lyrics = get_youtube_english_lyrics(video_id)

            metadata.append({
                'id': video_id,
                'title': title,
                'webpage_url': webpage_url,
                'thumbnail': thumbnail,
                'lyrics': lyrics,
                'found_lyrics': found_lyrics,
                'story': '',
                'summary': [],
            })
            save_json(metadata, METADATA_PATH)


if __name__ == '__main__':
    urls = load_json(YOUTUBE_URLS_PATH)
    for url in urls:
        print(f'\n--- Starting processing for {url["title"]} ---')
        fetch_youtube_metadata(url['url'])
        print(f'--- Finished processing {url["title"]} ---\n')