import os
import re
import tempfile
from typing import List, Tuple
import webvtt
import yt_dlp
import tqdm

from util import load_json, save_json


def _clean_text(text: str) -> str:
    """
    주어진 텍스트를 정제하여 불필요한 문자 제거

    Args:
        text (str): 정제할 텍스트

    Returns:
        str: 정제된 텍스트
    """
    cleaned_text = text.replace('&nbsp;', '') \
                         .replace('\u200b', ' ') \
                         .replace('“', '"').replace('”', '"') \
                         .replace('‘', "'").replace('’', "'") \
                         .replace('.', '')
    lang_pattern = re.compile(
        r'[\uac00-\ud7a3\u1100-\u11FF\u3130-\u318F'  # 한글
        r'\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'   # 일본어 (히라가나, 가타카나, CJK 통합 한자)
    )
    cleaned_text = lang_pattern.sub('', cleaned_text)
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()


def get_youtube_english_lyrics(video_id: str) -> Tuple[List[str], bool]:
    """
    YouTube 비디오의 영어 가사를 가져오고 정제 및 반환

    Args:
        video_id (str): YouTube 비디오 ID

    Returns:
        Tuple[List[str], bool]: 정제된 가사 리스트와 가사가 발견되었는지 여부
    """
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
                lyrics.append(cleaned_line)
                last_cleaned_line = cleaned_line

        found_lyrics = bool(lyrics)

    except Exception as e:
        print(f'Error processing lyrics for video {video_id}: {e}')
        lyrics = []
        found_lyrics = False
    finally:
        if downloaded_subtitle_path and os.path.exists(downloaded_subtitle_path):
            os.remove(downloaded_subtitle_path)

    return lyrics, found_lyrics


def fetch_youtube_metadata(url: str, metadata_file: str = 'metadata.json') -> None:
    """
    YouTube 비디오 또는 플레이리스트의 메타데이터를 가져오고 저장

    Args:
        url (str): YouTube URL (플레이리스트 또는 비디오)
        metadata_file (str): 메타데이터를 저장할 JSON 파일 경로
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
    }

    metadata: List[dict] = load_json(metadata_file)
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
            save_json(metadata, metadata_file)


if __name__ == '__main__':
    urls = [
        'https://www.youtube.com/playlist?list=PL4fGSI1pDJn6jXS_Tv_N9B8Z0HTRVJE0m', # Korean Top 100 Playlist
    ]

    for url in urls:
        print(f'\n--- Starting processing for {url} ---')
        fetch_youtube_metadata(url)
        print(f'--- Finished processing {url} ---\n')