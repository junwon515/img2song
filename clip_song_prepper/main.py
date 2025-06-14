import argparse
from clip_song_prepper.fetch_metadata import fetch_youtube_metadata
from clip_song_prepper.preprocessor import caption_images, preprocess
from clip_song_prepper.embedder import generate_embeddings
from clip_song_prepper.youtube_url_manager import (
    add_youtube_entry, remove_youtube_entry, list_youtube_entries
)
from core.utils import load_json
from core.config import YOUTUBE_URLS_PATH


def main(args):
    if args.step == 'add':
        if not args.url:
            print('You must provide a URL with --url to add.')
            return
        add_youtube_entry(args.url, args.title, args.description)
        return

    if args.step == 'remove':
        if not args.id:
            print('You must provide an ID with --id to remove.')
            return
        remove_youtube_entry(args.id)
        return

    if args.step == 'list':
        list_youtube_entries()
        return

    if args.step in ('all', 'fetch'):
        print('\n=== Fetching YouTube Metadata ===')

        if args.url:
            print(f'[Fetch] Fetching metadata for provided URL: {args.url}')
            fetch_youtube_metadata(args.url)
        else:
            urls = load_json(YOUTUBE_URLS_PATH)
            if not urls:
                print('No URLs found in YOUTUBE_URLS_PATH.')
                return
            for entry in urls:
                print(f'\n--- Processing: {entry.get("title", "")} ---')
                fetch_youtube_metadata(entry['url'])
                print(f'--- Done: {entry.get("title", "")} ---')

    if args.step in ('all', 'caption'):
        print('\n=== Generating Image Captions ===')
        caption_images()

    if args.step in ('all', 'preprocess'):
        print('\n=== Preprocessing Text (Lyrics or Captions) ===')
        preprocess()

    if args.step in ('all', 'embed'):
        print('\n=== Generating CLIP Embeddings ===')
        generate_embeddings()

    print('\nPipeline completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clip_song_prepper pipeline or manage YouTube URLs.')
    parser.add_argument(
        '--step',
        type=str,
        required=True,
        choices=['all', 'fetch', 'caption', 'preprocess', 'embed', 'add', 'remove', 'list'],
        help='Pipeline step or URL management command'
    )
    parser.add_argument('--url', type=str, help='YouTube URL to add or fetch')
    parser.add_argument('--title', type=str, default='', help='Optional title for the added URL')
    parser.add_argument('--description', type=str, default='', help='Optional description for the added URL')
    parser.add_argument('--id', type=str, help='YouTube video/playlist ID to remove')

    args = parser.parse_args()
    main(args)