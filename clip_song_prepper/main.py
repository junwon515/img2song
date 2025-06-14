import argparse
from clip_song_prepper.fetch_metadata import fetch_youtube_metadata, remove_existing_metadata
from clip_song_prepper.preprocessor import caption_images, preprocess
from clip_song_prepper.embedder import generate_embeddings
from clip_song_prepper.youtube_url_manager import (
    add_youtube_entry, remove_youtube_entry, list_youtube_entries, find_youtube_entry
)
from core.utils import load_json
from core.config import YOUTUBE_URLS_PATH


def main():
    parser = argparse.ArgumentParser(description='clip_song_prepper pipeline & YouTube manager')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- add ---
    parser_add = subparsers.add_parser('add', help='Add a YouTube URL to the list')
    parser_add.add_argument('--url', type=str, required=True, help='YouTube URL to add')
    parser_add.add_argument('--title', type=str, default='', help='Optional title')
    parser_add.add_argument('--description', type=str, default='', help='Optional description')

    # --- remove ---
    parser_remove = subparsers.add_parser('remove', help='Remove a YouTube entry by ID')
    parser_remove.add_argument('--id', type=str, required=True, help='YouTube video/playlist ID to remove')

    # --- list ---
    subparsers.add_parser('list', help='List all stored YouTube entries')

    # --- find ---
    parser_find = subparsers.add_parser('find', help='Find a YouTube entry by ID')
    parser_find.add_argument('--id', type=str, required=True, help='YouTube video/playlist ID to find')

    # --- fetch ---
    parser_fetch = subparsers.add_parser('fetch', help='Fetch YouTube metadata')
    parser_fetch.add_argument('--url', type=str, help='Optional single YouTube URL')

    # --- caption ---
    subparsers.add_parser('caption', help='Generate image captions')

    # --- preprocess ---
    subparsers.add_parser('preprocess', help='Preprocess text data')

    # --- embed ---
    subparsers.add_parser('embed', help='Generate CLIP embeddings')

    # --- all ---
    parser_all = subparsers.add_parser('all', help='Run all steps: fetch → caption → preprocess → embed')
    parser_all.add_argument('--url', type=str, help='Optional single YouTube URL for fetching')

    args = parser.parse_args()

    # --- COMMAND HANDLING ---
    if args.command == 'add':
        add_youtube_entry(args.url, args.title, args.description)
        fetch_youtube_metadata(args.url)

    elif args.command == 'remove':
        entry = find_youtube_entry(args.id)
        if not entry:
            print(f'No entry found with ID: {args.id}')
            return
        if remove_existing_metadata(entry.get('url', '')):
            remove_youtube_entry(args.id)
        else:
            print(f'Failed to remove metadata for URL: {entry.get("url", "")}. Entry not removed.')

    elif args.command == 'list':
        list_youtube_entries()

    elif args.command == 'find':
        entry = find_youtube_entry(args.id)
        if not entry:
            print(f'No entry found with ID: {args.id}')
        else:
            print(f'Found entry: {entry}')

    elif args.command == 'fetch':
        print('\n=== Fetching YouTube Metadata ===')
        if args.url:
            print(f'[Fetch] Single URL: {args.url}')
            fetch_youtube_metadata(args.url)
        else:
            urls = load_json(YOUTUBE_URLS_PATH)
            if not urls:
                print('No URLs found.')
                return
            for entry in urls:
                print(f'\n--- Processing: {entry.get("title", "")} ---')
                fetch_youtube_metadata(entry['url'])
                print(f'--- Done: {entry.get("title", "")} ---')

    elif args.command == 'caption':
        print('\n=== Generating Image Captions ===')
        caption_images()

    elif args.command == 'preprocess':
        print('\n=== Preprocessing Text ===')
        preprocess()

    elif args.command == 'embed':
        print('\n=== Generating CLIP Embeddings ===')
        generate_embeddings()

    elif args.command == 'all':
        print('\n=== Full Pipeline Started ===')

        # Fetch
        if args.url:
            print(f'[Fetch] Single URL: {args.url}')
            fetch_youtube_metadata(args.url)
        else:
            urls = load_json(YOUTUBE_URLS_PATH)
            if not urls:
                print('No URLs found.')
                return
            for entry in urls:
                print(f'\n--- Processing: {entry.get("title", "")} ---')
                fetch_youtube_metadata(entry['url'])

        # Caption
        print('\n=== Generating Image Captions ===')
        caption_images()

        # Preprocess
        print('\n=== Preprocessing Text ===')
        preprocess()

        # Embedding
        print('\n=== Generating CLIP Embeddings ===')
        generate_embeddings()

        print('\nPipeline completed.')


if __name__ == '__main__':
    main()