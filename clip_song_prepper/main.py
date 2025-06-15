import argparse

from clip_song_prepper.fetch_metadata import update_metadata
from clip_song_prepper.preprocessor import caption_images, preprocess
from clip_song_prepper.embedder import update_embeddings
from clip_song_prepper.youtube_url_manager import (
    add_youtube_entry, remove_youtube_entry, list_youtube_entries
)


def main():
    parser = argparse.ArgumentParser(description='YouTube Data Management CLI')
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

    # --- update ---
    parser_update = subparsers.add_parser('update', help='Update YouTube entries and process data')
    parser_update.add_argument('--fetch', action='store_true', help='Fetch YouTube metadata')
    parser_update.add_argument('--caption', action='store_true', help='Generate image captions')
    parser_update.add_argument('--preprocess', action='store_true', help='Preprocess text data')
    parser_update.add_argument('--embed', action='store_true', help='Generate CLIP embeddings')
    parser_update.add_argument('--all', action='store_true', help='Run all update steps')
    parser_update.add_argument('--url', type=str, help='Optional YouTube URL to process')

    args = parser.parse_args()

    # --- COMMAND HANDLING ---
    if args.command == 'add':
        add_youtube_entry(args.url, args.title, args.description)

    elif args.command == 'remove':
        remove_youtube_entry(args.id)

    elif args.command == 'list':
        list_youtube_entries()

    elif args.command == 'update':
        any_action_taken = False
        if args.fetch or args.all:
            print('\n=== Fetching YouTube Metadata ===')
            if args.url and not args.all:
                update_metadata(args.url)
            else:
                update_metadata()
            any_action_taken = True
        if args.caption or args.all:
            print('\n=== Generating Image Captions ===')
            caption_images()
            any_action_taken = True
        if args.preprocess or args.all:
            print('\n=== Preprocessing Text ===')
            preprocess()
            any_action_taken = True
        if args.embed or args.all:
            print('\n=== Generating CLIP Embeddings ===')
            update_embeddings()
            any_action_taken = True
        if not any_action_taken:
            update_metadata()
            preprocess()
            update_embeddings()


if __name__ == '__main__':
    main()