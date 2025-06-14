import argparse
from clip_song_matcher.recommender import MusicRecommender
from clip_song_matcher.train import train

from core.config import NPZ_PATH
from clip_song_matcher.config import (
    INPUT_DIM, PROJ_DIM, LEARNING_RATE, BATCH_SIZE, EPOCHS
)


def main():
    parser = argparse.ArgumentParser(
        description='Recommend songs or train the projection head using a fine-tuned CLIP model.'
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Image recommendation
    img_parser = subparsers.add_parser('image', help='Recommend by image')
    img_parser.add_argument('image_source', type=str, help='Image URL or local path')
    img_parser.add_argument('--top_k', type=int, default=5, help='Number of recommendations to return')

    # Text recommendation
    txt_parser = subparsers.add_parser('text', help='Recommend by text query')
    txt_parser.add_argument('query_text', type=str, help='Text query such as lyrics or mood')
    txt_parser.add_argument('--top_k', type=int, default=5, help='Number of recommendations to return')

    # Training
    train_parser = subparsers.add_parser('train', help='Train the projection head')
    train_parser.add_argument('--npz_path', type=str, default=NPZ_PATH)
    train_parser.add_argument('--input_dim', type=int, default=INPUT_DIM)
    train_parser.add_argument('--proj_dim', type=int, default=PROJ_DIM)
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    train_parser.add_argument('--epochs', type=int, default=EPOCHS)
    train_parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    if args.mode == 'image':
        recommender = MusicRecommender()
        recs = recommender.recommend_image(args.image_source, top_k=args.top_k)
        print(f'\nTop {args.top_k} recommendations:')
        for idx, (song_id, url, sim) in enumerate(recs, start=1):
            print(f'{idx}. ID: {song_id} | Similarity: {sim:.4f} | URL: {url}')

    elif args.mode == 'text':
        recommender = MusicRecommender()
        recs = recommender.recommend_text(args.query_text, top_k=args.top_k)
        print(f'\nTop {args.top_k} recommendations:')
        for idx, (song_id, url, sim) in enumerate(recs, start=1):
            print(f'{idx}. ID: {song_id} | Similarity: {sim:.4f} | URL: {url}')

    elif args.mode == 'train':
        train(
            npz_path=args.npz_path,
            input_dim=args.input_dim,
            proj_dim=args.proj_dim,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_path=args.save_path
        )


if __name__ == '__main__':
    main()