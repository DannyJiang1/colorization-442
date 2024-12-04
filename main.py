# main.py

import argparse
import torch
from colorizers.train import train_colorization


def main():
    parser = argparse.ArgumentParser(
        description="Train Image Colorization Model"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to directory containing training images",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to directory containing validation images",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size to resize images to (image_size x image_size)",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for training if available",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Checkpoint .pt file if applicable",
    )

    args = parser.parse_args()

    HW = (args.image_size, args.image_size)
    train_colorization(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        HW=HW,
        use_gpu=args.use_gpu,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()


# If no checkpoint:
# python main.py --train_dir data/landscape_images/train/ --val_dir data/landscape_images/val/ --epochs 50 --batch_size 16 --lr 0.0002 --image_size 256 --use_gpu

# If checkpoint:
# python main.py --train_dir data/landscape_images/train/ --val_dir data/landscape_images/val/ --epochs 50 --batch_size 16 --lr 0.0002 --image_size 256 --use_gpu --checkpoint checkpoints/latest_checkpoint.pt
