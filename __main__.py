from argparse import ArgumentParser

from .common.config import load_config
from .common.train import main


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Diffusion Vision Transformer with Classification"
    )
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="Train the model with the given config file.",
    )
    parser.add_argument("config", help="Path to a YAML config file.")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.train:
        main(config)
