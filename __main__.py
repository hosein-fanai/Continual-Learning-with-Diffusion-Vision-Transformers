"""Repository-root command-line entry point for configuration-driven training.

Run ``python __main__.py CONFIG.yaml`` from the repository root to validate and
load a YAML configuration. Add ``--train`` to execute the complete training
pipeline. The repository is not installed as a package, so this module uses
root-level absolute imports rather than package-relative imports.
"""

from argparse import ArgumentParser

from collections.abc import Sequence

from common.config import load_config


def build_parser() -> ArgumentParser:
    """Create the repository command-line parser.

    Args:
        None.

    Returns:
        ArgumentParser: Parser for a YAML config path and optional training flag.
    """

    parser = ArgumentParser(
        description=(
            "Diffusion Vision Transformer and "
            "Variational Autoencoder for Continual Learning"
        )
    )
    parser.add_argument(
        "--train", 
        "-t", 
        action="store_true", 
        help="Train the model after loading the configuration."
    )
    parser.add_argument("config", help="Path to a YAML config file.")

    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Load a configuration and optionally run training.

    Args:
        argv (Sequence[str] | None): Optional arguments excluding the executable
            name. ``None`` reads process arguments from ``sys.argv``.

    Returns:
        int: Zero after successful configuration loading or training.
    """

    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    # Run the configured training pipeline only when explicitly requested.
    if args.train:
        from common.train import main

        main(config)

    return 0


# Execute the command-line interface when this file is run as a script.
if __name__ == "__main__":
    raise SystemExit(cli())
