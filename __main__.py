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
    """Create the repository command-line parser without parsing arguments.

    The required positional config argument names a YAML file. --train/-t is a
    store_true switch: absent means configuration validation only, present means
    execute training after loading. ArgumentParser supplies standard --help output.
    This helper neither opens the config nor imports the training pipeline.

    Args:
        None.

    Returns:
        ArgumentParser: A new independently configurable parser with config and
        train destinations; train defaults to False when arguments are parsed.
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
    """Load a YAML configuration and optionally execute its training pipeline.

    Without --train/-t, successful loading and Config construction complete the
    command. With the flag, common.train.main builds, trains, evaluates, and saves
    the configured experiment. The function returns a status; script execution
    turns it into the process exit code through SystemExit.

    Args:
        argv (Sequence[str] | None): Arguments excluding the executable name.
            None reads process arguments from sys.argv. Defaults to None.

    Returns:
        int: Zero after successful loading or training. No model is returned.

    Raises:
        SystemExit: ArgumentParser displays help or rejects missing/invalid options.
        Exception: File, YAML, Config, or training failures propagate to the caller.
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
