"""Command-line entry point for configuration-driven training.

Run ``python -m <package> --train CONFIG.yaml`` to load the YAML configuration
through `common.config.load_config` and execute `common.train.main`.Without 
``--train`` the command validates and loads the configuration but does not build 
or train a model.  ``CONFIG.yaml`` may contain partial settings; omitted sections 
and fields receive the dataclass defaults described in the ``common`` documentation. 
The YAML examples are not loaded as default layers.

Command-line inputs are strings parsed by `argparse.ArgumentParser`.
The command produces no Python return value; training writes the artifacts
enabled by the configuration and reports progress through Keras/Matplotlib.
"""

from argparse import ArgumentParser

from .common.config import load_config
from .common.train import main


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Diffusion Vision Transformer and "
                    "Variational Autoencoder for "
                    "Continual Learning"
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
