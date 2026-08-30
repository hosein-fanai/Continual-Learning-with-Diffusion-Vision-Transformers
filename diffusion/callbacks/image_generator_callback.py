"""Epoch-end diffusion sampling, image plotting, and denoising GIF output."""

from tensorflow.keras import callbacks

import os

from datetime import datetime

from typing import Any

from common.utils import plot_images, create_gif


class ImageGeneratorCallback(callbacks.Callback):
    """Generate qualitative diffusion samples after every training epoch.

    The callback expects a ``DiffusionModel``-compatible bound model exposing
    ``test_steps``, ``test_cfg_scale``, ``test_eta``, ``test_network_name``, and
    ``sample``. Valid constructor combinations in the current implementation are:

    * display only: ``show_images=True``, ``save_gifs=False``, and
      ``results_path=None``;
    * save PNGs, with optional GIFs: a non-``None`` ``results_path``;
      ``show_images`` controls simultaneous display.

    A dated run directory is created immediately during construction when
    saving. GIF output additionally requires a result path.

    Args:
        add_null_label: Whether a CFG model's null condition is included in
            the generated grid.
        show_images: Whether ``plot_images`` displays the generated image grid.
        save_gifs: Whether to request intermediate ``x_t`` and ``x_0`` frames
            and write a denoising GIF per epoch.
        results_path: Optional string or path-like base directory. A timestamped
            child containing ``images`` is created; ``gifs`` is added when GIF
            saving is enabled.
        project_tag: Optional text appended to the timestamped directory name.
        seed: Optional sampling seed reused at each epoch.
        **kwargs: Arguments forwarded to ``tf.keras.callbacks.Callback``. The
            TensorFlow 2.10 base callback normally requires no extra options.

    Inputs:
        Keras supplies a zero-based integer epoch and optional metric mapping;
        the bound diffusion wrapper supplies sampling configuration and images.

    Outputs:
        Callback hooks return ``None``. Observable outputs are displayed image
        grids and, when configured, PNG and GIF files.
    """

    def __init__(
        self, 
        add_null_label: bool = True, 
        show_images: bool = True, 
        save_gifs: bool = False, 
        results_path: str | os.PathLike[str] | None = None,
        project_tag: str | None = None, 
        seed: int | None = None, 
        **kwargs: Any
    ) -> None:
        """Validate output mode and create the timestamped result directories.

        Args:
            add_null_label (bool): Include condition ID 0 for CFG models when
                generating the epoch grid.
            show_images (bool): Whether to display each generated image grid.
            save_gifs (bool): Whether to save intermediate denoising frames as
                a GIF for each epoch.
            results_path (str | os.PathLike[str] | None): Optional output base
                directory. A timestamped run directory is created beneath it.
            project_tag (str | None): Optional suffix for the run-directory
                name.
            seed (int | None): Optional seed forwarded to model sampling.
            **kwargs (Any): Options forwarded to the Keras callback base class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)

        # Reject configurations that neither display nor save generated images.
        if not show_images and results_path is None:
            raise ValueError("The callback must show or save images.")
        # Require an output directory whenever GIF saving is enabled.
        if save_gifs and results_path is None:
            raise ValueError("save_gifs requires results_path.")


        self.add_null_label = add_null_label
        self.show_images = show_images
        self.save_gifs = save_gifs
        self.results_path = results_path
        self.seed = seed
        self.base_seed = seed
        self.artifact_prefix = ""

        project_tag = "" if project_tag is None else " " + project_tag

        # Create a timestamped artifact directory when saving is requested.
        if self.results_path is not None:
            self.results_path = os.path.join(
                self.results_path, 
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + project_tag
            )
            os.makedirs(self.results_path, exist_ok=True)

            os.makedirs(
                os.path.join(self.results_path, "images"), 
                exist_ok=True
            )
            # Create the GIF subdirectory only for GIF-enabled runs.
            if save_gifs:
                os.makedirs(
                    os.path.join(self.results_path, "gifs"), 
                    exist_ok=True
                )

    def set_artifact_prefix(
        self,
        prefix: str | None,
    ) -> None:
        """Set a safe filename prefix for a later training phase or task.

        Args:
            prefix (str | None): Prefix without a trailing separator. ``None``
                or an empty string restores the ordinary epoch-only names.

        Returns:
            None: Subsequent PNG/GIF filenames are updated in place.

        Raises:
            TypeError: If ``prefix`` is neither a string nor ``None``.
        """

        # Reject filesystem prefixes with an unsupported runtime type.
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError("prefix must be a string or None.")
        normalized = "" if prefix is None else prefix.strip()
        self.artifact_prefix = normalized + "_" if normalized else ""

    def get_config(self) -> dict[str, object]:
        """Return behavior-defining callback options for recovery fingerprints.

        Returns:
            dict[str, object]: Sampling/display/GIF options and the initial
            seed. Filesystem paths and mutable task prefixes are excluded.
        """

        return {
            "add_null_label": self.add_null_label,
            "show_images": self.show_images,
            "save_gifs": self.save_gifs,
            "seed": self.base_seed,
        }

    def on_epoch_end(
        self, 
        epoch: int, 
        logs: dict[str, Any] | None = None
    ) -> None:
        """Sample the bound diffusion model and render epoch artifacts.

        Args:
            epoch (int): Zero-based epoch index. Output filenames use
                ``epoch + 1``.
            logs (dict[str, Any] | None): Optional Keras epoch-log mapping. It
                is accepted for callback compatibility and is not read.

        Returns:
            ``None``. ``model.sample`` returns images shaped
            ``[batch, height, width, channels]``. In GIF mode it must return
            ``(images, x_t_frames, x0_frames)``; the frame sequences are passed
            to ``create_gif``.
        """

        network_name = self.model.test_network_name
        network = self.model.get_network(network_name)

        sample_kwargs = {
            "network_name": network_name, 
            "labels": list(range(
                0 if self.add_null_label and network.use_cfg \
                else int(network.use_cfg), 
                network.num_labels
            )), 
            "steps": self.model.test_steps, 
            "scale": self.model.test_cfg_scale, 
            "eta": self.model.test_eta, 
            "return_x_ts": self.save_gifs, 
            "return_x0s": self.save_gifs, 
            "seed": self.seed
        }
        outputs = self.model.sample(**sample_kwargs)

        # Request intermediate denoising frames when a GIF will be written.
        if self.save_gifs:
            imgs, frames1, frames2 = outputs
            create_gif(
                os.path.join(
                    self.results_path, 
                    "gifs", 
                    f"{self.artifact_prefix}epoch-{epoch+1}_"
                        f"steps-{sample_kwargs['steps']}_"
                        f"scale-{sample_kwargs['scale']:.1f}_"
                        f"eta-{sample_kwargs['eta']:.4f}.gif"
                ), 
                frames1, 
                frames2, 
                verbose=0
            )
        # Sample only final images when no GIF frames are needed.
        else:
            imgs = outputs

        # Save the image grid, optionally displaying it at the same time.
        if self.results_path is not None: 
            plot_images(
                imgs, 
                show_images=self.show_images, 
                save_path=os.path.join(
                    self.results_path, 
                    "images", 
                    f"{self.artifact_prefix}epoch-{epoch+1}_"
                        f"steps-{sample_kwargs['steps']}_"
                        f"scale-{sample_kwargs['scale']:.1f}_"
                        f"eta-{sample_kwargs['eta']:.4f}.png"
                ) 
            )
        # Display the grid directly when no artifact directory is configured.
        else:
            plot_images(imgs)


def run_self_tests() -> dict[str, str]:
    """Test display and filesystem modes of :class:`ImageGeneratorCallback`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after constructor combinations, directory creation,
        sampling arguments, image/GIF paths, plotting flags, and hook returns.
    """

    import tempfile
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock, patch


    for invalid_kwargs in (
        {"show_images": False, "save_gifs": False, "results_path": None}, 
        {"show_images": True, "save_gifs": True, "results_path": None}, 
        {"show_images": False, "save_gifs": True, "results_path": None}, 
    ):
        try:
            ImageGeneratorCallback(**invalid_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid output-mode combinations must fail.")

    display_callback = ImageGeneratorCallback(
        add_null_label=False,
        show_images=True,
        seed=13,
    )
    display_sample = Mock(return_value="images")
    display_callback.set_model(SimpleNamespace(
        test_steps=4, 
        test_cfg_scale=1.5, 
        test_eta=0.25, 
        test_network_name="raw",
        get_network=Mock(return_value=SimpleNamespace(
            use_cfg=True,
            num_labels=3,
        )),
        sample=display_sample, 
    ))
    with patch.object(sys.modules[__name__], "plot_images") as plot_mock:
        assert display_callback.on_epoch_end(0, {"loss": 1.0}) is None
    display_sample.assert_called_once_with(
        network_name="raw", labels=[1, 2], steps=4, scale=1.5, eta=0.25,
        return_x_ts=False, return_x0s=False, seed=13,
    )
    plot_mock.assert_called_once_with("images")

    with tempfile.TemporaryDirectory() as png_directory:
        png_callback = ImageGeneratorCallback(
            show_images=False, 
            save_gifs=False, 
            results_path=png_directory, 
        )
        assert os.path.isdir(png_callback.results_path)

    with tempfile.TemporaryDirectory() as temporary_directory:
        saving_callback = ImageGeneratorCallback(
            show_images=False, 
            save_gifs=True, 
            results_path=temporary_directory, 
            project_tag="smoke", 
        )
        result_root = Path(saving_callback.results_path)
        assert result_root.parent == Path(temporary_directory)
        assert result_root.name.endswith(" smoke")
        assert (result_root / "images").is_dir()
        assert (result_root / "gifs").is_dir()
        saving_callback.set_artifact_prefix("task-2_classes-4-5")
        assert saving_callback.get_config() == {
            "add_null_label": True,
            "show_images": False,
            "save_gifs": True,
            "seed": None,
        }
        try:
            saving_callback.set_artifact_prefix(3)
        except TypeError:
            pass
        else:
            raise AssertionError("Non-string artifact prefixes must fail.")

        frames_one = ["frame-1"]
        frames_two = ["frame-2"]
        save_sample = Mock(return_value=("saved-images", frames_one, frames_two))
        saving_callback.set_model(SimpleNamespace(
            test_steps=3, 
            test_cfg_scale=2.0, 
            test_eta=0.125, 
            test_network_name="ema",
            get_network=Mock(return_value=SimpleNamespace(
                use_cfg=True,
                num_labels=3,
            )),
            sample=save_sample, 
        ))
        with patch.object(
            sys.modules[__name__], "create_gif",
        ) as gif_mock, patch.object(
            sys.modules[__name__], "plot_images",
        ) as saved_plot_mock:
            assert saving_callback.on_epoch_end(1, None) is None
        save_sample.assert_called_once_with(
            network_name="ema", labels=[0, 1, 2],
            steps=3, 
            scale=2.0, 
            eta=0.125, 
            return_x_ts=True, 
            return_x0s=True, 
            seed=None,
        )
        gif_args, gif_kwargs = gif_mock.call_args
        assert Path(gif_args[0]).name == (
            "task-2_classes-4-5_epoch-2_steps-3_scale-2.0_eta-0.1250.gif"
        )
        assert gif_args[1:] == (frames_one, frames_two)
        assert gif_kwargs == {"verbose": 0}
        plot_args, plot_kwargs = saved_plot_mock.call_args
        assert plot_args == ("saved-images",)
        assert plot_kwargs["show_images"] is False
        assert Path(plot_kwargs["save_path"]).name == (
            "task-2_classes-4-5_epoch-2_steps-3_scale-2.0_eta-0.1250.png"
        )

        shown_saving_callback = ImageGeneratorCallback(
            show_images=True, 
            save_gifs=True, 
            results_path=temporary_directory, 
        )
        shown_sample = Mock(return_value=("shown-images", [], []))
        shown_saving_callback.set_model(SimpleNamespace(
            test_steps=1, 
            test_cfg_scale=1.0, 
            test_eta=0.0, 
            test_network_name="raw",
            get_network=Mock(return_value=SimpleNamespace(
                use_cfg=False,
                num_labels=2,
            )),
            sample=shown_sample, 
        ))
        with patch.object(
            sys.modules[__name__], "create_gif",
        ) as shown_gif_mock, patch.object(
            sys.modules[__name__], "plot_images",
        ) as shown_plot_mock:
            shown_saving_callback.on_epoch_end(0)
        shown_sample.assert_called_once_with(
            network_name="raw", labels=[0, 1],
            steps=1, 
            scale=1.0, 
            eta=0.0, 
            return_x_ts=True, 
            return_x0s=True, 
            seed=None,
        )
        assert shown_gif_mock.call_count == 1
        assert shown_plot_mock.call_args.kwargs["show_images"] is True

    return {"ImageGeneratorCallback": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
