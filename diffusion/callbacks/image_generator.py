"""Epoch-end diffusion sampling, image plotting, and denoising GIF output.

ImageGenerator samples the selected raw/EMA diffusion network at a configurable
epoch frequency, displays or saves image grids, and optionally saves denoising
trajectories as GIFs. Constructor output modes control immediate directory
creation; portable phase prefixes keep later task artifacts distinct.
"""

from tensorflow.keras import callbacks

import os

from datetime import datetime

from numbers import Integral
from typing import Any

from common.utils import plot_images, create_gif
from common.result_directory import reserve_result_directory


class ImageGenerator(callbacks.Callback):
    """Generate qualitative diffusion samples on selected training epochs.

    The callback expects a ``DiffusionModel``-compatible bound model exposing
    ``test_steps``, ``test_cfg_scale``, ``test_eta``, ``test_network_name``, and
    ``sample``, plus ``use_cfg`` to identify null previews. Valid constructor
    combinations in the current implementation are:

    * display only: ``show_images=True``, ``save_gifs=False``, and
      ``results_path=None``;
    * save PNGs, with optional GIFs: a non-``None`` ``results_path``;
      ``show_images`` controls simultaneous display.

    A dated run directory is created immediately during construction when
    saving. GIF output additionally requires a result path. Sampling occurs
    after every ``frequency`` completed epochs, when ``epoch + 1`` is divisible
    by ``frequency``; skipped epochs do no sampling, plotting, or GIF work.
    The supplied Keras epoch index determines the schedule, including when
    resuming with ``initial_epoch``. There is no extra sample at training end.

    Args:
        add_null_label (bool): Whether a CFG model's null condition is included in
            the generated grid. Forwarded to ``sample`` with labels omitted, so
            the wrapper chooses observed dynamic or all fixed-width classes.
            Defaults to ``True``.
        show_images (bool): Whether ``plot_images`` displays the generated image grid.
            Defaults to ``True``.
        save_gifs (bool): Whether to request intermediate ``x_t`` and ``x_0`` frames
            and write a denoising GIF per selected epoch.
            Defaults to ``False``.
        results_path (str | os.PathLike[str] | None): Optional string or path-like base directory. A timestamped
            child containing ``images`` is created; ``gifs`` is added when GIF
            saving is enabled.
            Defaults to ``None``.
        project_tag (str | None): Optional text appended to the timestamped directory name.
            Defaults to ``None``.
        frequency (int): Positive integer interval between samples.
            Defaults to ``1`` (every epoch). For example, ``frequency=5`` samples
            after one-based epochs 5, 10, 15, and so on. Booleans and nonintegers
            are rejected.
        seed (int | None): Optional sampling seed reused at each epoch.
            Defaults to ``None``.
        **kwargs (Any): Arguments forwarded to ``tf.keras.callbacks.Callback``. The
            base callback normally requires no extra options.

    Inputs:
        Keras supplies a zero-based integer epoch and optional metric mapping;
        the bound diffusion wrapper supplies sampling configuration and images.

    Outputs:
        Callback hooks return ``None``. Observable outputs are displayed image
        grids and, when configured, PNG and GIF files.

    Attributes:
        results_path (str | os.PathLike | None): Resolved timestamped run directory, or
            None for display-only mode.
        seed (int | None): Current seed forwarded to each sampling call.
        base_seed (int | None): Original constructor seed retained for recovery fingerprints.
        artifact_prefix (str): Validated filename prefix, initially empty.
        frequency (int): Positive epoch interval, normalized to a Python integer.
    """

    def __init__(
        self, 
        add_null_label: bool = True, 
        show_images: bool = True, 
        save_gifs: bool = False, 
        results_path: str | os.PathLike[str] | None = None, 
        project_tag: str | None = None, 
        frequency: int = 1, 
        seed: int | None = None, 
        **kwargs: Any
    ) -> None:
        """Validate output mode and create the timestamped result directories.

        Args:
            add_null_label (bool): Include condition ID 0 for CFG models when
                generating the epoch grid.
                Defaults to ``True``.
            show_images (bool): Whether to display each generated image grid.
                Defaults to ``True``.
            save_gifs (bool): Whether to save intermediate denoising frames as
                a GIF for each selected epoch.
                Defaults to ``False``.
            results_path (str | os.PathLike[str] | None): Optional output base
                directory. A timestamped run directory is created beneath it.
                Defaults to ``None``.
            project_tag (str | None): Optional suffix for the run-directory
                name.
                Defaults to ``None``.
            frequency (int): Positive integer sampling interval. The first
                sample is generated after one-based epoch ``frequency``.
                Defaults to ``1`` (every epoch). Booleans and nonintegers are
                rejected before creating output directories.
            seed (int | None): Optional seed forwarded to model sampling.
                Defaults to ``None``.
                None is forwarded unchanged to model.sample, leaving seed resolution to the
                bound wrapper.
            **kwargs (Any): Options forwarded to the Keras callback base class.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: If ``frequency`` is not a positive integer,
                ``project_tag`` is not a portable filename fragment, or the
                requested output mode cannot emit artifacts.
        """

        super().__init__(**kwargs)

        # Reject invalid intervals before modulo arithmetic or filesystem writes.
        if isinstance(frequency, bool) or not isinstance(frequency, Integral) or frequency <= 0:
            raise ValueError("frequency must be a positive integer.")
        # Reject configurations that neither display nor save generated images.
        if not show_images and results_path is None:
            raise ValueError("The callback must show or save images.")
        # Require an output directory whenever GIF saving is enabled.
        if save_gifs and results_path is None:
            raise ValueError("save_gifs requires results_path.")
        # Treat an omitted project tag as empty; trim a supplied filename suffix.
        normalized_project_tag = "" if project_tag is None else project_tag.strip()
        invalid_filename_characters = frozenset('/\\<>:"|?*')
        # Keep the directory suffix portable and inside the requested root.
        if any(
            ord(character) < 32
            or character in invalid_filename_characters
            for character in normalized_project_tag
        ) or normalized_project_tag.endswith("."):
            raise ValueError(
                "project_tag must be a portable filename fragment."
            )

        self.add_null_label = add_null_label
        self.show_images = show_images
        self.save_gifs = save_gifs
        self.results_path = results_path
        self.frequency = int(frequency)
        self.seed = seed
        self.base_seed = seed
        self.artifact_prefix = ""

        # Atomically reserve a distinct artifact directory for each new execution.
        if self.results_path is not None:
            self.results_path = str(reserve_result_directory(
                self.results_path,
                normalized_project_tag,
                timestamp=datetime.now()
            ))

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
        prefix: str | None
    ) -> None:
        """Set a safe filename prefix for a later training phase or task.

        Args:
            prefix (str | None): Prefix without a trailing separator. ``None``
                or an empty string restores the ordinary epoch-only names.

        Returns:
            None: Subsequent PNG/GIF filenames are updated in place.

        Raises:
            ValueError: If ``prefix`` is not a portable filename fragment.
        """

        # Treat an omitted task prefix as empty; trim a supplied filename prefix.
        normalized = "" if prefix is None else prefix.strip()
        invalid_filename_characters = frozenset('/\\<>:"|?*')
        # Keep the artifact name portable and inside its callback-owned folder.
        if any(
            ord(character) < 32
            or character in invalid_filename_characters
            for character in normalized
        ) or normalized.endswith("."):
            raise ValueError("prefix must be a portable filename fragment.")

        # Append a separator only to a nonempty artifact prefix.
        self.artifact_prefix = normalized + "_" if normalized else ""

    def get_config(self) -> dict[str, object]:
        """Return behavior-defining callback options for recovery fingerprints.

        Returns:
            dict[str, object]: Sampling frequency, display/GIF options and the
            initial seed. Filesystem paths and mutable task prefixes are excluded.
        """

        return {
            "add_null_label": self.add_null_label, 
            "show_images": self.show_images, 
            "save_gifs": self.save_gifs, 
            "frequency": self.frequency, 
            "seed": self.base_seed
        }

    def on_epoch_end(
        self, 
        epoch: int, 
        logs: dict[str, Any] | None = None
    ) -> None:
        """Sample and render artifacts when the epoch index is due.

        Args:
            epoch (int): Zero-based epoch index. Output filenames use
                ``epoch + 1``. Artifacts are generated only when this one-based
                epoch number is divisible by ``frequency``; all other calls
                return before accessing the model.
            logs (dict[str, Any] | None): Optional Keras epoch-log mapping. It
                is accepted for callback compatibility and is not read.
                Defaults to ``None``. No caller-owned log mapping is available in that case.

        Returns:
            None: ``model.sample`` returns images shaped
            ``[batch, height, width, channels]``. In GIF mode it must return
            ``(images, x_t_frames, x0_frames)``; the frame sequences are passed
            to ``create_gif``.
        """

        # Sample after each complete interval of one-based training epochs.
        if (epoch + 1) % self.frequency != 0:
            return

        sample_kwargs = {
            "network_name": self.model.test_network_name,
            "add_null_label": self.add_null_label,
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

        has_null_label = self.add_null_label and self.model.use_cfg
        # Save the image grid, optionally displaying it at the same time.
        if self.results_path is not None: 
            plot_images(
                imgs, 
                has_null_label=has_null_label,
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
            plot_images(imgs, has_null_label=has_null_label)


def run_self_tests() -> dict[str, str]:
    """Test sampling cadence and output modes of :class:`ImageGenerator`.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after frequency validation, positional
        argument order, cadence, directory creation, sampling arguments,
        image/GIF paths, plotting flags, and hook returns pass.
    """

    import tempfile
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    import numpy as np


    with tempfile.TemporaryDirectory() as validation_directory:
        absent_root = Path(validation_directory) / "must-not-exist"
        for invalid_frequency in (0, -1, True, False, 1.0, 1.5, "2", None, np.bool_(True)):
            try:
                ImageGenerator(frequency=invalid_frequency, results_path=absent_root)
            except ValueError as error:
                assert str(error) == "frequency must be a positive integer."
            # Invalid intervals must fail before they reserve a result directory.
            else:
                raise AssertionError("Invalid sampling frequencies must fail.")
            assert not absent_root.exists()

    positional_callback = ImageGenerator(False, True, False, None, None, 5, 13)
    assert positional_callback.add_null_label is False
    assert positional_callback.show_images is True
    assert positional_callback.save_gifs is False
    assert positional_callback.results_path is None
    assert positional_callback.seed == 13
    assert positional_callback.frequency == 5

    interval_callback = ImageGenerator(frequency=np.int64(3))
    assert type(interval_callback.frequency) is int
    assert interval_callback.get_config()["frequency"] == 3
    # A skipped hook must not even require a bound model.
    assert interval_callback.on_epoch_end(0) is None
    assert interval_callback.on_epoch_end(1) is None

    for invalid_kwargs in (
        {"show_images": False, "save_gifs": False, "results_path": None}, 
        {"show_images": True, "save_gifs": True, "results_path": None}, 
        {"show_images": False, "save_gifs": True, "results_path": None}, 
    ):
        try:
            ImageGenerator(**invalid_kwargs)
        except ValueError:
            pass
        # This invalid case should already have raised: Invalid output-mode combinations
        # must fail.
        else:
            raise AssertionError("Invalid output-mode combinations must fail.")
    for unsafe_tag in (
        "../escape",
        "..\\escape",
        "bad\0name",
        "bad:name",
        "bad*name",
        "bad\nname",
        "trailing.",
    ):
        try:
            ImageGenerator(project_tag=unsafe_tag)
        except ValueError:
            pass
        # Project tags must not escape the callback's results root.
        else:
            raise AssertionError("Path-like project tags must fail.")
    display_callback = ImageGenerator(
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
        use_cfg=True,
        sample=display_sample, 
    ))
    with patch.object(sys.modules[__name__], "plot_images") as plot_mock:
        assert display_callback.on_epoch_end(0, {"loss": 1.0}) is None
    display_sample.assert_called_once_with(
        network_name="raw", add_null_label=False, steps=4, scale=1.5, eta=0.25,
        return_x_ts=False, return_x0s=False, seed=13,
    )
    plot_mock.assert_called_once_with("images", has_null_label=False)

    interval_sample = Mock(return_value="interval-images")
    interval_callback.set_model(SimpleNamespace(
        test_steps=4, test_cfg_scale=1.5, test_eta=0.25,
        test_network_name="raw", use_cfg=False, sample=interval_sample,
    ))
    with patch.object(sys.modules[__name__], "plot_images") as interval_plot:
        # Starting partway through a fit retains the supplied absolute epoch schedule.
        for epoch, expected_count in ((3, 0), (4, 0), (5, 1), (6, 1), (7, 1), (8, 2)):
            interval_callback.on_epoch_end(epoch)
            assert interval_sample.call_count == interval_plot.call_count == expected_count

    with patch.object(sys.modules[__name__], "plot_images") as default_plot:
        for epoch in (1, 2, 3):
            display_callback.on_epoch_end(epoch)
        assert display_sample.call_count == 4
        assert default_plot.call_count == 3

    with tempfile.TemporaryDirectory() as png_directory:
        png_callback = ImageGenerator(
            show_images=False, 
            save_gifs=False, 
            results_path=png_directory, 
        )
        assert os.path.isdir(png_callback.results_path)

    with tempfile.TemporaryDirectory() as temporary_directory:
        saving_callback = ImageGenerator(
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
            "frequency": 1,
            "add_null_label": True,
            "show_images": False,
            "save_gifs": True,
            "seed": None,
        }
        for unsafe_prefix in (
            "../escape",
            "..\\escape",
            "bad\0name",
            "bad:name",
            "bad*name",
            "bad\nname",
            "trailing.",
        ):
            try:
                saving_callback.set_artifact_prefix(unsafe_prefix)
            except ValueError:
                pass
            # Unsafe path fragments must never reach artifact path assembly.
            else:
                raise AssertionError("Path-like artifact prefixes must fail.")

        frames_one = ["frame-1"]
        frames_two = ["frame-2"]
        save_sample = Mock(return_value=("saved-images", frames_one, frames_two))
        saving_callback.set_model(SimpleNamespace(
            test_steps=3, 
            test_cfg_scale=2.0, 
            test_eta=0.125, 
            test_network_name="ema",
            use_cfg=True,
            sample=save_sample, 
        ))
        with patch.object(
            sys.modules[__name__], "create_gif",
        ) as gif_mock, patch.object(
            sys.modules[__name__], "plot_images",
        ) as saved_plot_mock:
            assert saving_callback.on_epoch_end(1, None) is None
        save_sample.assert_called_once_with(
            network_name="ema", add_null_label=True,
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
        assert plot_kwargs["has_null_label"] is True
        assert Path(plot_kwargs["save_path"]).name == (
            "task-2_classes-4-5_epoch-2_steps-3_scale-2.0_eta-0.1250.png"
        )

        periodic_saving_callback = ImageGenerator(
            frequency=5, show_images=False, save_gifs=True,
            results_path=temporary_directory,
        )
        periodic_saving_callback.set_model(saving_callback.model)
        save_sample.reset_mock()
        with patch.object(sys.modules[__name__], "create_gif") as periodic_gif, \
             patch.object(sys.modules[__name__], "plot_images") as periodic_plot:
            expected_counts = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
            for epoch, expected_count in enumerate(expected_counts):
                logs = {"loss": 1.0}
                periodic_saving_callback.on_epoch_end(epoch, logs)
                assert logs == {"loss": 1.0}
                assert save_sample.call_count == expected_count
                assert periodic_plot.call_count == periodic_gif.call_count == expected_count
        assert [Path(call.args[0]).name for call in periodic_gif.call_args_list] == [
            f"epoch-{epoch}_steps-3_scale-2.0_eta-0.1250.gif" for epoch in (5, 10, 15)
        ]
        assert [Path(call.kwargs["save_path"]).name for call in periodic_plot.call_args_list] == [
            f"epoch-{epoch}_steps-3_scale-2.0_eta-0.1250.png" for epoch in (5, 10, 15)
        ]

        shown_saving_callback = ImageGenerator(
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
            use_cfg=False,
            sample=shown_sample, 
        ))
        with patch.object(
            sys.modules[__name__], "create_gif",
        ) as shown_gif_mock, patch.object(
            sys.modules[__name__], "plot_images",
        ) as shown_plot_mock:
            shown_saving_callback.on_epoch_end(0)
        shown_sample.assert_called_once_with(
            network_name="raw", add_null_label=True,
            steps=1, 
            scale=1.0, 
            eta=0.0, 
            return_x_ts=True, 
            return_x0s=True, 
            seed=None,
        )
        assert shown_gif_mock.call_count == 1
        assert shown_plot_mock.call_args.kwargs["show_images"] is True
        assert shown_plot_mock.call_args.kwargs["has_null_label"] is False

    return {"ImageGenerator": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
