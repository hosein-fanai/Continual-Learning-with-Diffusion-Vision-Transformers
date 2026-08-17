"""Epoch-end diffusion sampling, image plotting, and denoising GIF output."""

from tensorflow.keras import callbacks

import os

from datetime import datetime

from common.utils import plot_images, create_gif


class ImageGeneratorCallback(callbacks.Callback):
    """Generate qualitative diffusion samples after every training epoch.

    The callback expects a ``DiffusionModel``-compatible bound model exposing
    ``test_steps``, ``test_cfg_scale``, ``test_eta``, and ``sample``. Valid
    constructor combinations in the current implementation are:

    * display only: ``show_images=True``, ``save_gifs=False``, and
      ``results_path=None``;
    * save PNGs and GIFs: ``save_gifs=True`` and a non-``None``
      ``results_path``; ``show_images`` then controls simultaneous display.

    Supplying ``results_path`` while ``save_gifs=False`` is rejected. A dated
    run directory is created immediately during construction when saving.

    Args:
        show_images: Whether ``plot_images`` displays the generated image grid.
        save_gifs: Whether to request intermediate ``x_t`` and ``x_0`` frames
            and write a denoising GIF per epoch.
        results_path: Optional string or path-like base directory. A timestamped
            child containing ``images`` and ``gifs`` is created when GIF saving
            is enabled.
        project_tag: Optional text appended to the timestamped directory name.
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
        show_images: bool = True, 
        save_gifs: bool = False, 
        results_path: str | None = None, 
        project_tag: str | None = None, 
        **kwargs
    ):
        """Validate output mode and create the timestamped result directories.

        Arguments and accepted types are documented on the class.

        Returns:
            ``None``.
        """

        super().__init__(**kwargs)

        assert show_images or results_path is not None, \
            "The callback needs to either show images or save them."
        assert (save_gifs and results_path is not None) or \
            (not save_gifs and results_path is None), \
                "save_gifs needs to be matched with results_path."


        self.show_images = show_images
        self.save_gifs = save_gifs
        self.results_path = results_path

        project_tag = "" if project_tag is None else " " + project_tag

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
            if save_gifs:
                os.makedirs(
                    os.path.join(self.results_path, "gifs"), 
                    exist_ok=True
                )

    def on_epoch_end(self, epoch, logs=None):
        """Sample the bound diffusion model and render epoch artifacts.

        Args:
            epoch: Zero-based integer epoch index. Output filenames use
                ``epoch + 1``.
            logs: Optional Keras epoch-log mapping. It is accepted for callback
                compatibility and is not read or modified.

        Returns:
            ``None``. ``model.sample`` returns images shaped
            ``[batch, height, width, channels]``. In GIF mode it must return
            ``(images, x_t_frames, x0_frames)``; the frame sequences are passed
            to ``create_gif``.
        """

        steps = self.model.test_steps
        cfg_scale = self.model.test_cfg_scale
        eta = self.model.test_eta

        if self.save_gifs:
            imgs, frames1, frames2 = self.model.sample(
                steps=steps, 
                scale=cfg_scale, 
                eta=eta, 
                return_x_ts=True, 
                return_x0s=True, 
            )
            create_gif(
                os.path.join(self.results_path, "gifs", 
                            f"epoch-{epoch+1}_steps-{steps}_scale-{cfg_scale:.1f}_eta-{eta:.4f}.gif"), 
                frames1, frames2, 
                verbose=0
            )
        else:
            imgs = self.model.sample(
                steps=steps, 
                scale=cfg_scale, 
                eta=eta, 
            )

        if self.results_path is not None: 
            plot_images(
                imgs, 
                show_images=self.show_images, 
                save_path=os.path.join(self.results_path, "images", 
                                    f"epoch-{epoch+1}_steps-{steps}_scale-{cfg_scale:.1f}_eta-{eta:.4f}.png") 
            )
        else:
            plot_images(imgs)
