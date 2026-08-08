from tensorflow.keras import callbacks

import os

from datetime import datetime

from common.utils import plot_images, create_gif


class ImageGeneratorCallback(callbacks.Callback):

    def __init__(
        self, 
        show_images=True, 
        save_gifs=False, 
        results_path=None, 
        project_tag=None, 
        **kwargs
    ):
        super().__init__(**kwargs)

        assert show_images or results_path is not None
        assert (save_gifs and results_path is not None) or \
            (not save_gifs and results_path is None)


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
