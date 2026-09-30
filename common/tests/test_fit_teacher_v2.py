"""Exercise both V2 teacher optimizers through the shared training API."""

import unittest

import numpy as np
import tensorflow as tf

from common.tests import test_fit_teacher as fixtures
from common.train import train_model
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


class FitTeacherV2Tests(unittest.TestCase):
    """Train fixed-width V2 teacher phases while their student stays unchanged."""

    def setUp(self) -> None:
        """Reuse the existing tiny-network and bounded-dataset fixture."""

        self.fixture = fixtures.FitTeacherTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        """Restore the numerical policy and release the fixture models."""

        self.fixture.tearDown()

    def check_teacher_fit(self, fit_method: str) -> None:
        """Train both teacher phases and verify independent updates and optional growth."""

        model = DiffusionClassifierV2(
            network=self.fixture.make_network(True), 
            teacher_network=self.fixture.make_network(True), 
            trainable_teacher=True, use_ema=False, scheduler_name="linear", 
            test_steps=2, p_uncond=0., seed=541
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="mse", 
                      run_eagerly=True, jit_compile=False)
        self.assertFalse(model.teacher_network.trainable)
        student_before = model.network.get_weights()
        teacher_before = model.teacher_network.get_weights()
        fit_kwargs = {"fit_method": fit_method}
        # Progressive fitting grows only the teacher's generative transformer.
        if fit_method == "fit_progressively":
            fit_kwargs.update(
                stage_tasks=[("depth", "vision_transformer_block")], 
                stage_epochs=1, final_epochs=1, stages_verbose=False
            )
        history = train_model(
            None, model, self.fixture.dataset([0, 1]), fit_method="fit_teacher", 
            fit_kwargs=fit_kwargs, epochs=1, results_path=None, patience=0, 
            save_config_=False, show_images=True, save_gifs=False, report_every_epoch=False, 
            save_weights=False, verbose=0
        )
        teacher = model._teacher_model
        expected_generator_steps = 2 if fit_method == "fit_progressively" else 1
        self.assertEqual(int(teacher.gen_optimizer.iterations.numpy()), expected_generator_steps)
        self.assertEqual(int(teacher.clf_optimizer.iterations.numpy()), 1)
        self.assertEqual(int(model.gen_optimizer.iterations.numpy()), 0)
        self.assertEqual(int(model.clf_optimizer.iterations.numpy()), 0)
        self.assertEqual(model.teacher_network.depth, expected_generator_steps)
        self.assertEqual(model.network.depth, 1)
        self.assertFalse(model.teacher_network.trainable)
        self.assertIsNot(teacher.gen_optimizer, model.gen_optimizer)
        self.assertIsNot(teacher.clf_optimizer, model.clf_optimizer)
        self.assertTrue(any(not np.array_equal(old, new) for old, new in
                            zip(teacher_before, model.teacher_network.get_weights())))
        self.assertTrue(all(np.isfinite(values).all() for values in history.values()))
        self.assertEqual(len(history["noise_loss"]), expected_generator_steps)
        self.assertEqual(len(history["classifier_loss"]), 1)
        self.fixture.assert_student_unchanged(model, student_before)

    def test_ordinary_fit_updates_both_teacher_optimizers(self) -> None:
        """A fixed-width frozen teacher acquires nonempty V2 optimizer groups."""

        self.check_teacher_fit("fit")

    def test_progressive_fit_grows_teacher_then_trains_classifier(self) -> None:
        """Shared dispatch forwards the generator curriculum and classifier phase."""

        self.check_teacher_fit("fit_progressively")


# Allow the focused V2 suite to run as a script.
if __name__ == "__main__":
    unittest.main()
