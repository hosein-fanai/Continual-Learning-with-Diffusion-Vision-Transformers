"""Exercise exclusive new-run ownership through callbacks, training and reports."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from common.config import Config
from common.result_directory import reserve_result_directory
from common.train import report, train_model


class _FixedClock:
    """Supply the same instant to every callback allocation in a regression."""

    @staticmethod
    def now() -> datetime:
        """Return a deterministic timestamp, including for concurrent calls."""
        return datetime(2026, 9, 6, 12, 0, 0)


class _EvaluationOnly:
    """Expose a controlled value through the actual ordinary-report interface."""

    def __init__(self, accuracy: float) -> None:
        """Retain the synthetic accuracy belonging to this independent report."""
        self.accuracy = accuracy

    def evaluate(self, *args: object, **kwargs: object) -> dict:
        """Return a known value without suggesting it is a trained outcome."""
        return {"accuracy": self.accuracy}


class ResultDirectoryTests(unittest.TestCase):
    """A new execution owns one distinct destination across its artifact writers."""

    def test_same_clock_callbacks_preserve_independent_reports(self) -> None:
        """The archived overwrite counterexample now retains both original CSVs."""
        module = importlib.import_module('diffusion.callbacks.image_generator_callback')
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'datetime', _FixedClock):
            first = module.ImageGeneratorCallback(show_images=False, results_path=directory, project_tag='paired')
            second = module.ImageGeneratorCallback(show_images=False, results_path=directory, project_tag='paired')
            self.assertNotEqual(first.results_path, second.results_path)
            for callback, accuracy in ((first, .9), (second, .1)):
                report(history={}, model=_EvaluationOnly(accuracy), trainset=None, valset=object(),
                       results_path=callback.results_path, save_csv=True, show_history_plot=False,
                       save_history_plot=False, run_trainset_eval=False, run_valset_eval=True,
                       show_final_images=False, save_final_images=False, save_final_gifs=False)
            for callback, accuracy in ((first, .9), (second, .1)):
                values = pd.read_csv(Path(callback.results_path) / 'evals history.csv', index_col=0)
                self.assertEqual(values.loc['valset_eval', 'accuracy'], accuracy)
                self.assertTrue((Path(callback.results_path) / 'images').is_dir())

    def test_concurrent_same_clock_reservations_are_exclusive(self) -> None:
        """Filesystem ownership remains unique when 64 callers contend at once."""
        with tempfile.TemporaryDirectory() as directory:
            def reserve(index: int) -> Path:
                """Claim and mark one independent destination without a process lock."""
                path = reserve_result_directory(directory, 'paired', timestamp=_FixedClock.now())
                (path / 'owner.txt').write_text(str(index))
                return path
            with ThreadPoolExecutor(max_workers=16) as executor:
                paths = list(executor.map(reserve, range(64)))
            self.assertEqual(len(set(paths)), 64)
            for index, path in enumerate(paths):
                self.assertEqual(path.parent, Path(directory).resolve())
                self.assertEqual((path / 'owner.txt').read_text(), str(index))

    def test_occupied_files_and_invalid_tags_never_alias(self) -> None:
        """Occupied names survive retries and path-like tags fail before creation."""
        with tempfile.TemporaryDirectory() as directory:
            occupied = Path(directory) / '2026-09-06_12-00-00 paired'
            occupied.write_text('historical evidence')
            path = reserve_result_directory(directory, 'paired', timestamp=_FixedClock.now())
            self.assertNotEqual(path, occupied)
            self.assertEqual(occupied.read_text(), 'historical evidence')
            for tag in ('../outside', 'bad\\path', 'bad:', 'trailing.', 12):
                absent = Path(directory) / 'must-not-exist'
                with self.subTest(tag=tag), self.assertRaises(ValueError):
                    reserve_result_directory(absent, tag)
                self.assertFalse(absent.exists())

    def test_typed_training_shares_one_reservation_with_artifact_writers(self) -> None:
        """Two real one-epoch fits keep configs, weights and reports in separate roots."""
        module = importlib.import_module('diffusion.callbacks.image_generator_callback')
        dataset = tf.data.Dataset.from_tensor_slices((
            np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]], dtype=np.float32),
            np.array([0, 1, 0, 1], dtype=np.int32),
        )).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = dataset.with_options(options)
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'datetime', _FixedClock):
            roots = []
            for index in range(2):
                config = Config()
                config.training.task = 'classification'
                config.training.epochs = 1
                config.training.results_path = directory
                config.training.project_tag = 'same'
                config.training.show_images = False
                config.training.save_gifs = False
                config.training.save_weights = True
                config.training.report_every_epoch = False
                config.training.use_tensorboard = False
                config.training.verbose = 0
                config.training.patience = 0
                model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2, activation='softmax')])
                model.compile(optimizer=tf.keras.optimizers.SGD(.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                state = {}
                history = train_model(config=config, model=model, trainset=dataset, valset=dataset, _run_state=state)
                self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
                output = Path(state['results_path'])
                self.assertEqual(str(output), config.training.results_path)
                self.assertTrue((output / 'input_config.yaml').is_file())
                self.assertTrue((output / 'config.yaml').is_file())
                self.assertTrue((output / 'model.weights.h5').is_file())
                report(history=history, model=model, trainset=None, valset=dataset,
                       results_path=str(output), save_csv=True, show_history_plot=False,
                       save_history_plot=False, run_trainset_eval=False, run_valset_eval=True,
                       show_final_images=False, save_final_images=False, save_final_gifs=False)
                self.assertTrue((output / 'evals history.csv').is_file())
                roots.append(output)
            self.assertNotEqual(roots[0], roots[1])
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
        tf.keras.backend.clear_session()


# Execute just these isolated ownership checks when invoked as a module.
if __name__ == '__main__':
    unittest.main()
