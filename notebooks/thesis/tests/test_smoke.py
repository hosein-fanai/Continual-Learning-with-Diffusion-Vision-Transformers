"""Small isolated actual API execution; synthetic pixels are never thesis results."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import yaml

from common.dataloader import get_datasets
from common.model import get_model
from common.train import train_model
from notebooks.thesis import workflow
from semantic_consolidation.config import load_route_config


class SelectedApiSmokeTests(unittest.TestCase):
    """Bounded saved-evidence regression fixtures; never research outcomes."""

    def tearDown(self) -> None:
        """Release this test case's runtime and temporary resources.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        tf.keras.backend.clear_session()

    def test_chosen_capacity_constructs_and_predicts_full_cifar_vocabularies(self) -> None:
        """Exercise chosen DiT capacity and all-seen clean prediction, without fitting.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        directory = Path(workflow.__file__).parent
        for dataset, classes in (("cifar10", 10), ("cifar100", 100)):
            config = load_route_config(directory / "configs" / f"{dataset}.yaml")
            bundle = get_model(config.common)
            model = bundle["generative_model"]
            model._check_new_labels(y=np.arange(classes), verbose=False)
            images = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
            times = tf.zeros((2,), dtype=tf.int32)
            probabilities = model.network.predict_class((images, times, times), training=False)
            self.assertEqual(tuple(probabilities.shape), (2, classes))
            self.assertTrue(np.isfinite(probabilities.numpy()).all())
            np.testing.assert_allclose(np.sum(probabilities, axis=1), 1., atol=1e-6)
            self.assertEqual(model.network.dim, 128)
            integer_state = sum(int(np.prod(variable.shape)) for variable in model.network.weights
                                if tf.as_dtype(variable.dtype).is_integer)
            print(f"{dataset}: full-width model parameters={model.network.count_params()}, "
                  f"integer random-state scalars={integer_state}")
            tf.keras.backend.clear_session()

    def test_interrupted_stream_resumes_and_recovers_publication_without_retraining(self) -> None:
        """Resume real committed fit progress through the staged notebook API.

        Synthetic images exercise actual native loading, fitting, checkpoint
        restoration and reporting. Only the checkpoint writer is interrupted
        after its successful commit; no training objective is replaced.

        Args:
            None. Fixtures are owned by this unittest instance.

        Returns:
            result (None): None; unittest records the assertions and any failure.

        Raises:
            AssertionError: If the stated regression invariant fails.
        """
        notebook_dir = Path(workflow.__file__).parent
        with tempfile.TemporaryDirectory(prefix="SYNTHETIC_STAGED_SMOKE_", dir=notebook_dir / "tests") as temporary:
            directory = Path(temporary)
            templates = {}
            for dataset in workflow.CONDITIONS:
                config = load_route_config(notebook_dir / "configs" / f"{dataset}.yaml")
                config.common.dataset.batch_size = 8
                config.common.dataset.max_train_samples = None
                config.common.model.kwargs.update(dim=8, depth=1, patch_size=8,
                                                  mha_num_heads=1, clf_mha_num_heads=1, timesteps=4)
                config.common.model.wrapper_kwargs.update(test_steps=2)
                config.common.training.epochs = 1
                config.common.training.verbose = 0
                config.common.continually_learn.class_num = 4
                config.common.continually_learn.task_size = 2
                config.common.continually_learn.replay_old_examples = 16
                config.route.acquisition_steps = 4
                config.route.consolidation_steps = 4
                config.route.checkpoint_interval = 1
                config.route.batch_size = 8
                config.route.probe_batches = 2
                config.route.experimental.update(batch_size=8)
                path = directory / f"{dataset}.yaml"
                path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
                templates[dataset] = path
            frozen = workflow.prepare_campaign(directory / "campaign", templates, workflow.CONFIRMATION_SEEDS)
            labels = np.repeat(np.arange(4, dtype="uint8"), 20)
            images = np.random.default_rng(6).integers(0, 255, (80, 32, 32, 3), dtype="uint8")
            test_labels = np.repeat(np.arange(4, dtype="uint8"), 4)
            test_images = np.full((16, 32, 32, 3), 127, dtype="uint8")
            with patch("tensorflow.keras.datasets.cifar10.load_data", return_value=(
                    (images, labels[:, None]), (test_images, test_labels[:, None]))):
                config, context = workflow.load_run(frozen, "cifar10", "learned", repeat_index=None)
                trainset, valset = get_datasets(config.common)
                bundle = get_model(config.common)
                original = bundle["generative_model"].network
                optimizer = bundle["generative_model"].optimizer
                workflow.attach_route(context, bundle)
                self.assertIs(bundle["generative_model"].network, original)
                self.assertIs(bundle["generative_model"].optimizer, optimizer)
                receipt = context["started_marker"].read_bytes()
                planned = context["config_path"].read_bytes()
                from common import learner
                original_writer = learner.save_task_progress

                def interrupt_after_commit(root: str, task: int, state: dict, trackables: dict) -> Path:
                    """Commit actual state before simulating process failure.

                    Args:
                        root (str): Native stream checkpoint destination.
                        task (int): Zero-based active task index.
                        state (dict): Real fit state from the production adapter.
                        trackables (dict): Native checkpointable dataset iterator.

                    Returns:
                        checkpoint (Path): Never returns; the successful commit
                            is followed by the deliberate injected interruption.

                    Raises:
                        OSError: After the production checkpoint commit succeeds.
                    """
                    original_writer(root, task, state, trackables)
                    raise OSError("injected interruption after real fit commit")

                try:
                    with patch.object(learner, "save_task_progress", side_effect=interrupt_after_commit):
                        with self.assertRaisesRegex(OSError, "after real fit commit"):
                            train_model(config.common, bundle, trainset, valset=valset)
                finally:
                    workflow.close_run(context, release=True)
                self.assertFalse(context["finished"])
                tf.keras.backend.clear_session()
                config, context = workflow.load_run(frozen, "cifar10", "learned", repeat_index=None)
                self.assertIsNotNone(config.common.continually_learn.resume_from)
                self.assertEqual(context["entry"]["block_id"], "stream-01")
                trainset, valset = get_datasets(config.common)
                bundle = get_model(config.common)
                workflow.attach_route(context, bundle)
                self.assertEqual(context["started_marker"].read_bytes(), receipt)
                self.assertEqual(context["config_path"].read_bytes(), planned)
                try:
                    history = train_model(config.common, bundle, trainset, valset=valset)
                finally:
                    workflow.close_run(context)
                from notebooks.thesis import completion
                with patch.object(completion, "replace_completed_index", side_effect=OSError("fault after completed artifact")):
                    with self.assertRaisesRegex(OSError, "fault after completed artifact"):
                        workflow.finish_run(context, config, bundle, history, trainset, valset)
            completed_path = context["manifest_path"].parent / f"{context['entry']['run_id']}.completed.json"
            self.assertTrue(completed_path.is_file())
            with patch("common.train.train_model", side_effect=AssertionError("Recovery must never train")), \
                 patch("common.train.report", side_effect=AssertionError("Recovery must never re-report")):
                workflow.finish_run(context, config, bundle, history, trainset, valset)
                record, manifests = workflow._campaign(frozen)
                outputs = workflow._outputs(context["manifest_path"], manifests["cifar10"])
                self.assertEqual(len(outputs), 1)
                with patch.object(workflow, "_initialize", side_effect=lambda cfg, ctx: (cfg, ctx)):
                    _, next_context = workflow.load_run(frozen, "cifar10", "learned", repeat_index=None)
                self.assertEqual(next_context["entry"]["block_id"], "stream-02")
            self.assertTrue(context["finished"])
            self.assertEqual(np.asarray(outputs[context["entry"]["run_id"]]["accuracy_matrix"], dtype=float).shape, (2, 2))


# Run this isolated unittest module when invoked as a script.
if __name__ == "__main__":
    unittest.main()
