"""Small real learner lifecycles for data views, callback identity and recovery."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.config import Config
from common.dataloader import get_datasets, _resolve_dataset_options
from common.learner import _run_continual_tasks
from common.model import get_model, validate_progressive_classifier_growth
from common.recovery import (
    callback_recovery_descriptor, fingerprint_state, load_task_checkpoint,
    save_task_checkpoint,
)
from common.train import main
from common.tests import test_continual_integration as integration_fixtures


def image_loader(indices: list[int], **kwargs: object) -> tuple:
    """Return tiny canonical images with separate validation and test arrays.

    Args:
        indices (list[int]): Original class IDs; four float32 4x4x1 rows are generated per class.
        **kwargs (object): onehot_labels=True returns float32 width-four targets; otherwise sparse int64 IDs. Other loader options are ignored.

    Returns:
        arrays (tuple): Six NumPy arrays in train/validation/test image-label order; validation and test are independent copies.
    """
    labels = np.repeat(np.asarray(indices, dtype="int64"), 4)
    images = np.repeat((labels[:, None] / 2. - .75).astype("float32"), 16, axis=1)
    images = images.reshape((-1, 4, 4, 1))
    labels = np.eye(4, dtype="float32")[labels] if kwargs.get("onehot_labels") else labels
    return images, labels, images.copy(), labels.copy(), images.copy(), labels.copy()


class PersistentRateCallback(tf.keras.callbacks.Callback):
    """A declared custom policy whose fit counter must survive task recovery."""

    def __init__(self, rate: float = .002) -> None:
        """Retain immutable behavior separately from the evolving fit count.

        Args:
            rate (float): Numerator for the rate divided by the persistent fit count, default .002.

        Returns:
            result (None): Initialize the callback and its integer zero fit count.
        """
        super().__init__()
        self.rate, self.fit_count = rate, 0

    def get_recovery_config(self) -> dict:
        """Describe the behavior without serializing framework-owned runtime objects.

        Returns:
            config (dict): Immutable behavior declaration containing the floating rate.
        """
        return {"rate": self.rate}

    def get_recovery_state(self) -> dict:
        """Expose the counter needed by the next task's callback invocation.

        Returns:
            state (dict): Persistent integer fit_count required by the next training phase.
        """
        return {"fit_count": self.fit_count}

    def set_recovery_state(self, state: dict) -> None:
        """Restore the authenticated fit counter before resumed optimization.

        Args:
            state (dict): Authenticated callback state containing integer fit_count.

        Returns:
            result (None): Replace the live fit counter.

        Raises:
            KeyError: If fit_count is absent.
        """
        self.fit_count = state["fit_count"]

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Count each task fit across interruption and recovery.

        Args:
            logs (dict | None): Optional Keras metric mapping, unused by this counter.

        Returns:
            result (None): Increment fit_count once for this fit.
        """
        self.fit_count += 1

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Make restored persistent state materially affect the optimizer updates.

        Args:
            epoch (int): Zero-based epoch index, unused by this per-fit policy.
            logs (dict | None): Optional Keras logs, unused.

        Returns:
            result (None): Set the optimizer learning rate to rate/fit_count in its variable dtype.

        Raises:
            ZeroDivisionError: If invoked before any on_train_begin call.
        """
        self.model.optimizer.learning_rate.assign(self.rate / self.fit_count)


class AliasedRateSchedule:
    """Expose nested configuration owned by a configured learning-rate policy."""

    def __init__(self) -> None:
        """Initialize one mutable dictionary whose aliases must not escape recovery.

        Returns:
            result (None): Initialize the mutable nested floating rate configuration.
        """
        self.config = {"rate": {"value": .01}}

    def get_config(self) -> dict:
        """Return the original nested mapping, as supported configured objects may do.

        Returns:
            config (dict): Live nested configuration, intentionally aliased for the snapshot regression.
        """
        return self.config

    def __call__(self, epoch: int, rate: float) -> float:
        """Apply the currently declared constant learning rate.

        Args:
            epoch (int): Epoch argument accepted by LearningRateScheduler, unused.
            rate (float): Previous optimizer rate, ignored by this constant schedule.

        Returns:
            rate (float): Current configured rate, independent of epoch and prior rate.
        """
        return self.config["rate"]["value"]


class AliasedRateCallback(tf.keras.callbacks.Callback):
    """Expose a nested custom configuration without declaring persistent state."""

    recovery_state_scope = "per_fit"

    def __init__(self) -> None:
        """Initialize the callback policy independently of Keras-owned state.

        Returns:
            result (None): Initialize a Keras callback with mutable nested rate configuration.
        """
        super().__init__()
        self.config = {"rate": {"value": .01}}

    def get_recovery_config(self) -> dict:
        """Return the callback's live configuration to exercise defensive snapshots.

        Returns:
            config (dict): Live nested callback policy; the recovery layer must snapshot it.
        """
        return self.config

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Make the declared policy affect the optimizer used by real fitting.

        Args:
            epoch (int): Accepted epoch index, unused.
            logs (dict | None): Optional metric logs, unused.

        Returns:
            result (None): Assign the configured constant optimizer rate in its variable dtype.
        """
        self.model.optimizer.learning_rate.assign(self.config["rate"]["value"])


class LearnerVerifiedRepairTests(unittest.TestCase):
    """Verify complete small streams rather than substituting mocked optimizer updates."""

    def tearDown(self) -> None:
        """Release Keras state and retain the ordinary float32 test environment.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.
        """
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")

    @staticmethod
    def bundle(classifier: str = "dnn") -> dict:
        """Construct the actual conditional VAE and requested independent classifier.

        Args:
            classifier (str): dnn selects the dense classifier; cnn selects a small convolutional classifier.

        Returns:
            models (dict): Compiled conditional VAE and separate classifier in the current policy, initialized with seed 13.
        """
        tf.keras.backend.clear_session()
        classifier_kwargs = {"architecture_kwargs": {"conv_filters": (2,), "conv_depths": (1,)}} \
            if classifier == "cnn" else {}
        return get_model(
            task="continual", model_name="vae", classifier_name=classifier,
            class_num=4, image_shape=(4, 4, 1), flat_dim=16, seed=13,
            model_kwargs={"latent_dim": 2, "hiddens_dims": (4,),
                          "hiddens_kwargs": {"use_batch_norm": False}},
            classifier_kwargs=classifier_kwargs, schedule="constant", initial_learning_rate=.001,
        )

    @staticmethod
    def arguments(root: Path, template: Path, generator: object) -> dict:
        """Build the same two-task protocol for uninterrupted and resumed models.

        Args:
            root (Path): Destination for task checkpoints.
            template (Path): Saved independent classifier architecture.
            generator (object): Conditional VAE instance to train and recover.

        Returns:
            options (dict): Two tasks of two classes, one epoch each, with actual replay and strict task checkpoints.
        """
        return dict(
            class_num=4, class_order=[0, 1, 2, 3], task_groups=[[0, 1], [2, 3]],
            load_dataset_fn=image_loader,
            load_dataset_fn_kwargs={"preprocess": "standardize", "onehot_labels": True},
            tuned_model_path=str(template), generative_model=generator,
            generative_model_kwargs={"train_num": -1, "samples_per_class": 1},
            compile_args={"loss": "categorical_crossentropy", "metrics": ["accuracy"]},
            epochs=1, batch_size=4, plot_results=False, verbose=0, seed=13,
            return_features=False, save_task_checkpoints=True, checkpoint_dir=str(root),
        )

    def assert_same_run(self, expected: dict, actual: dict) -> None:
        """Compare weights, optimizer slots, task seeds, metadata and metric histories.

        Args:
            expected (dict): Uninterrupted detailed learner result.
            actual (dict): Resumed detailed learner result.

        Returns:
            result (None): Verify exact model/optimizer state, equal-NaN accuracy matrices, histories, task seeds, and VAE metadata.

        Raises:
            AssertionError: If any paired state or metric differs.
        """
        for role in ("model", "generative_model"):
            first, second = expected[role], actual[role]
            self.assertEqual(len(first.weights), len(second.weights))
            for left, right in zip(first.get_weights(), second.get_weights()):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(first.optimizer.variables, second.optimizer.variables):
                np.testing.assert_array_equal(left.numpy(), right.numpy())
            self.assertEqual(len(first.optimizer.variables), len(second.optimizer.variables))
        for key in ("ordinary_accuracy_matrix", "validation_accuracy_matrix"):
            np.testing.assert_allclose(expected[key], actual[key], rtol=0., atol=0., equal_nan=True)
        for key in ("histories", "generative_histories", "task_seeds", "classifier_evaluations", "generative_evaluations"):
            self.assertEqual(fingerprint_state(expected[key]), fingerprint_state(actual[key]))
        for name in ("seed", "reparameterization_seed", "seen_classes"):
            self.assertEqual(getattr(expected["generative_model"], name), getattr(actual["generative_model"], name))

    def test_automatic_vae_preprocessing_uses_actual_target_ranges(self) -> None:
        """Omitted and explicit None follow activation for aliases and both kwargs forms.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        labels = np.tile(np.asarray([0, 1], dtype="uint8"), 4)
        values = np.arange(8, dtype="uint8") * 32
        images = np.broadcast_to(values[:, None, None], (8, 28, 28)).copy()
        raw = ((images, labels), (images.copy(), labels.copy()))
        for family in ("vae", "variational_autoencoder", "vae_classifier"):
            for spelling in ("model_kwargs", "kwargs"):
                for activation, scaling in (("tanh", "standardize"), ("sigmoid", "min-max"), ("linear", "normalize"), (None, "normalize")):
                    for automatic in ({}, {"preprocess": None}):
                        with self.subTest(family=family, spelling=spelling, activation=activation, automatic=automatic):
                            options = dict(model_name=family, use_valset=False, batch_size=8,
                                           shuffle_buffer=0, **{spelling: {"last_activation": activation}}, **automatic)
                            with patch("tensorflow.keras.datasets.mnist.load_data", return_value=raw):
                                dataset, _ = get_datasets(**options)
                            batch = next(iter(dataset))[0].numpy()
                            self.assertEqual(_resolve_dataset_options(None, options)["preprocess"], scaling)
                            self.assertTrue(np.isfinite(batch).all())
                            # Sigmoid and tanh targets stay in their compatible training ranges.
                            if scaling == "min-max":
                                self.assertEqual((float(batch.min()), float(batch.max())), (0., 1.))
                            # Tanh scaling uses both endpoints of the compatible interval.
                            elif scaling == "standardize":
                                self.assertEqual((float(batch.min()), float(batch.max())), (-1., 1.))
                            # Linear decoders retain mean/std normalization without range clipping.
                            else:
                                self.assertAlmostEqual(float(batch.mean()), 0., places=6)
                                self.assertAlmostEqual(float(batch.std()), 1., places=6)
        for explicit in ("fixed-standardize", "fixed-min-max", ""):
            options = _resolve_dataset_options(None, {"model_name": "vae", "preprocess": explicit,
                "model_kwargs": {"last_activation": "sigmoid"}})
            self.assertEqual(options["preprocess"], explicit)

    def test_vae_dnn_interruption_resume_matches_full_state(self) -> None:
        """A failed second generator fit resumes from task zero without losing any state.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        from autoencoder import VariationalAutoencoder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = self.bundle()
            template = root / "template.h5"
            initial["classifier"].save(template)
            full = _run_continual_tasks(**self.arguments(root / "full", template, initial["generative_model"]))
            fresh = self.bundle()["generative_model"]
            options = self.arguments(root / "interrupted", template, fresh)
            original_train = VariationalAutoencoder.train
            fit_count = 0

            def interrupt(model: object, *args: object, **kwargs: object) -> object:
                """Inject a failure outside the experiment callback identity.

                Args:
                    model (object): VAE whose train call is intercepted.
                    *args (object): Positional training arguments.
                    **kwargs (object): Keyword training options.

                Returns:
                    history (object): Actual training history on the first and later noninterrupted calls.

                Raises:
                    RuntimeError: On the second VAE training call, before generator training can commit.
                """
                nonlocal fit_count
                fit_count += 1
                # Fail after task two's classifier fit, before its generator can commit.
                if fit_count == 2:
                    raise RuntimeError("external interruption")
                return original_train(model, *args, **kwargs)

            with patch.object(VariationalAutoencoder, "train", new=interrupt):
                with self.assertRaisesRegex(RuntimeError, "external interruption"):
                    _run_continual_tasks(**options)
            options["generative_model"] = self.bundle()["generative_model"]
            resumed = _run_continual_tasks(**options, resume_from=str(root / "interrupted"))
            self.assert_same_run(full, resumed)
            before = load_task_checkpoint(root / "full")
            after = load_task_checkpoint(root / "interrupted")
            self.assertEqual(fingerprint_state(before.rng_state), fingerprint_state(after.rng_state))
            self.assertEqual(before.next_task_index, after.next_task_index)
            self.assertEqual(before.experiment_state["vae_task_state"], after.experiment_state["vae_task_state"])
            original = load_task_checkpoint(root / "full" / "task-0000")
            state = copy.deepcopy(original.experiment_state)
            state.update(class_order=original.class_order, task_groups=original.task_groups)
            state["vae_task_state"]["seed"] += 1
            save_task_checkpoint(root / "invalid_metadata", 0, state, fingerprint=fingerprint_state(state["run_descriptor"]))
            options = self.arguments(root / "never_used", template, self.bundle()["generative_model"])
            with self.assertRaisesRegex(ValueError, "VAE task seed metadata"):
                _run_continual_tasks(**options, resume_from=str(root / "invalid_metadata"))

    def test_vae_cnn_real_generated_replay_and_reload(self) -> None:
        """Both task phases and checkpoint recovery preserve independent image/flat views.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = self.bundle("cnn")
            template = root / "cnn.h5"
            initial["classifier"].save(template)
            full = _run_continual_tasks(**self.arguments(root / "full", template, initial["generative_model"]))
            self.assertEqual(full["model"].input_shape, (None, 4, 4, 1))
            self.assertGreater(full["task_resource_metrics"][1]["replay"]["selected_count"], 0)
            for task in full["task_resource_metrics"]:
                self.assertGreater(task["optimizer_updates"]["classifier_optimizer"], 0)
                self.assertGreater(task["optimizer_updates"]["replay_optimizer"], 0)
            options = self.arguments(root / "resumed", template, self.bundle("cnn")["generative_model"])
            resumed = _run_continual_tasks(**options, resume_from=str(root / "full" / "task-0000"))
            self.assert_same_run(full, resumed)
            images, labels, *_ = image_loader([0, 1, 2, 3], onehot_labels=True)
            predictions = np.argmax(resumed["model"](images, training=False).numpy(), axis=1)
            targets = labels.argmax(axis=1)
            independently_scored = [float(np.mean(predictions[targets // 2 == task] == targets[targets // 2 == task])) for task in range(2)]
            np.testing.assert_array_equal(independently_scored, resumed["ordinary_accuracy_matrix"][-1])

    def test_public_direct_and_config_vae_contracts_match(self) -> None:
        """Actual public two-task runs share inferred labels, seed, scaling and reports.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        labels = np.repeat(np.arange(4, dtype="uint8"), 8)
        images = np.broadcast_to((labels * 60)[:, None, None], (32, 28, 28)).copy()
        raw = ((images, labels), (images.copy(), labels.copy()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mechanism = {"latent_dim": 2, "hiddens_dims": (4,), "last_activation": "sigmoid",
                         "hiddens_kwargs": {"use_batch_norm": False}}
            continual = {"class_num": 4, "task_groups": [[0, 1], [2, 3]], "seed": 23,
                         "plot_results": False,
                         "generative_model_kwargs": {"train_num": -1, "samples_per_class": 1}}
            reporting = {"show_history_plot": False, "show_final_images": False,
                         "save_history_plot": False, "save_final_images": False, "save_final_gifs": False,
                         "save_csv": True, "run_trainset_eval": False, "run_valset_eval": True}
            direct = dict(task="continual", model_name="vae", model_kwargs=mechanism,
                          classifier_name="dnn", schedule="constant", initial_learning_rate=.001,
                          class_num=4, task_groups=[[0, 1], [2, 3]],
                          epochs=1, batch_size=4, shuffle_buffer=32, validation_ratio=.25, seed=13,
                          results_path=str(root / "direct"), report_every_epoch=False, show_images=True,
                          save_weights=False, save_gifs=False, verbose=0,
                          continually_learn_kwargs={key: value for key, value in continual.items() if key not in {"class_num", "task_groups"}},
                          **reporting)
            config = Config(model={"name": "vae", "kwargs": mechanism, "classifier_name": "dnn"},
                dataset={"name": "mnist", "batch_size": 4, "shuffle_buffer": 32, "validation_ratio": .25},
                optimizer={"schedule": "constant", "initial_learning_rate": .001},
                training={"task": "continual", "epochs": 1, "seed": 13, "results_path": str(root / "typed"),
                          "report_every_epoch": False, "show_images": True, "save_weights": False,
                          "save_gifs": False, "verbose": 0},
                continually_learn=continual, reporting=reporting)
            with patch("tensorflow.keras.datasets.mnist.load_data", return_value=raw):
                first = main(**direct)
                second = main(config)
                cnn = main(**{**direct, "classifier_name": "cnn",
                    "classifier_kwargs": {"architecture_kwargs": {"conv_filters": (2,), "conv_depths": (1,)}},
                    "results_path": str(root / "public_cnn")})
                null_options = {**direct, "results_path": str(root / "nested_unset"),
                    "continually_learn_kwargs": {**direct["continually_learn_kwargs"],
                        "class_num": None, "class_order": None, "task_groups": None},
                    "class_order": [0, 1, 2, 3]}
                resolved = _resolve_dataset_options(None, null_options)
                self.assertEqual(resolved["continual_kwargs"]["class_num"], 4)
                self.assertEqual(resolved["continual_kwargs"]["class_order"], [0, 1, 2, 3])
                self.assertEqual(resolved["continual_kwargs"]["task_groups"], [[0, 1], [2, 3]])
                loader, validation = get_datasets(**null_options)
                self.assertTrue(callable(loader))
                self.assertIsNone(validation)
                initial = get_model(**null_options)
                self.assertEqual(initial["generative_model"].class_num, 4)
                null_nested = main(**null_options)
            self.assertTrue(config.dataset.onehot_labels)
            self.assertEqual(config.dataset.preprocess, "min-max")
            for result in (first, second, cnn, null_nested):
                details = result["model"]["continual_details"]
                self.assertEqual(details["run_descriptor"]["data"]["loader_kwargs"]["onehot_labels"], True)
                self.assertEqual(details["run_descriptor"]["data"]["loader_kwargs"]["preprocess"], "min-max")
                self.assertEqual(details["run_descriptor"]["runtime"]["seed"], 23)
                self.assertEqual(details["next_task_index"], 2)
                self.assertEqual(details["class_order"], [0, 1, 2, 3])
                self.assertEqual(details["task_classes"], [[0, 1], [2, 3]])
                for artifact in ("accuracy_matrices.csv", "summary.csv", "epoch_metrics.csv"):
                    self.assertTrue((Path(result["results_path"]) / artifact).is_file())
            self.assertEqual(cnn["model"]["classifier"].input_shape, (None, 28, 28, 1))
            np.testing.assert_allclose(first["model"]["continual_details"]["accuracy_matrix"],
                                       second["model"]["continual_details"]["accuracy_matrix"], equal_nan=True)
            with self.assertRaisesRegex(ValueError, "Conflicting direct continual schedule"):
                _resolve_dataset_options(None, {"task": "continual", "class_num": 4,
                    "continually_learn_kwargs": {"class_num": 6}})

    def test_mutable_callback_declarations_are_frozen_before_task_commits(self) -> None:
        """Changed nested policy cannot bypass a three-task run's commit identity check.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.h5"
            integration_fixtures.ContinualIntegrationTests._template(template)
            for kind in ("callback", "configured_schedule"):
                with self.subTest(kind=kind):
                    policy = AliasedRateCallback() if kind == "callback" else AliasedRateSchedule()
                    callbacks = [policy] if kind == "callback" else [tf.keras.callbacks.LearningRateScheduler(policy)]
                    frozen = callback_recovery_descriptor(callbacks, strict=True)
                    frozen_hash = fingerprint_state(frozen)
                    checkpoint_dir = root / kind
                    fit_calls = []
                    original_fit = tf.keras.Model.fit

                    def fit_then_change_policy(model: object, *args: object, **kwargs: object) -> object:
                        """Inject an external policy change after a real task fit has completed.

                        Args:
                            model (object): Model passed to the real Keras fit method.
                            *args (object): Positional fit arguments.
                            **kwargs (object): Keyword fit options.

                        Returns:
                            history (object): Actual fit history; the nested callback policy is changed to .2 before returning.
                        """
                        fit_calls.append(model)
                        history = original_fit(model, *args, **kwargs)
                        policy.config["rate"]["value"] = .2
                        return history

                    with patch.object(tf.keras.Model, "fit", new=fit_then_change_policy):
                        with self.assertRaisesRegex(ValueError, "Callback behavior changed"):
                            _run_continual_tasks(class_num=3,
                                load_dataset_fn=integration_fixtures.ContinualIntegrationTests._loader,
                                tuned_model_path=str(template), compile_args={"optimizer": tf.keras.optimizers.Adam(.01),
                                    "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
                                batch_size=4, epochs=1, use_buffer=True,
                                buffer_kwargs={"maxlen": 8, "sample_num": 2, "insert_num": 2},
                                callbacks_list=callbacks, plot_results=False, verbose=0, seed=73,
                                checkpoint_dir=str(checkpoint_dir), save_task_checkpoints=True)
                    self.assertEqual(len(fit_calls), 1)
                    self.assertFalse(list(checkpoint_dir.glob("task-*")))
                    self.assertEqual(fingerprint_state(frozen), frozen_hash)
                    self.assertNotEqual(callback_recovery_descriptor(callbacks, strict=True), frozen)

    def test_callback_policy_and_persistent_state_are_authenticated(self) -> None:
        """Same schedule resumes exactly; changed closures fail before the first fit.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.h5"
            integration_fixtures.ContinualIntegrationTests._template(template)

            def options(path: Path, rate: float) -> dict:
                """Declare a pure closure schedule alongside a stateful custom callback.

                Args:
                    path (Path): Checkpoint destination for this run.
                    rate (float): Constant closure learning rate that enters callback identity.

                Returns:
                    options (dict): Seeded three-task replay-buffer run with closure and persistent callback policies.
                """
                return dict(class_num=3, load_dataset_fn=integration_fixtures.ContinualIntegrationTests._loader,
                    tuned_model_path=str(template), compile_args={"optimizer": tf.keras.optimizers.Adam(.01),
                    "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
                    batch_size=4, epochs=1, use_buffer=True,
                    buffer_kwargs={"maxlen": 8, "sample_num": 2, "insert_num": 2},
                    callbacks_list=[tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: rate), PersistentRateCallback()],
                    plot_results=False, verbose=0, seed=73, checkpoint_dir=str(path), save_task_checkpoints=True)

            full = _run_continual_tasks(**options(root / "full", .01))
            resumed_options = options(root / "same", .01)
            resumed = _run_continual_tasks(**resumed_options, resume_from=str(root / "full" / "task-0000"))
            for left, right in zip(full["model"].get_weights(), resumed["model"].get_weights()):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(resumed_options["callbacks_list"][-1].fit_count, 3)
            with patch.object(tf.keras.Model, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    _run_continual_tasks(**options(root / "changed", .2), resume_from=str(root / "full" / "task-0000"))
                fit.assert_not_called()
            opaque = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: None)
            with self.assertRaisesRegex(ValueError, "opaque callback"):
                callback_recovery_descriptor([opaque], strict=True)
            self.assertEqual(len(callback_recovery_descriptor([opaque], strict=False)), 1)

    def test_invalid_future_checkpoint_requires_fresh_root_before_fit(self) -> None:
        """Incomplete and corrupted evidence survives fallback followed by a new commit.

        Returns:
            result (None): The stated assertions or fixture reset complete; no experiment result is returned.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = self.bundle()
            template = root / "template.h5"
            initial["classifier"].save(template)
            full = _run_continual_tasks(**self.arguments(root / "full", template, initial["generative_model"]))
            for kind in ("incomplete", "corrupt_committed"):
                damaged_root = root / kind
                shutil.copytree(root / "full", damaged_root)
                target = damaged_root / "task-0001"
                # Both states preserve the entire failed slot; only the failure trigger differs.
                if kind == "incomplete":
                    (target / "COMMITTED").unlink()
                # A once-committed payload whose checksum no longer matches is preserved too.
                else:
                    (target / "state.json").write_text("{}", encoding="utf-8")
                retained = {item.relative_to(damaged_root).as_posix(): item.read_bytes()
                            for item in damaged_root.rglob("*") if item.is_file()}
                self.assertEqual(load_task_checkpoint(damaged_root).next_task_index, 1)
                options = self.arguments(damaged_root, template, self.bundle()["generative_model"])
                with patch.object(tf.keras.Model, "fit") as fit:
                    with self.assertRaisesRegex(FileExistsError, "fresh checkpoint_dir"):
                        _run_continual_tasks(**options, resume_from=str(damaged_root))
                    fit.assert_not_called()
                options = self.arguments(root / (kind + "_fresh"), template, self.bundle()["generative_model"])
                resumed = _run_continual_tasks(**options, resume_from=str(damaged_root))
                self.assert_same_run(full, resumed)
                self.assertEqual(load_task_checkpoint(options["checkpoint_dir"]).next_task_index, 2)
                self.assertEqual(retained, {item.relative_to(damaged_root).as_posix(): item.read_bytes()
                                           for item in damaged_root.rglob("*") if item.is_file()})

    def test_depth_zero_growth_preflight_covers_composite_classifier(self) -> None:
        """Reject classifier growth from zero while allowing denoiser-only requests.

        Returns:
            result (None): Raw and wrapper metadata reject a nonempty classifier
                branch; an empty classifier branch preserves valid denoiser growth.

        Raises:
            AssertionError: If the measured behavior violates a stated invariant.
        """
        from types import SimpleNamespace

        for wrapped in (SimpleNamespace(clf_depth=0), SimpleNamespace(network=SimpleNamespace(clf_depth=0))):
            with self.assertRaisesRegex(ValueError, "clf_depth=0"):
                validate_progressive_classifier_growth(wrapped, {"stage_tasks": ["depth"],
                    "depths": [{"classifier": "vision_transformer_block"}]})
            validate_progressive_classifier_growth(wrapped, {"stage_tasks": "depths_only",
                "depths": [{"network": "vision_transformer_block", "classifier": []}]})


# Direct execution uses the same tests as unittest discovery.
if __name__ == "__main__":
    unittest.main()
