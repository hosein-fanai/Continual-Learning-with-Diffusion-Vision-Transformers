"""Real Keras epoch-pruning dispatch and TensorBoard cleanup without model updates."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import tensorflow as tf

from common.callbacks.hpo_guard import TrainingDiverged
from common.config import Config, load_config
from common.hpo_pruning import TrialPerformancePruned, create_exchange
from common.train import _fit_with_callback_cleanup, main as train_main, train_model


class _ScriptedNoiseModel(tf.keras.Model):
    """Expose fixed noise-loss curves through real fit callbacks without updating weights."""

    def __init__(self, validation_values: list[float]) -> None:
        """Initialize independent training/validation counters.

        Args:
            validation_values: One fixed scalar per validation epoch.
        """

        super().__init__()
        self.validation_values = validation_values
        self.train_calls = 0
        self.validation_calls = 0
        self.noise_tracker = tf.keras.metrics.Mean(name="noise_loss")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Return the metric reset by Keras before every training and validation epoch."""

        return [self.noise_tracker]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return supplied tensors unchanged.

        Args:
            inputs: Arbitrary synthetic batch satisfying Keras' model interface.

        Returns:
            tf.Tensor: The identical input without trainable operations.
        """

        return inputs

    def train_step(self, data: object) -> dict[str, tf.Tensor]:
        """Record one finite training observation without applying an optimizer.

        Args:
            data: Ignored synthetic batch.

        Returns:
            dict: Constant noise-loss aggregate.
        """

        del data
        self.train_calls += 1
        self.noise_tracker.update_state(1.)
        return {"noise_loss": self.noise_tracker.result()}

    def test_step(self, data: object) -> dict[str, tf.Tensor]:
        """Report the next fixed validation score.

        Args:
            data: Ignored synthetic validation batch.

        Returns:
            dict: Noise loss that Keras prefixes with val_ in epoch logs.
        """

        del data
        value = self.validation_values[self.validation_calls]
        self.validation_calls += 1
        self.noise_tracker.update_state(value)
        return {"noise_loss": self.noise_tracker.result()}


class PruningTrainingTests(TestCase):
    """Verify real callback ordering, writer finalization, and early exit before reporting."""

    def setUp(self) -> None:
        """Allocate isolated artifacts and arrange deterministic Keras cleanup."""

        temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(tf.keras.backend.clear_session)
        self.root = Path(temporary.name)

    def fixtures(self, exchange: bool = True) -> tuple[Config, _ScriptedNoiseModel, tf.data.Dataset]:
        """Build one-batch epochs with all heavyweight model/data factories bypassed.

        Args:
            exchange: Include runtime HPO reporting transport when true.

        Returns:
            tuple: Typed training config, scripted model, and single-batch dataset.
        """

        hpo = {"prune_nonfinite_losses": True}
        # Only search workers carry a live pruning exchange.
        if exchange:
            hpo["pruning_exchange"] = create_exchange(self.root / "exchange", "val_noise_loss", 7)
        config = Config(
            model={"name": "diffusion_transformer"}, 
            hpo=hpo, 
            training={
                "task": "generation", "epochs": 3, "verbose": 0, 
                "results_path": str(self.root), "save_weights": False, 
                "tensorboard": True, "tensorboard_path": str(self.root / "tensorboard"), 
                "project_tag": "tiny", "patience": 0, "reduce_lr_patience": 0, 
                "report_every_epoch": False, "show_images": False, "save_gifs": False
            }
        )
        model = _ScriptedNoiseModel([0.9, 0.8, 0.7])
        model.compile(optimizer="sgd", run_eagerly=True)
        rows = [[1.], [2.]]
        dataset = tf.data.Dataset.from_tensor_slices((rows, rows)).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        return config, model, dataset.with_options(options)

    @staticmethod
    def decision(exchange: dict, step: int, value: float) -> tuple[bool, dict]:
        """Reject the second epoch while preserving the actual transport report schema.

        Args:
            exchange: Runtime exchange identity passed by the callback.
            step: Zero-based epoch index.
            value: Current validation noise loss.

        Returns:
            tuple: Prune flag and immutable report identity used as evidence.
        """

        return step == 1, {
            "protocol_version": 1, "token": exchange["token"], 
            "trial_number": exchange["trial_number"], "monitor": exchange["monitor"], 
            "step": step, "request_token": f"{step:032x}", "value": value
        }

    @staticmethod
    def scalar_values(directory: Path) -> dict[str, list[float]]:
        """Read persisted scalar tensors before callbacks are garbage collected.

        Args:
            directory: Root containing actual TensorBoard event files.

        Returns:
            dict: Ordered scalar observations grouped by tag.
        """

        values: dict[str, list[float]] = {}
        for path in directory.rglob("events.out.tfevents.*"):
            for event in tf.compat.v1.train.summary_iterator(str(path)):
                for value in event.summary.value:
                    # Hyperparameter text and other plugin payloads are not scalar observations.
                    if value.metadata.plugin_data.plugin_name == "scalars":
                        values.setdefault(value.tag, []).append(float(tf.make_ndarray(value.tensor).item()))
        return values

    def test_real_pruning_keeps_epoch_events_and_skips_weights_and_final_evaluation(self) -> None:
        """A coordinator stop propagates through main after two fully logged epochs."""

        config, model, dataset = self.fixtures()
        config.training.save_weights = True
        model.save_weights = Mock()
        tensorboards = []

        class TensorBoardSpy(tf.keras.callbacks.TensorBoard):
            """Track writer cleanup without replacing actual TensorBoard event output."""

            def __init__(self, **kwargs: object) -> None:
                """Register this callback and forward all real TensorBoard settings.

                Args:
                    kwargs: Standard TensorBoard constructor options.
                """

                super().__init__(**kwargs)
                self.close_calls = 0
                tensorboards.append(self)

            def on_train_end(self, logs: dict | None = None) -> None:
                """Count finalization and close real writers.

                Args:
                    logs: Optional terminal metric mapping.
                """

                self.close_calls += 1
                super().on_train_end(logs)

        observer = tf.keras.callbacks.Callback()
        observer.on_epoch_end = Mock()
        observer.on_train_end = Mock()
        with patch("common.train.ImageGenerator") as image, patch(
            "common.train.callbacks.TensorBoard", TensorBoardSpy
        ), patch("common.callbacks.hpo_pruning.report_epoch", side_effect=self.decision) as decisions, patch(
            "common.train.get_datasets", return_value=(dataset, dataset)
        ), patch("common.train.get_model", return_value=model), patch("common.train.report") as report, patch("builtins.print"):
            image.return_value.results_path = str(self.root)
            with self.assertRaises(TrialPerformancePruned) as raised:
                train_main(config, extra_callbacks=[observer])
        self.assertEqual(model.train_calls, 2)
        self.assertEqual(model.validation_calls, 2)
        self.assertEqual(decisions.call_count, 2)
        self.assertEqual([call.args[1] for call in decisions.call_args_list], [0, 1])
        self.assertEqual(observer.on_epoch_end.call_count, 2)
        observer.on_train_end.assert_not_called()
        model.save_weights.assert_not_called()
        report.assert_not_called()
        self.assertNotIn("pruning_exchange", config.hpo)
        for filename in ("input_config.yaml", "config.yaml"):
            self.assertNotIn("pruning_exchange", load_config(self.root / filename).hpo)
        self.assertEqual(len(tensorboards), 1)
        self.assertEqual(tensorboards[0].close_calls, 1)
        evidence = raised.exception.evidence
        self.assertEqual(evidence["epoch"], 1)
        self.assertEqual(evidence["metric"], "val_noise_loss")
        self.assertEqual(len(evidence["partial_history"]), 2)
        self.assertEqual(json.loads(raised.exception.evidence_path.read_text(encoding="utf-8")), evidence)
        values = self.scalar_values(self.root / "tensorboard")["epoch_noise_loss"]
        self.assertTrue(any(abs(value - 0.9) < 1e-6 for value in values))
        self.assertTrue(any(abs(value - 0.8) < 1e-6 for value in values))
        self.assertFalse(any(abs(value - 0.7) < 1e-6 for value in values))

    def test_declined_pruning_retains_full_history_and_normal_completion(self) -> None:
        """A reporting callback does not shorten training when every decision continues."""

        config, model, dataset = self.fixtures()

        def continue_trial(exchange: dict, step: int, value: float) -> tuple[bool, dict]:
            """Keep the valid report but override the synthetic stop decision.

            Args:
                exchange: Current worker exchange identity.
                step: Zero-based epoch coordinate.
                value: Observed validation scalar.

            Returns:
                tuple: False pruning decision with valid report evidence.
            """

            return False, self.decision(exchange, step, value)[1]

        observer = tf.keras.callbacks.Callback()
        observer.on_train_end = Mock()
        with patch("common.train.ImageGenerator") as image, patch(
            "common.callbacks.hpo_pruning.report_epoch", side_effect=continue_trial
        ) as decisions:
            image.return_value.results_path = str(self.root)
            history = train_model(config, model, dataset, valset=dataset, extra_callbacks=[observer])
        self.assertEqual(model.train_calls, 3)
        self.assertEqual(decisions.call_count, 3)
        self.assertEqual(len(history["val_noise_loss"]), 3)
        observer.on_train_end.assert_called_once()

    def test_without_exchange_keeps_existing_training_lifecycle(self) -> None:
        """Ordinary and confirmation fits never install a runtime pruning callback."""

        config, model, dataset = self.fixtures(exchange=False)
        with patch("common.train.ImageGenerator") as image, patch(
            "common.callbacks.hpo_pruning.EpochPruningCallback"
        ) as pruning:
            image.return_value.results_path = str(self.root)
            history = train_model(config, model, dataset, valset=dataset)
        pruning.assert_not_called()
        self.assertEqual(model.train_calls, 3)
        self.assertEqual(len(history["val_noise_loss"]), 3)

    def test_nonfinite_guard_precedes_coordinator_ranking(self) -> None:
        """NaN validation retains the existing divergence outcome without finite comparisons."""

        config, model, dataset = self.fixtures()
        model.validation_values[0] = float("nan")
        with patch("common.train.ImageGenerator") as image, patch(
            "common.callbacks.hpo_pruning.report_epoch"
        ) as decisions:
            image.return_value.results_path = str(self.root)
            with self.assertRaises(TrainingDiverged) as raised:
                train_model(config, model, dataset, valset=dataset)
        decisions.assert_not_called()
        self.assertEqual(raised.exception.evidence["metric"], "val_noise_loss")
        self.assertEqual(model.train_calls, 1)

    def test_pruning_cleanup_preserves_exception_and_deduplicates_writers(self) -> None:
        """Even a writer-close failure cannot mask pruning or invoke successful-fit hooks."""

        error = TrialPerformancePruned({
            "reason": "performance_pruning", "epoch": 1, "metric": "val_noise_loss", 
            "value": 0.8, "partial_history": []
        })
        first = tf.keras.callbacks.TensorBoard(log_dir=str(self.root / "first"))
        second = tf.keras.callbacks.TensorBoard(log_dir=str(self.root / "second"))
        first.on_train_end = Mock(side_effect=RuntimeError("synthetic writer failure"))
        second.on_train_end = Mock()
        stopper = tf.keras.callbacks.EarlyStopping()
        stopper.on_train_end = Mock()
        model = SimpleNamespace(fit=Mock(side_effect=error))
        with self.assertRaises(TrialPerformancePruned) as raised:
            _fit_with_callback_cleanup(
                model, callbacks=[first, stopper], 
                gen_kwargs={"callbacks": [first, stopper]}, 
                clf_kwargs={"callbacks": [first, second, stopper]}
            )
        self.assertIs(raised.exception, error)
        first.on_train_end.assert_called_once_with()
        second.on_train_end.assert_called_once_with()
        stopper.on_train_end.assert_not_called()
        self.assertTrue(any("synthetic writer failure" in note for note in error.__notes__))


# Run the focused integration suite only when this module is invoked directly.
if __name__ == "__main__":
    main()
