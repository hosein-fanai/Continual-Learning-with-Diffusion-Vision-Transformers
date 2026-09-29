"""Finite semantic-route compatibility and real two-task teacher interaction cases.

The configuration products enumerate implemented categorical controls. The real
runs retain common learner training, generated replay, dense class remapping,
acquisition, consolidation and previous/current teachers on tiny synthetic images.
They test protocol boundaries, not accuracy on a public image benchmark.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gc
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from common.learner import _run_continual_tasks
from common.runtime import configure_runtime
from common.train import train_model
from common.tests.test_continual_combination_matrix import image_cohorts, tiny_network
from diffusion import DiffusionClassifier
from semantic_consolidation.config import RouteSettings, load_route_config, validate_route_config
from semantic_consolidation.controller import RouteController
from semantic_consolidation.model import adapt_model


_ROOT = Path(__file__).resolve().parents[2]
_CONDITIONS = ("baseline", "learned", "random", "no_consolidation", 
               "feature_distillation", "unmodulated_feature_distillation", 
               "extra_joint", "time_matched_joint")


class SemanticCombinationMatrixTests(unittest.TestCase):
    """Enumerate semantic protocol constraints and genuine task-boundary effects."""

    def tearDown(self) -> None:
        """Release test-only graphs without changing a notebook kernel's state."""

        tf.keras.backend.clear_session()
        gc.collect()

    def test_architecture_wrapper_fit_and_buffer_product(self) -> None:
        """Accept only the declared V1 DiT ordinary-fit/no-buffer route in 84 cases."""

        base = load_route_config(_ROOT / "semantic_consolidation/configs/smoke.yaml")
        families = ("diffusion_transformer", "dit_decoder", "dit_encoder_decoder", "unet", 
                    "dit_classifier", "dit_encoder_decoder_classifier", "unet_classifier")
        for family, wrapper, fit, buffer in itertools.product(
            families, ("diffusion_model", "diffusion_classifier", "diffusion_classifier_v2"), 
            ("fit", "fit_progressively"), (False, True)
        ):
            with self.subTest(family=family, wrapper=wrapper, fit=fit, buffer=buffer):
                config = deepcopy(base)
                config.common.model.name = family
                config.common.model.wrapper_name = wrapper
                config.common.training.fit_method = fit
                config.common.continually_learn.use_buffer = buffer
                valid = (family == "dit_classifier" and wrapper == "diffusion_classifier"
                         and fit == "fit" and not buffer)
                # Only the explicitly supported architecture/fit/replay tuple may execute.
                if valid:
                    validate_route_config(config)
                # Every other tuple would change the declared scientific mechanism.
                else:
                    with self.assertRaises(ValueError):
                        validate_route_config(config)

    def test_treatment_memory_scope_objective_and_replay_product(self) -> None:
        """Check 128 treatment combinations against positive-pair and gate contracts."""

        base = load_route_config(_ROOT / "semantic_consolidation/configs/smoke.yaml")
        for condition, retain, scope, objective, replay in itertools.product(
            _CONDITIONS, (False, True), ("semantic", "backbone"), 
            ("contrastive", "true_class_ce"), (False, True)
        ):
            with self.subTest(condition=condition, retain=retain, scope=scope, 
                              objective=objective, replay=replay):
                config = deepcopy(base)
                platform = condition in ("baseline", "extra_joint", "time_matched_joint")
                valid = (objective != "true_class_ce" or retain) and (platform or not retain or replay)
                config.common.continually_learn.use_generative_replay = replay
                config.common.continually_learn.replay_old_examples = 8 if replay else 0

                def configure() -> None:
                    """Construct and validate this exact categorical semantic treatment."""

                    config.route = replace(config.route, condition=condition, 
                        retain_modulators=retain, consolidation_scope=scope, 
                        acquisition_objective=objective, 
                        extra_joint_seconds=(.001, .001) if condition == "time_matched_joint" else None)
                    validate_route_config(config)

                # Retained semantic gates require old positive pairs; true CE requires all gates.
                if valid:
                    configure()
                # Reject contradictory memory/replay/objective settings before fitting.
                else:
                    with self.assertRaises(ValueError):
                        configure()

    def test_two_task_consolidation_composes_with_previous_and_current_teachers(self) -> None:
        """Run three phase/teacher/budget treatments and audit post-consolidation snapshots."""

        cases = (
            ("learned", "both", "fresh", "replay_only", "fixed_total"), 
            ("feature_distillation", "current", "student", "old_classes", "match_current"), 
            ("unmodulated_feature_distillation", "previous", "fresh", "current_and_replay", "fixed_total")
        )
        for condition, role, initialization, scope, budget in cases:
            with self.subTest(condition=condition, role=role, initialization=initialization, 
                              scope=scope, budget=budget):
                tf.keras.backend.clear_session()
                gc.collect()
                configure_runtime(953, "float32", True)
                dual = role in ("both", "current")
                previous_weight = float(role in ("both", "previous"))
                base = DiffusionClassifier(
                    network=tiny_network("dit", True), use_ema=False, seed=953, 
                    scheduler_name="clipped_cosine", test_steps=2, p_uncond=1., 
                    defer_teacher=True, noise_distil_loss_coef=.1, clf_distil_loss_coef=.1, 
                    clf_loss_coef=1., clf_distil_type="soft", clf_distil_scope=scope, 
                    mask_by_nulls=True, mask_by_t_threshold=False, 
                    previous_teacher_noise_loss_weight=previous_weight, 
                    previous_teacher_clf_loss_weight=previous_weight, 
                    current_teacher_noise_loss_weight=float(dual), 
                    current_teacher_clf_loss_weight=float(dual)
                )
                base.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", 
                             run_eagerly=True, jit_compile=False)
                settings = RouteSettings(condition=condition, acquisition_steps=2, 
                    consolidation_steps=1, batch_size=4, noise_levels=(0, 1), 
                    probe_batches=1, seed=953)
                controller = RouteController(settings)
                model = adapt_model(base, controller)
                self.assertIs(model.network, base.network)
                self.assertIs(model.optimizer, base.optimizer)
                options = dict(class_num=4, class_order=[2, 0, 3, 1], task_size=2, 
                    load_dataset_fn=image_cohorts, 
                    load_dataset_fn_kwargs={"preprocess": "diffusion"}, 
                    generative_model=model, use_generative_model_classifier=True, 
                    generative_model_kwargs={"train_num": -1, "samples_per_class": 2}, 
                    use_generative_replay=True, use_distillation=True, 
                    dual_teacher_distillation=dual, current_teacher_init=initialization, 
                    replay_budget_mode=budget, batch_size=8, epochs=1, 
                    callback_patience=0, plot_results=False, verbose=0, seed=953, 
                    deterministic_ops=True, experiment_phase="development", 
                    show_generated_images=False, show_network_summary=False)
                # Fixed totals expose two rows per old and new class, matching inferred mode.
                if budget == "fixed_total":
                    options.update(replay_current_examples=4, replay_old_examples=4)
                def inspect(*args: object, **kwargs: object) -> dict:
                    """Verify semantic fitting freezes every attached previous/current teacher.

                    Args:
                        *args (object): Actual common training config, model and dataset.
                        **kwargs (object): Unchanged fit and validation controls.

                    Returns:
                        dict: Actual train_model result with all semantic phases completed.
                    """

                    fitted = args[1]
                    teachers = [teacher for teacher in (model.teacher_network, model.current_teacher_network)
                                if teacher is not None] if fitted is model else []
                    before = [teacher.get_weights() for teacher in teachers]
                    trained = train_model(*args, **kwargs)
                    for teacher, weights in zip(teachers, before):
                        self.assertFalse(teacher.trainable)
                        for actual, expected in zip(teacher.get_weights(), weights):
                            np.testing.assert_array_equal(actual, expected)
                    return trained

                with patch("common.train.train_model", side_effect=inspect):
                    result = _run_continual_tasks(**options)
                self.assertEqual(model.network.num_classes, 4)
                self.assertEqual(model.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
                self.assertEqual(result["task_classes"], [[2, 0], [3, 1]])
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(result["ordinary_accuracy_matrix"], [])
                self.assertEqual(len(controller.records), 2)
                for task, record in enumerate(controller.records):
                    self.assertEqual(record["joint_updates"], 1)
                    self.assertEqual(record["acquisition"]["updates"], 2)
                    self.assertEqual(record["consolidation"]["updates"], 1)
                    self.assertEqual(record["total_updates"], 4)
                    self.assertTrue(all(record["invariants"].values()))
                    self.assertEqual(set(map(int, record["pool_class_counts"])), 
                                     set(range(2 * (task + 1))))
                    self.assertEqual(set(record["pool_class_counts"].values()), {2})
                    for phase in ("acquisition", "consolidation"):
                        self.assertTrue(np.isfinite(record[phase]["history"]["loss"]).all())
                self.assertEqual(set(controller.bank.vectors), {0, 1, 2, 3})
                self.assertIsNone(model.current_teacher_network)
                self.assertFalse(model.teacher_network.trainable)
                self.assertEqual(model.teacher_network.num_classes, 4)
                # Common snapshots the fully consolidated student, not the pre-semantic model.
                for actual, expected in zip(model.network.get_weights(), model.teacher_network.get_weights()):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual([row["optimizer_updates"]["replay_optimizer"]
                                  for row in result["task_resource_metrics"]], [1, 1])


# Direct execution uses only synthetic cohorts and preserves import side-effect freedom.
if __name__ == "__main__":
    unittest.main()
