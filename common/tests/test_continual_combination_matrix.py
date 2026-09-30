"""Bounded continual interaction matrix using synthetic, disjoint image cohorts.

Real two-task fits cover every standalone diffusion raw family and supported
plain/V1/V2 wrapper family. Class order, replay provenance, teacher ownership,
phase optimizer counters, and development-only evaluation are assertions. These
small executions establish integration contracts, not benchmark accuracy or the
Cartesian product of every numeric hyperparameter. Categorical tests separately
exhaust the named baseline switches and teacher scope/provenance intersections.
"""

from __future__ import annotations

import gc
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from autoencoder import VariationalAutoencoder
from common.learner import (
    _has_positive_distillation_objective, 
    _resolve_baseline_controls, _run_continual_tasks
)
from common.runtime import configure_runtime
from common.current_task_teacher import make_current_task_teacher
from common.train import train_model
from diffusion import (
    DiTClassifier, DiTDecoder, DiTEncoderDecoder, DiTEncoderDecoderClassifier, 
    DiffusionClassifier, DiffusionClassifierV2, DiffusionModel, 
    DiffusionTransformer, UNet, UNetClassifier
)


_SEED = 953
_ORDER = [2, 0, 3, 1]


def image_cohorts(indices: list[int], **kwargs: object) -> tuple[np.ndarray, ...]:
    """Return two independent float32 4x4x1 rows per original int32 class ID.

    Args:
        indices (list[int]): Original IDs selected by the learner, from 0 through 3.
        **kwargs (object): Loader compatibility options, ignored; coordinates
            already lie in [-1,1].

    Returns:
        tuple[np.ndarray, ...]: Train, validation, test image/label pairs. Pixel
        centers encode original IDs as ID/4-.75, plus .02 for validation and .04
        for test, so split and label remapping mistakes remain observable.
    """

    del kwargs
    labels = np.repeat(np.asarray(indices, dtype="int32"), 2)
    images = np.broadcast_to((labels / 4. - .75)[:, None, None, None], 
                             (len(labels), 4, 4, 1)).astype("float32").copy()
    return (images, labels, images + np.float32(.02), labels.copy(), 
            images + np.float32(.04), labels.copy())


def tiny_network(family: str, classifier: bool = False) -> tf.keras.Model:
    """Construct an independent dynamic-vocabulary native diffusion network.

    Args:
        family (str): unet, dit, encoder_decoder, or standalone decoder.
        classifier (bool): Attach the native classifier; decoder has no such variant.

    Returns:
        tf.keras.Model: Built float32 4x4x1 model with four diffusion timesteps,
        zero initial classes, CFG enabled for dynamic growth, width four, and no
        dropout or batch normalization.
    """

    common = dict(image_size=4, channels=1, timesteps=4, num_classes=None, 
                  use_cfg=True, seed=_SEED)
    # U-Net uses one down/up level instead of transformer patch tokens.
    if family == "unet":
        cls = UNetClassifier if classifier else UNet
        return cls(widths=tuple([4]), block_depth=1, bottleneck_width=4, bottleneck_depth=1, 
                   image_embedding_dim=4, time_embedding_dim=2, label_embedding_dim=2, 
                   use_batch_norm=False, **common)
    common.update(patch_size=2, dim=4, depth=1, mha_num_heads=1, vit_block_mlp_ratio=1.)
    # A standalone decoder uses its own condition and no external feature routes.
    if family == "decoder":
        return DiTDecoder(encoder_output_grid_size=2, encoder_output_dim=4, decoder_separate_cond=True, 
                          shift_inputs=False, use_causal_mask=False, **common)
    # Composite decoder depth must be bounded independently of encoder depth.
    if family == "encoder_decoder":
        common["decoder_kwargs"] = dict(depth=1, mha_num_heads=1, vit_block_mlp_ratio=1.)
    # Native classifier families share the same small feature projection contract.
    if classifier:
        common.update(clf_depth=1, clf_mha_num_heads=1, clf_vit_block_mlp_ratio=1., 
                      classifier_mlp_ratio=1)
    cls = ({False: DiTEncoderDecoder, True: DiTEncoderDecoderClassifier}
           if family == "encoder_decoder" else {False: DiffusionTransformer, True: DiTClassifier})[classifier]
    return cls(**common)


def standalone_classifier(path: Path) -> None:
    """Save a two-output image classifier for the learner's task-head growth.

    Args:
        path (Path): Temporary .keras artifact destination owned by the caller.

    Returns:
        None: Writes a compiled float32 4x4x1 linear softmax classifier.
    """

    inputs = tf.keras.Input((4, 4, 1))
    outputs = tf.keras.layers.Dense(2, activation="softmax")(tf.keras.layers.Flatten()(inputs))
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.save(path)


class ContinualCombinationMatrixTests(unittest.TestCase):
    """Verify categorical contracts and real task transitions without downloads."""

    def tearDown(self) -> None:
        """Release graphs and restore the matrix's explicit float32 precision."""

        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        gc.collect()

    def run_lifecycle(
        self, family: str, wrapper_kind: str, teacher_role: str = "previous", 
        scope: str = "old_classes", loss_kind: str = "soft", budget: str = "fixed_total", 
        initialization: str = "fresh", replay: bool = True
    ) -> None:
        """Fit two real tasks and inspect every student phase and retained teacher.

        Args:
            family (str): Native architecture name accepted by tiny_network.
            wrapper_kind (str): plain, v1, v2, raw_plain, or raw_classifier.
            teacher_role (str): off, previous, current, or both effective KD roles.
                Current/both construct independent new-class experts each task.
            scope (str): Previous classifier scope old_classes, replay_only, or
                current_and_replay; current experts retain their own task support.
            loss_kind (str): hard CE or soft KL classifier distillation.
            budget (str): fixed_total exposes four real/four old rows, legacy
                exposes two replay rows per old class, match_current matches two.
            initialization (str): fresh local current expert or full student copy.
            replay (bool): Generate old rows when True; otherwise retain only new
                real rows and configure the fixed old budget to zero.

        Returns:
            None: Assertions verify mapping, cohort, teacher and optimizer invariants.
        """

        tf.keras.backend.clear_session()
        gc.collect()
        configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
        raw = wrapper_kind.startswith("raw_")
        classifier = wrapper_kind in ("v1", "v2", "raw_classifier")
        network = tiny_network(family, classifier)
        enabled = teacher_role != "off"
        dual = teacher_role in ("current", "both")
        previous_weight = float(teacher_role in ("previous", "both"))
        current_weight = float(dual)
        primary = None
        # Raw inputs exercise the learner's automatic wrapper construction path.
        if raw:
            supplied = network
        # Explicit wrappers allow the independent role and classifier scope controls.
        else:
            options = dict(network=network, use_ema=False, scheduler_name="clipped_cosine", 
                           test_steps=2, p_uncond=0., 
                           defer_teacher=enabled, noise_distil_loss_coef=.1 if enabled else 0., 
                           previous_teacher_noise_loss_weight=previous_weight, 
                           current_teacher_noise_loss_weight=current_weight, 
                           seed=_SEED)
            # Classifier KD needs row provenance only for the replay_only treatment.
            if classifier:
                options.update(clf_loss_coef=1., clf_distil_loss_coef=.1 if enabled else 0., 
                               clf_distil_type=loss_kind, clf_distil_scope=scope, 
                               clf_distil_temperature=2., mask_by_nulls=False, 
                               mask_by_t_threshold=False, 
                               clf_train_class_input_type="null_class_only" if wrapper_kind == "v2" else "all_classes", 
                               previous_teacher_clf_loss_weight=previous_weight, 
                               current_teacher_clf_loss_weight=current_weight)
            wrapper_cls = {"plain": DiffusionModel, "v1": DiffusionClassifier, 
                           "v2": DiffusionClassifierV2}[wrapper_kind]
            primary = wrapper_cls(**options)
            primary.compile(optimizer=tf.keras.optimizers.SGD(.01), loss="mse", 
                            run_eagerly=True, jit_compile=False)
            supplied = primary
        observations = []

        def inspect_fit(*args: object, **kwargs: object) -> dict:
            """Assert finite raw cohorts and frozen teacher ownership around actual fit.

            Args:
                *args (object): common.train.train_model arguments: config, model,
                    and finite raw batched tf.data.Dataset precede other values.
                **kwargs (object): Unchanged phase fit controls and validation data.

            Returns:
                dict: Actual train_model result after verifying teachers did not change.
            """

            nonlocal primary
            model, dataset = args[1:3]
            # The first diffusion fit is the primary student for raw automatic routes.
            if raw and primary is None and isinstance(model, DiffusionModel):
                primary = model
            # Independent current experts and external classifiers use their own fits.
            if model is not primary:
                return train_model(*args, **kwargs)
            task = model.network.num_classes // 2 - 1
            batches = list(dataset.as_numpy_iterator())
            labels = np.concatenate([row[1] for row in batches]).astype("int32")
            images = np.concatenate([row[0] for row in batches])
            expected_ids = {0, 1} if task == 0 else ({0, 1, 2, 3} if replay else {2, 3})
            self.assertEqual(set(labels.tolist()), expected_ids)
            for label in expected_ids:
                self.assertEqual(int(np.count_nonzero(labels == label)), 2)
            new_rows = labels >= 2 * task
            original = np.asarray(_ORDER, dtype="float32")[labels[new_rows]]
            expected = np.broadcast_to((original / 4. - .75)[:, None, None, None], images[new_rows].shape)
            np.testing.assert_array_equal(images[new_rows], expected)
            # Replay-only fitting includes explicit row origin, never inferred labels.
            if scope == "replay_only" and classifier and not raw:
                provenance = np.concatenate([row[2] for row in batches])
                self.assertEqual(provenance.dtype, np.dtype("bool"))
                np.testing.assert_array_equal(provenance, labels < 2 * task)
            validation = list(kwargs["valset"].as_numpy_iterator())
            val_labels = np.concatenate([row[1] for row in validation]).astype("int32")
            val_images = np.concatenate([row[0] for row in validation])
            expected_val = np.broadcast_to((np.asarray(_ORDER, dtype="float32")[val_labels] / 4. - .75
                                            + np.float32(.02))[:, None, None, None], val_images.shape)
            np.testing.assert_array_equal(val_images, expected_val)
            previous = model.teacher_network
            current = model.current_teacher_network
            self.assertEqual(previous is not None, enabled and task > 0)
            self.assertEqual(current is not None, dual)
            # The previous snapshot contains exactly the last completed vocabulary.
            if previous is not None:
                self.assertEqual(previous.num_classes, 2)
                self.assertEqual(tuple(previous._diffusion_task_class_ids), (0, 1))
            # Fresh experts expose only new output columns; copied experts retain all seen columns.
            if current is not None:
                expected_columns = tuple(range(2 * task, 2 * task + 2)) if initialization == "fresh" else tuple(range(2 * task + 2))
                self.assertEqual(tuple(model.current_teacher_class_ids), expected_columns)
                self.assertEqual(tuple(model.current_teacher_task_class_ids), (2 * task, 2 * task + 1))
            before_student = model.network.get_weights()
            frozen = []
            for teacher in (previous, current):
                # Only attached teachers have independent variables to freeze.
                if teacher is not None:
                    self.assertFalse(teacher.trainable)
                    self.assertTrue({id(v) for v in teacher.weights}.isdisjoint(
                        {id(v) for v in model.network.weights}))
                    frozen.append((teacher, teacher.get_weights()))
            result = train_model(*args, **kwargs)
            for teacher, old_weights in frozen:
                for before, after in zip(old_weights, teacher.get_weights()):
                    np.testing.assert_array_equal(before, after)
            self.assertTrue(any(not np.array_equal(before, after)
                                for before, after in zip(before_student, model.network.get_weights())))
            observations.append((task, kwargs.get("fit_method", "fit")))
            return result

        with tempfile.TemporaryDirectory() as directory:
            options = dict(class_num=4, class_order=_ORDER, task_size=2, 
                           load_dataset_fn=image_cohorts, 
                           load_dataset_fn_kwargs={"preprocess": None}, 
                           generative_model=supplied, use_generative_model_classifier=classifier, 
                           generative_model_compile_args={"optimizer": tf.keras.optimizers.SGD(.01), 
                                                          "loss": "mse", "run_eagerly": True}, 
                           generative_model_kwargs={"train_num": -1, "samples_per_class": 2}, 
                           use_generative_replay=replay, use_distillation=enabled, 
                           dual_teacher_distillation=dual, current_teacher_init=initialization, 
                           replay_budget_mode=budget, batch_size=8, epochs=1, 
                           optimizer_steps_per_epoch=1, callback_patience=0, 
                           plot_results=False, deterministic_ops=True, show_generated_images=False, show_network_summary=False, 
                           experiment_phase="development", verbose=0, 
                           seed=_SEED)
            # Fixed budgets are explicit; the other two modes infer their source count.
            if budget == "fixed_total":
                options.update(replay_current_examples=4, replay_old_examples=4 if replay else 0)
            # Plain denoisers keep a separately trained and expanded task classifier.
            if not classifier:
                template = Path(directory) / "classifier.keras"
                standalone_classifier(template)
                options.update(tuned_model_path=str(template), compile_args={
                    "optimizer": tf.keras.optimizers.SGD(.01), 
                    "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]})
            with patch("common.train.train_model", side_effect=inspect_fit):
                result = _run_continual_tasks(**options)
        primary = result["generative_model"]
        self.assertEqual(primary.network.num_classes, 4)
        self.assertEqual(primary.seen_classes, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(result["task_classes"], [[2, 0], [3, 1]])
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["ordinary_accuracy_matrix"], [])
        matrix = np.asarray(result["validation_accuracy_matrix"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.isnan(matrix[0, 1]))
        self.assertTrue(np.isfinite(matrix[np.tril_indices(2)]).all())
        expected_phases = {"fit_generator", "fit_discriminator"} if wrapper_kind == "v2" else {"fit"}
        self.assertEqual(set(observations), set(itertools.product(range(2), expected_phases)))
        for task, ledger in enumerate(result["task_resource_metrics"]):
            self.assertEqual(ledger["current_examples_available"], 4)
            self.assertEqual(ledger["current_examples_exposed"], 4)
            self.assertEqual(ledger["training_examples_total"], 8 if task and replay else 4)
            self.assertEqual(ledger["replay"]["selected_count"], 4 if task and replay else 0)
            self.assertEqual(ledger["replay"]["source"], "generated" if replay else "none")
            for name, updates in ledger["optimizer_updates"].items():
                self.assertEqual(updates, 1, (name, wrapper_kind))
            # Each current expert sees its real task cohort before replay is appended.
            if dual:
                expert = ledger["current_teacher"]
                self.assertEqual(expert["dataset_class_ids"], [2 * task, 2 * task + 1])
                self.assertEqual(expert["task_class_ids"], [2 * task, 2 * task + 1])
                self.assertEqual(expert["training_examples"], 4)
                self.assertEqual(expert["validation_examples"], 4)
                self.assertTrue(all(value == 1 for value in expert["optimizer_updates"].values()))
        self.assertIsNone(primary.current_teacher_network)
        # The persisted teacher is the completed expanded student, including task two.
        if enabled:
            self.assertEqual(primary.teacher_network.num_classes, 4)
            self.assertEqual(tuple(primary.teacher_network._diffusion_task_class_ids), (0, 1, 2, 3))
            for student, teacher in zip(primary.network.get_weights(), primary.teacher_network.get_weights()):
                np.testing.assert_array_equal(student, teacher)

    def test_plain_wrappers_cover_every_denoiser_family(self) -> None:
        """Exercise real two-task noise KD, generated replay, and external head growth."""

        for family in ("dit", "unet", "encoder_decoder", "decoder"):
            with self.subTest(family=family):
                self.run_lifecycle(family, "plain")

    def test_classifier_wrappers_cover_roles_scopes_and_budgets(self) -> None:
        """Cover all V1/V2 classifier families with bounded complementary treatments."""

        cases = (
            ("dit", "v1", "both", "replay_only", "soft", "fixed_total", "fresh", True), 
            ("unet", "v1", "current", "old_classes", "hard", "match_current", "student", True), 
            ("encoder_decoder", "v1", "off", "current_and_replay", "soft", "legacy", "fresh", True), 
            ("dit", "v2", "previous", "replay_only", "hard", "match_current", "fresh", True), 
            ("unet", "v2", "both", "current_and_replay", "hard", "legacy", "student", True), 
            ("encoder_decoder", "v2", "current", "old_classes", "soft", "fixed_total", "fresh", False)
        )
        for case in cases:
            with self.subTest(case=case):
                self.run_lifecycle(*case)

    def test_current_only_replay_scope_does_not_require_inactive_old_replay(self) -> None:
        """Current experts retain their real-row objective with previous replay KD disabled."""

        for wrapper in ("v1", "v2"):
            with self.subTest(wrapper=wrapper):
                self.run_lifecycle("dit", wrapper, "current", "replay_only", "soft", 
                                   "fixed_total", "fresh", False)

    def test_previous_replay_scope_requires_positive_actual_exposure(self) -> None:
        """Refuse enabled previous replay-only KD with absent or zero generated pools."""

        configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
        model = DiffusionClassifier(network=tiny_network("dit", True), use_ema=False, 
            scheduler_name="clipped_cosine", test_steps=2, defer_teacher=True, noise_distil_loss_coef=.1, 
            clf_distil_loss_coef=.1, clf_distil_scope="replay_only", seed=_SEED)
        model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
        for generated in (False, True):
            with self.subTest(generated=generated):
                loader = Mock(side_effect=AssertionError("invalid KD must fail before loading"))
                with self.assertRaisesRegex(ValueError, "replay_only"):
                    _run_continual_tasks(class_num=4, task_size=2, load_dataset_fn=loader, 
                        generative_model=model, use_generative_model_classifier=True, 
                        use_distillation=True, use_generative_replay=generated, 
                        replay_budget_mode="fixed_total", replay_old_examples=0, 
                        plot_results=False, deterministic_ops=True, show_generated_images=False, show_network_summary=False, 
                        verbose=0, seed=_SEED)
                loader.assert_not_called()

    def test_unused_regularizer_configuration_does_not_count_as_distillation(self) -> None:
        """A positive coefficient and distil mode need an actual classifier token target."""

        configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
        network = tiny_network("dit", True)
        network.clf_cls_token_regularizer_kwargs["train_type"] = "distil"
        model = DiffusionClassifier(network=network, use_ema=False, scheduler_name="clipped_cosine", 
            test_steps=2, defer_teacher=True, ctr_loss_coef=1., 
            noise_distil_loss_coef=0., clf_distil_loss_coef=0., seed=_SEED)
        self.assertEqual(network.clf_cls_token_regularizer_ids, [])
        self.assertFalse(_has_positive_distillation_objective(model))
        model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
        model._check_new_labels(y=np.asarray([0, 1]), verbose=False)
        model.network.clf_cls_token_regularizer_kwargs["train_type"] = "distil"
        with self.assertRaisesRegex(ValueError, "active current-teacher objective"):
            make_current_task_teacher(model, [0, 1], seed=_SEED)

    def test_named_no_kd_baselines_reject_attached_active_teacher_objectives(self) -> None:
        """Reject six undeclared KD treatments while retaining six inert teacher controls."""

        for baseline, role, active in itertools.product(
            ("joint_none", "joint_replay", "diffusion_replay"), ("previous", "current"), (False, True)
        ):
            with self.subTest(baseline=baseline, role=role, active=active), tempfile.TemporaryDirectory() as directory:
                tf.keras.backend.clear_session()
                configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
                classifier = baseline != "diffusion_replay"
                options = dict(network=tiny_network("dit", classifier), use_ema=False, 
                    scheduler_name="clipped_cosine", test_steps=2, defer_teacher=True, noise_distil_loss_coef=.1 if active else 0., 
                    seed=_SEED)
                # Joint baselines add classifier KD; the denoiser baseline has only noise KD.
                if classifier:
                    options["clf_distil_loss_coef"] = .1 if active else 0.
                wrapper = DiffusionClassifier if classifier else DiffusionModel
                model = wrapper(**options)
                model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
                model._check_new_labels(y=np.asarray([0, 1]), verbose=False)
                teacher = model.snapshot_teacher_network("raw")
                # These are independently valid explicit-teacher models before naming a baseline.
                if role == "previous":
                    model.set_teacher_network(teacher)
                # A current teacher is equally capable of activating an undeclared objective.
                else:
                    model.set_current_teacher_network(teacher, class_ids=[0, 1])
                template = Path(directory) / "classifier.keras"
                standalone_classifier(template)
                loader = Mock(side_effect=RuntimeError("reached data boundary"))
                arguments = dict(class_num=4, task_size=2, load_dataset_fn=loader, 
                    generative_model=model, tuned_model_path=str(template), 
                    use_generative_model_classifier=classifier, baseline=baseline, 
                    plot_results=False, deterministic_ops=True, show_generated_images=False, show_network_summary=False, 
                    verbose=0, seed=_SEED)
                # Active teacher objectives contradict the named no-KD treatment.
                if active:
                    with self.assertRaisesRegex(ValueError, "no-KD"):
                        _run_continual_tasks(**arguments)
                    loader.assert_not_called()
                # A teacher with zero effective coefficients is an inert, valid attachment.
                else:
                    with self.assertRaisesRegex(RuntimeError, "reached data boundary"):
                        _run_continual_tasks(**arguments)
                    loader.assert_called_once()
                # An unnamed custom run retains the caller's explicit teacher semantics.
                if active:
                    arguments["baseline"] = None
                    with self.assertRaisesRegex(RuntimeError, "reached data boundary"):
                        _run_continual_tasks(**arguments)
                    loader.assert_called_once()

    def test_raw_networks_keep_automatic_wrapper_lifecycle(self) -> None:
        """Exercise seven raw native families through actual learner auto-wrapping."""

        for classifier, families in ((False, ("dit", "unet", "encoder_decoder", "decoder")), 
                                     (True, ("dit", "unet", "encoder_decoder"))):
            for family in families:
                with self.subTest(family=family, classifier=classifier):
                    self.run_lifecycle(family, "raw_classifier" if classifier else "raw_plain", "off")

    def test_classifier_only_no_replay_baselines_use_their_declared_real_cohorts(self) -> None:
        """Train sequential and cumulative classifiers with exact current/history access."""

        for baseline in ("sequential", "cumulative"):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as directory:
                tf.keras.backend.clear_session()
                configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
                template = Path(directory) / "classifier.keras"
                standalone_classifier(template)
                result = _run_continual_tasks(
                    class_num=4, class_order=_ORDER, task_size=2, load_dataset_fn=image_cohorts, 
                    tuned_model_path=str(template), baseline=baseline, 
                    compile_args={"optimizer": tf.keras.optimizers.SGD(.01), 
                                  "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}, 
                    batch_size=8, epochs=1, callback_patience=0, plot_results=False, 
                    deterministic_ops=True, experiment_phase="development", show_generated_images=False, show_network_summary=False, 
                    verbose=0, seed=_SEED
                )
                self.assertEqual(result["task_classes"], [[2, 0], [3, 1]])
                self.assertIsNone(result["generative_model"])
                self.assertEqual(result["generative_histories"], [None, None])
                ledgers = result["task_resource_metrics"]
                expected = [4, 8] if baseline == "cumulative" else [4, 4]
                self.assertEqual([row["current_examples_exposed"] for row in ledgers], expected)
                self.assertEqual([row["training_examples_total"] for row in ledgers], expected)
                self.assertEqual([row["replay"]["selected_count"] for row in ledgers], [0, 0])
                self.assertTrue(all(row["replay"]["source"] == "none" for row in ledgers))
                self.assertEqual([row["optimizer_updates"]["classifier_optimizer"] for row in ledgers], [1, 1])
                self.assertFalse(result["test_evaluated"])
                self.assertEqual(result["ordinary_accuracy_matrix"], [])

    def test_buffer_strategies_preserve_real_replay_across_unequal_task_growth(self) -> None:
        """Train FIFO/reservoir/balanced buffers across 2/1/1-class task boundaries."""

        for strategy in ("fifo", "reservoir", "class_balanced"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                tf.keras.backend.clear_session()
                gc.collect()
                configure_runtime(dtype_policy="float32", deterministic_ops=True, seed=_SEED)
                template = Path(directory) / "classifier.keras"
                standalone_classifier(template)
                fitted = []

                def inspect(*args: object, **kwargs: object) -> dict:
                    """Check every buffered pixel against its remapped original real class.

                    Args:
                        *args (object): Unchanged config/model/dataset training arguments.
                        **kwargs (object): Unchanged phase fit and validation options.

                    Returns:
                        dict: Actual trained classifier result after exact cohort checks.
                    """

                    width = args[1].output_shape[-1]
                    rows = list(args[2].as_numpy_iterator())
                    images = np.concatenate([row[0] for row in rows])
                    labels = np.concatenate([row[1] for row in rows]).astype("int32")
                    expected = np.broadcast_to((np.asarray(_ORDER, dtype="float32")[labels] / 4. - .75)
                                               [:, None, None, None], images.shape)
                    np.testing.assert_array_equal(images, expected)
                    self.assertTrue(set(labels).issubset(set(range(width))))
                    self.assertIn(width - 1, labels)
                    fitted.append(width)
                    return train_model(*args, **kwargs)

                with patch("common.train.train_model", side_effect=inspect):
                    result = _run_continual_tasks(
                        class_num=4, task_groups=[[2, 0], [3], [1]], load_dataset_fn=image_cohorts, 
                        load_dataset_fn_kwargs={"preprocess": None}, 
                        tuned_model_path=str(template), compile_args={
                            "optimizer": tf.keras.optimizers.SGD(.01), 
                            "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}, 
                        use_buffer=True, buffer_kwargs={"strategy": strategy, "maxlen": 8, 
                                                        "insert_num": 4, "sample_num": 4}, 
                        use_generative_replay=False, replay_budget_mode="fixed_total", 
                        replay_old_examples=4, replay_current_examples=None, 
                        batch_size=8, epochs=1, callback_patience=0, 
                        plot_results=False, deterministic_ops=True, experiment_phase="development", show_generated_images=False, 
                        show_network_summary=False, verbose=0, 
                        seed=_SEED
                    )
                self.assertEqual(fitted, [2, 3, 4])
                self.assertEqual(result["task_classes"], [[2, 0], [3], [1]])
                ledgers = result["task_resource_metrics"]
                self.assertEqual([row["current_examples_exposed"] for row in ledgers], [4, 2, 2])
                self.assertEqual([row["training_examples_total"] for row in ledgers], [4, 6, 6])
                self.assertEqual([row["replay"]["selected_count"] for row in ledgers], [0, 4, 4])
                self.assertTrue(all(row["replay"]["source"] == "buffer" for row in ledgers))
                self.assertEqual([row["optimizer_updates"]["classifier_optimizer"] for row in ledgers], [1, 1, 1])
                self.assertEqual(result["ordinary_accuracy_matrix"], [])
                matrix = np.asarray(result["validation_accuracy_matrix"])
                self.assertTrue(np.isfinite(matrix[np.tril_indices(3)]).all())

    def test_teacher_scope_masks_exhaust_role_provenance_intersections(self) -> None:
        """Check 192 exact masks against independent set membership and Boolean logic."""

        model = object.__new__(DiffusionClassifier)
        classes = tf.constant([0, 1, 2, 3], tf.int32)
        support = {"previous": tf.constant([0, 1]), "current": tf.constant([2, 3])}
        for dual_scope, scope, role, bits in itertools.product(
            ("task", "all"), ("old_classes", "replay_only", "current_and_replay"), 
            ("previous", "current"), itertools.product((False, True), repeat=4)
        ):
            with self.subTest(dual=dual_scope, scope=scope, role=role, provenance=bits):
                object.__setattr__(model, "dual_teacher_scope", dual_scope)
                expected = np.ones(4, dtype=bool)
                # Previous old_classes remains class-restricted even in all-row mode.
                if dual_scope == "task" or role == "previous" and scope == "old_classes":
                    expected = np.isin(np.arange(4), support[role].numpy())
                # Only previous-role replay scope intersects the explicit origin bits.
                if role == "previous" and scope == "replay_only":
                    expected &= np.asarray(bits)
                actual = model._classifier_teacher_mask(classes, tf.constant(bits), support[role], 
                                                        tf.constant(4), scope, role)
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_baseline_switches_override_all_boolean_input_combinations(self) -> None:
        """Exhaust 320 named baseline/input combinations without constructing models."""

        classifier = object.__new__(DiffusionClassifier)
        generator = object.__new__(DiffusionModel)
        vae = object.__new__(VariationalAutoencoder)
        cases = {
            "sequential": (None, True, False, False, False, False), 
            "cumulative": (None, False, False, False, False, False), 
            "reservoir_er": (None, True, True, False, False, False), 
            "diffusion_replay": (generator, True, False, True, False, False), 
            "vae_replay": (vae, True, False, True, False, False), 
            "lwf": (classifier, True, False, False, True, True), 
            "joint_none": (classifier, True, False, False, True, False), 
            "joint_replay": (classifier, True, False, True, True, False), 
            "joint_kd": (classifier, True, False, False, True, True), 
            "joint_both": (classifier, True, False, True, True, True)
        }
        for baseline, (model, *expected) in cases.items():
            for flags in itertools.product((False, True), repeat=5):
                with self.subTest(baseline=baseline, flags=flags):
                    buffer, remove, replay, attached, distilled = flags
                    original = {"strategy": "fifo", "maxlen": 4}
                    actual = _resolve_baseline_controls(baseline, model, buffer, original, 
                                                        remove, replay, attached, distilled)
                    self.assertEqual(actual[1:6], tuple(expected))
                    self.assertEqual(original, {"strategy": "fifo", "maxlen": 4})
                    self.assertEqual(actual[-1]["strategy"], "reservoir" if baseline == "reservoir_er" else "fifo")


# Keep imports side-effect free; direct execution selects this bounded matrix.
if __name__ == "__main__":
    unittest.main()
