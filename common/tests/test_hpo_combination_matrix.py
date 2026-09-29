"""Audit categorical HPO construction, serialization, scoring and tiny real trials.

The matrix covers every declared task/model and dataset pairing, every reachable
categorical edge under recorded parent choices, and crossed classifier-distillation
orchestration factors. It does not enumerate the Cartesian product of every
architecture width, depth, continuous rate or random seed. Synthetic runtime checks
replace downloaded pixels and reduce compute after normal HPO configuration; they
do not estimate benchmark accuracy. Every configuration round trip uses public YAML
serialization, including conditional integer-keyed architecture mappings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna

from common.config import Config, load_config, save_config
from common.hpo import (
    SEARCH_SPACES, _build_trial_config, _objective_values, _study_json_value, run_hpo
)
from common.hpo_profiles import (
    JOINT_CLASSIFIER_PROFILE, JOINT_CLASSIFIER_PROFILE_VERSION, 
    JOINT_CLASSIFIER_SEARCH_SPACE
)


class _MatrixTrial:
    """Record available distributions and choose explicit values or lower endpoints.

    Attributes:
        number (int): Stable synthetic trial identity zero.
        params (dict[str, object]): Values returned by suggestion methods.
        categories (dict[str, tuple[object, ...]]): Categorical distributions visited.
        user_attrs (dict[str, object]): Metadata saved by HPO builders.
    """

    number = 0

    def __init__(self, choices: dict[str, object] | None = None, 
                 upper: bool = False) -> None:
        """Copy categorical choices; upper selects numeric high rather than low.

        Args:
            choices (dict[str, object] | None): Stored parameter names to force;
                omitted names choose the first offered category.
            upper (bool): Select upper numeric endpoints when true.

        Returns:
            None: Initializes independent recording mappings.
        """

        self.choices = dict(choices or {})
        self.upper = upper
        self.params: dict[str, object] = {}
        self.categories: dict[str, tuple[object, ...]] = {}
        self.user_attrs: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        """Select an offered category and record its complete local distribution.

        Args:
            name (str): Exact possibly namespaced Optuna parameter name.
            choices (list[object]): Current nonempty categorical support.

        Returns:
            object: Forced value or first category, also recorded in params.
        """

        value = self.choices.get(name, choices[0])
        assert value in choices, (name, value, choices)
        self.categories[name] = tuple(choices)
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, 
                      **kwargs: object) -> float:
        """Record the selected finite endpoint of a floating distribution.

        Args:
            name (str): Stored parameter name.
            low (float): Inclusive lower endpoint.
            high (float): Inclusive upper endpoint.
            **kwargs (object): Optuna step/log declarations validated by _TrialView.

        Returns:
            float: Low by default, high when upper=True.
        """

        del kwargs
        value = float(high if self.upper else low)
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, 
                    **kwargs: object) -> int:
        """Record an endpoint of an integer distribution on its declared step grid.

        Args:
            name (str): Stored parameter name.
            low (int): Inclusive lower endpoint.
            high (int): Upper endpoint before step-grid adjustment.
            **kwargs (object): Optional step integer and log flag.

        Returns:
            int: Lower endpoint or last allowed stepped upper value.
        """

        step = int(kwargs.get("step", 1))
        value = low + ((high - low) // step) * step if self.upper else low
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value: object) -> None:
        """Record a builder's metadata without adding sampled dimensions.

        Args:
            name (str): Metadata key.
            value (object): JSON-compatible metadata value retained unchanged.

        Returns:
            None: Updates user_attrs.
        """

        self.user_attrs[name] = value


class HpoCombinationMatrixTests(unittest.TestCase):
    """Verify declared categorical domains and interacting orchestration controls."""

    def setUp(self) -> None:
        """Create one temporary YAML destination reused by this test's matrix rows."""

        self.temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC_HPO_MATRIX_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = 0

    def _round_trip(self, config: Config) -> None:
        """Preserve all configuration values through public compact YAML persistence.

        Args:
            config (Config): Constructed HPO configuration with no live teacher.

        Returns:
            None: Checks normalized values after save/load, including tuple/list equivalence.
        """

        path = self.root / "config.yaml"
        save_config(config, path, shorten=True)
        restored = load_config(path)
        self.assertEqual(_study_json_value(asdict(restored)), _study_json_value(asdict(config)))
        self.assertFalse(config.optimizer.clipnorm is not None
                         and config.optimizer.global_clipnorm is not None)
        self.rows += 1

    def _build(self, trial: _MatrixTrial, task: str, model: str, 
               dataset: str = "cifar10", **options: object) -> Config:
        """Construct a bounded configuration without model allocation or data loading.

        Args:
            trial (_MatrixTrial): Deterministic distribution recorder.
            task (str): Declared HPO task.
            model (str): Model key in the selected task.
            dataset (str): Dataset geometry to resolve, default CIFAR10.
            **options (object): Additional builder controls overriding the tiny budget.

        Returns:
            Config: Full typed configuration with a validated two-task schedule for continual.
        """

        settings = {"epochs": 1, "seed": 31, "results_path": self.root}
        # Continual comparisons need at least two explicit multiclass tasks.
        if task == "continual":
            settings.update(class_num=4, task_size=2, class_order=[2, 0, 3, 1], 
                            task_groups=[[2, 0], [3, 1]])
        settings.update(options)
        return _build_trial_config(trial, task, model, dataset, **settings)

    def test_every_declared_task_model_dataset_pair_round_trips(self) -> None:
        """Cover all 96 pairings; four grayscale-pretrained combinations reject explicitly."""

        rejected = 0
        pairs = {(task, model) for task, models in SEARCH_SPACES.items() for model in models}
        self.assertEqual(len(pairs), 24)
        for (task, model), dataset in product(sorted(pairs), ("mnist", "fmnist", "cifar10", "cifar100")):
            with self.subTest(task=task, model=model, dataset=dataset):
                # Xception's RGB-only input contract excludes the two grayscale datasets.
                if model == "pretrained" and dataset in ("mnist", "fmnist"):
                    with self.assertRaisesRegex(ValueError, "three-channel"):
                        self._build(_MatrixTrial(), task, model, dataset)
                    rejected += 1
                    continue
                config = self._build(_MatrixTrial(), task, model, dataset)
                self._round_trip(config)
                self.assertEqual(config.training.task, task)
                self.assertEqual(config.dataset.name, dataset)
                self.assertEqual(config.hpo["study_model"], model)
                # Continual HPO must select validation and retain native boundary recovery.
                if task == "continual":
                    self.assertEqual(config.continually_learn.experiment_phase, "development")
                    self.assertTrue(config.continually_learn.save_task_checkpoints)
                    self.assertEqual(config.continually_learn.task_groups, [[2, 0], [3, 1]])
        self.assertEqual((self.rows, rejected), (92, 4))
        print(f"HPO_MATRIX declared: {self.rows} valid YAML rows, {rejected} required rejections", flush=True)

    def test_reachable_categorical_edges_preserve_parent_choices(self) -> None:
        """Visit every discovered category under its enabling parent choices on 28/32 pixel grids.

        This is categorical branch-edge coverage, not the enormous full architecture
        Cartesian product. Each new distribution value adds a replayable parent-choice
        row, including conditional optimizer, topology, masking and replay settings.
        """

        edge_count = 0
        for task, models in SEARCH_SPACES.items():
            for model, dataset in product(models, ("mnist", "cifar10")):
                # The explicit grayscale rejection is covered by the pairing matrix.
                if model == "pretrained" and dataset == "mnist":
                    continue
                pending = deque([{}])
                scheduled = set()
                while pending:
                    preferences = pending.popleft()
                    trial = _MatrixTrial(preferences)
                    with self.subTest(task=task, model=model, dataset=dataset, choices=preferences):
                        config = self._build(trial, task, model, dataset)
                        self._round_trip(config)
                        self._round_trip(self._build(_MatrixTrial(preferences, upper=True), task, model, dataset))
                    for name, choices in trial.categories.items():
                        for value in choices:
                            key = (name, repr(choices), repr(value))
                            # Schedule each local distribution edge once with its enabling parents.
                            if key not in scheduled:
                                scheduled.add(key)
                                edge_count += 1
                                # The current row already demonstrates its selected category.
                                if trial.params[name] != value:
                                    pending.append({**trial.choices, name: value})
                self.assertTrue(scheduled)
        print(f"HPO_MATRIX categorical: {self.rows} YAML rows, {edge_count} distribution edges", flush=True)

    def test_classifier_distillation_orchestration_cross_product(self) -> None:
        """Cross every valid teacher scope with wrappers, curricula, heads and noise KD.

        All three concrete classifier families use both V1/V2, fit/progressive,
        ordinary/ensemble, hard/soft, raw/EMA and noise-KD on/off combinations.
        Replay strategy determines its mathematically available row-selection scopes.
        """

        families = ("dit_classifier", "dit_encoder_decoder_classifier", "unet_classifier")
        scopes = {"generative_replay": ("old_classes", "replay_only", "current_and_replay"), 
                  "cumulative": ("old_classes", "current_and_replay"), 
                  "new_only": tuple(["current_and_replay"])}
        strategies = [(strategy, scope) for strategy, values in scopes.items() for scope in values]
        factors = product(families, ("diffusion_classifier", "diffusion_classifier_v2"), 
                          ("fit", "fit_progressively"), (False, True), 
                          ("hard", "soft"), ("raw", "ema"), (False, True), strategies, (False, True))
        for family, wrapper, fit, ensemble, kind, snapshot, noise, (strategy, scope), upper in factors:
            choices = {"wrapper_name": wrapper, "continual_strategy_multiclass": strategy, 
                       "clf_distil_scope_" + strategy: scope, "clf_distil_type": kind, 
                       "use_noise_distillation": noise}
            fit_kwargs = {"stage_tasks": "timesteps_only", "stage_epochs": 1, "final_epochs": 1}
            with self.subTest(family=family, wrapper=wrapper, fit=fit, ensemble=ensemble, 
                              kind=kind, snapshot=snapshot, noise=noise, strategy=strategy, scope=scope):
                config = self._build(_MatrixTrial(choices, upper), "continual", family, 
                                     use_distillation=True, use_ensemble_accuracy=ensemble, 
                                     snapshot_network_name=snapshot, fit_method=fit, 
                                     fit_kwargs=fit_kwargs if fit == "fit_progressively" else None)
                self._round_trip(config)
                values = config.model.wrapper_kwargs
                self.assertEqual(values["clf_distil_scope"], scope)
                self.assertEqual(values["clf_distil_type"], kind)
                expected_temperature = (8. if upper else 0.5) if kind == "soft" else 1.
                self.assertEqual(values["clf_distil_temperature"], expected_temperature)
                self.assertEqual(values["noise_distil_loss_coef"] > 0, noise)
                self.assertEqual(config.continually_learn.snapshot_network_name, snapshot)
                self.assertEqual(config.continually_learn.use_generative_replay, strategy == "generative_replay")
                self.assertEqual(config.continually_learn.remove_prev_classes, strategy != "cumulative")
                self.assertEqual(config.continually_learn.train_classifier_separately, 
                                 wrapper == "diffusion_classifier_v2")
                self.assertEqual(config.continually_learn.use_ensemble_accuracy, ensemble)
                self.assertEqual(config.optimizer.schedule, "constant")
                self.assertEqual(sum(values[key] for key in ("clf_acc_coef", "clf_distil_acc_coef", "ctr_acc_coef")), 1.)
        self.assertEqual(self.rows, 2304)
        print(f"HPO_MATRIX distilled orchestration: {self.rows} YAML rows", flush=True)

    def test_named_profile_categories_numeric_endpoints_and_inputs(self) -> None:
        """Cover every profile category, both numeric endpoints and all 12 input combinations."""

        category_rows = [{name: value} for name, values in JOINT_CLASSIFIER_SEARCH_SPACE.items()
                         if isinstance(values, list) for value in values]
        input_rows = [{"clf_train_batch_fraction": fraction, "clf_train_noisy_input_type": noise, 
                       "clf_train_class_input_type": condition}
                      for fraction, noise, condition in product((0., .25, .5), ("noisy", "clean"), 
                                                               ("null_class_only", "all_classes"))]
        for dataset, upper, choices in product(("cifar10", "cifar100"), (False, True), 
                                               category_rows + input_rows):
            with self.subTest(dataset=dataset, upper=upper, choices=choices):
                config = self._build(_MatrixTrial(choices, upper), "joint", "dit_classifier", dataset, 
                                     search_profile=JOINT_CLASSIFIER_PROFILE)
                self._round_trip(config)
                self.assertEqual(config.hpo["profile_version"], JOINT_CLASSIFIER_PROFILE_VERSION)
                self.assertEqual(config.model.wrapper_name, "diffusion_classifier")
                self.assertEqual(config.model.wrapper_kwargs["clf_loss_coef"], 1.)
                self.assertEqual(config.training.fit_kwargs["validation_freq"], [])
                self.assertFalse(config.reporting.evaluate_ensemble_accuracy)
                self.assertEqual(config.optimizer.initial_learning_rate, 1e-3 if upper else 1e-5)
        print(f"HPO_MATRIX named profile: {self.rows} YAML rows", flush=True)

    def test_nondistilled_and_plain_wrapper_modes_round_trip(self) -> None:
        """Cross no-teacher classifier modes and plain-wrapper noise-distillation curricula."""

        classifiers = ("dit_classifier", "dit_encoder_decoder_classifier", "unet_classifier")
        for family, wrapper, fit, ensemble, strategy in product(
                classifiers, ("diffusion_classifier", "diffusion_classifier_v2"), 
                ("fit", "fit_progressively"), (False, True), 
                ("generative_replay", "cumulative", "new_only")):
            trial = _MatrixTrial({"wrapper_name": wrapper, "continual_strategy_multiclass": strategy})
            curriculum = {"stage_tasks": "timesteps_only", "stage_epochs": 1, "final_epochs": 1}
            config = self._build(trial, "continual", family, fit_method=fit, 
                                 use_ensemble_accuracy=ensemble, 
                                 fit_kwargs=curriculum if fit == "fit_progressively" else None)
            self._round_trip(config)
            self.assertFalse(config.continually_learn.use_distillation)
            self.assertEqual(config.model.wrapper_kwargs["clf_distil_acc_coef"], 0.)
            self.assertNotIn("clf_distil_type", trial.params)
        generators = ("diffusion_transformer", "dit_decoder", "dit_encoder_decoder", "unet")
        for family, fit, distilled, snapshot in product(generators, ("fit", "fit_progressively"), 
                                                       (False, True), ("raw", "ema")):
            curriculum = {"stage_tasks": "timesteps_only", "stage_epochs": 1, "final_epochs": 1}
            config = self._build(_MatrixTrial(), "continual", family, fit_method=fit, 
                                 use_distillation=distilled, snapshot_network_name=snapshot, 
                                 fit_kwargs=curriculum if fit == "fit_progressively" else None)
            self._round_trip(config)
            self.assertEqual(config.model.wrapper_kwargs.get("noise_distil_loss_coef", 0.) > 0., distilled)
            self.assertFalse(config.continually_learn.use_generative_model_classifier)
        self.assertEqual(self.rows, 104)
        print(f"HPO_MATRIX nondistilled/plain modes: {self.rows} YAML rows", flush=True)

    def test_unsupported_mode_combinations_fail_before_allocating_studies(self) -> None:
        """Reject incompatible ensemble, teacher-free distillation and nondiffusion curricula."""

        rows = 0
        for task, models in SEARCH_SPACES.items():
            for model in models:
                diffusion = model.startswith(("dit", "diffusion", "unet"))
                classifier = model in ("dit_classifier", "dit_encoder_decoder_classifier", 
                                       "unet_classifier", "diffusion_classifier")
                options = []
                # Only joint/continual diffusion classifiers define timestep-ensemble feedback.
                if not (task in ("joint", "continual") and classifier):
                    options.append({"use_ensemble_accuracy": True})
                # An absent runtime teacher is meaningful only for continual diffusion snapshots.
                if not (task == "continual" and diffusion):
                    options.append({"use_distillation": True})
                # Nondiffusion models cannot consume diffusion timestep/depth curricula.
                if not diffusion:
                    options.append({"fit_method": "fit_progressively", "fit_kwargs": {"stage_tasks": "timesteps_only"}})
                for invalid in options:
                    destination = self.root / f"invalid-{rows}"
                    with self.subTest(task=task, model=model, options=invalid), self.assertRaises(ValueError):
                        run_hpo(task, model, "mnist" if model == "vae" else "cifar10", 
                                epochs=1, n_trials=1, results_path=destination, **invalid)
                    self.assertFalse(destination.exists())
                    rows += 1
        self.assertGreater(rows, 30)
        print(f"HPO_MATRIX invalid modes: {rows} pre-storage rejections", flush=True)

    def test_replay_sampling_override_matches_wrapper_endpoint_contract(self) -> None:
        """Reject one-step/out-of-horizon HPO sampling and preserve both valid endpoints."""

        for steps in (1, 501):
            with self.subTest(invalid_steps=steps), self.assertRaisesRegex(ValueError, "test_steps"):
                self._build(_MatrixTrial(), "continual", "diffusion_transformer", 
                            search_space_overrides={"timesteps": [500], "test_steps": [steps]})
        for steps in (2, 500):
            with self.subTest(valid_steps=steps):
                config = self._build(_MatrixTrial(), "continual", "diffusion_transformer", 
                                     search_space_overrides={"timesteps": [500], "test_steps": [steps]})
                self.assertEqual(config.model.wrapper_kwargs["test_steps"], steps)
                self._round_trip(config)

    def test_tiny_real_trials_cover_distinct_ordinary_and_progressive_paths(self) -> None:
        """Run eight real one-trial studies using synthetic pixels and reduced compute.

        Covers ordinary diffusion and explicit-context decoder generation, V1/V2
        joint fitting, progressive V2, VAE generation/joint fitting and standalone
        classification. Network architecture is reduced after normal search and
        recorded as synthetic software-check metadata; no dataset is downloaded.
        """

        from common.train import main as train_main


        rng = np.random.default_rng(83)
        labels = np.repeat(np.arange(2, dtype=np.uint8), 12)
        images = rng.integers(0, 256, (24, 28, 28), dtype=np.uint8)
        dataset = ((images, labels), (255 - images[::3], labels[::3]))
        routes = (("generation", "diffusion_transformer", None, "fit"), 
                  ("generation", "dit_decoder", None, "fit"), 
                  ("joint", "dit_classifier", "diffusion_classifier", "fit"), 
                  ("joint", "dit_classifier", "diffusion_classifier_v2", "fit"), 
                  ("joint", "dit_classifier", "diffusion_classifier_v2", "fit_progressively"), 
                  ("generation", "vae", None, "fit"), 
                  ("joint", "vae_classifier", None, "fit"), 
                  ("classification", "dnn", None, "fit"))
        actual_runs = []

        def reduced_config(*args: object, **kwargs: object) -> Config:
            """Retain real search/dispatch choices while limiting synthetic trial cost.

            Args:
                *args (object): Builder positional arguments supplied by run_hpo.
                **kwargs (object): Builder keyword controls supplied by run_hpo.

            Returns:
                Config: Same route with small networks, four-step diffusion and reporting caps.
            """

            config = _build_trial_config(*args, **kwargs)
            config.dataset.batch_size = 4
            config.dataset.drop_remainder = False
            config.model.show_network_summary = False
            config.model.kwargs.setdefault("compile_args", {}).update(run_eagerly=True, jit_compile=False)
            # Diffusion routes keep the sampled topology but use tiny widths/horizons.
            if config.model.name.startswith(("dit", "diffusion")):
                config.model.kwargs.update(dim=8, depth=1, mha_num_heads=1, timesteps=4)
                config.model.wrapper_kwargs.update(test_steps=2)
                # Attached classifier heads need their own compatible tiny attention width.
                if config.model.name == "dit_classifier":
                    config.model.kwargs.update(clf_depth=1, clf_mha_num_heads=1)
                    config.model.wrapper_kwargs["clf_train_noisified_max_timesteps"] = None
            # Dense VAE routes preserve objective types while shrinking hidden/latent state.
            elif config.model.name in ("vae", "vae_classifier"):
                config.model.kwargs.update(latent_dim=4, hiddens_dims=[8])
                config.model.classifier_kwargs = {"architecture_kwargs": {"hidden_dims": [8]}, "dropout_rate": 0.}
            config.training.tensorboard = False
            config.training.patience = 0
            config.training.verbose = 0
            config.reporting.save_history_plot = False
            config.reporting.save_final_images = False
            config.reporting.save_final_gifs = False
            config.reporting.final_images_steps = 2
            config.hpo["software_check"] = "Synthetic MNIST rows and reduced architecture/budget"
            return config

        def record_run(*args: object, **kwargs: object) -> dict[str, object]:
            """Execute the real training pipeline and retain final reports for score checks.

            Args:
                *args (object): Main positional arguments, including the trial Config.
                **kwargs (object): Main runtime teacher keyword, forwarded unchanged.

            Returns:
                dict[str, object]: Actual main result, also appended to actual_runs.
            """

            result = train_main(*args, **kwargs)
            actual_runs.append(result)
            return result

        for index, (task, model, wrapper, fit) in enumerate(routes):
            space = {"optimizer": ["adam"], "batch_size": [4]}
            # Constrain topology rather than letting a one-trial smoke choose deep variational routes.
            if model.startswith(("dit", "diffusion")):
                space.update(capacity=["32x4"], patch_size=[4], patchify_with_cnn=[False], 
                             use_refiner_cnn=[False], schedule=["clipped_cosine"], image_loss_coef=[0.], 
                             droppath_rate=[0.], mlp_ratio=[2.], timesteps=[500])
            # Each joint row explicitly selects its wrapper and deterministic one-head topology.
            if wrapper is not None:
                space.update(wrapper_name=[wrapper], classifier_architecture=["linear"], 
                             classifier_only_cls_token=[True], feature_aggregation=["last"], 
                             clf_cls_token_type=["new_weight"], clf_droppath_rate=[0.], 
                             classifier_dropout_rate=[0.], classifier_mlp_ratio=[None], ctr_loss_coef=[0.], 
                             clf_vars_recipe=["separate"], clf_train_noisified_max_timesteps=[None])
            curriculum = {"stage_tasks": [("timesteps", (0, 4))], "stage_epochs": 1, "final_epochs": 1}
            with self.subTest(task=task, model=model, wrapper=wrapper, fit=fit), \
                    patch("tensorflow.keras.datasets.mnist.load_data", return_value=dataset), \
                    patch("common.hpo._build_trial_config", side_effect=reduced_config), \
                    patch("common.hpo.main", side_effect=record_run):
                study = run_hpo(task, model, "mnist", epochs=1, n_trials=1, seed=13, 
                                fit_method=fit, fit_kwargs=curriculum if fit == "fit_progressively" else None, 
                                max_train_samples=4, max_val_samples=4, search_space_overrides=space, 
                                results_path=self.root / f"real-{index}")
                self.assertEqual(study.trials[0].state, optuna.trial.TrialState.COMPLETE)
                self.assertTrue(np.isfinite(study.trials[0].values).all())
                saved = load_config(study.trials[0].user_attrs["resolved_config_path"])
                self.assertEqual(saved.hpo["objectives"], study.trials[0].values)
                expected = _objective_values(task, model, actual_runs[-1]["history"], actual_runs[-1]["evaluations"])
                values = list(expected) if isinstance(expected, tuple) else [expected]
                np.testing.assert_allclose(values, study.trials[0].values)
                self.assertTrue(actual_runs[-1]["history"])
        self.assertEqual(len(actual_runs), len(routes))
        print(f"HPO_MATRIX real synthetic studies: {len(routes)} complete trials", flush=True)

    def test_nonfinite_generic_final_scores_are_not_completed(self) -> None:
        """Reject infinite and NaN final objectives before they can enter best-trial selection.

        Only the training boundary is stubbed. Real Optuna allocation, SQLite,
        configuration persistence, objective extraction and subsequent-trial
        continuation execute for generation, classification, joint and continual studies.
        """

        cases = (("generation", "vae", "valset_eval", "mean_squared_error", float("inf")), 
                 ("classification", "dnn", "valset_eval", "accuracy", float("inf")), 
                 ("classification", "dnn", "valset_eval", "accuracy", float("-inf")), 
                 ("classification", "dnn", "valset_eval", "accuracy", float("nan")), 
                 ("joint", "dit_classifier", "valset_ema_eval", "classifier_accuracy", float("inf")), 
                 ("continual", "dnn", "validation_continual_metrics", "final_average_accuracy", float("inf")))
        for index, (task, model, report_key, metric, invalid) in enumerate(cases):
            def final_report(config: Config, **kwargs: object) -> dict[str, object]:
                """Supply a nonfinite first validation result and a finite second result.

                Args:
                    config (Config): Real persisted trial configuration.
                    **kwargs (object): Runtime teacher argument, unused by this score-only fixture.

                Returns:
                    dict[str, object]: Ordinary main-shaped history/evaluation/artifact mapping.
                """

                del kwargs
                output = Path(config.training.results_path) / f"trial-{config.hpo['trial_number']}"
                output.mkdir(parents=True)
                config.training.results_path = str(output)
                score = invalid if config.hpo["trial_number"] == 0 else .6
                values = {"noise_loss": .25, metric: score}
                return {"history": {}, "evaluations": {report_key: values}, "results_path": str(output)}

            with self.subTest(task=task, model=model, invalid=invalid), patch("common.hpo.main", side_effect=final_report):
                study = run_hpo(task, model, "mnist", n_trials=2, epochs=1, 
                                seed=11, results_path=self.root / f"nonfinite-{index}")
                self.assertEqual([trial.state for trial in study.trials], 
                                 [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE])
                self.assertEqual([trial.number for trial in study.best_trials], [1])
                evidence = Path(study.trials[0].user_attrs["divergence_path"])
                self.assertTrue(evidence.is_file())
                details = json.loads(evidence.read_text(encoding="utf-8"))
                self.assertEqual(details["reason"], "nonfinite_objective")
                expected_names = study.user_attrs["study_spec"]["objective_metrics"]
                self.assertEqual(set(details["objectives"]), set(expected_names))

    def test_objective_matrix_uses_only_selected_validation_branch(self) -> None:
        """Use distinct sentinels to verify every model/task's raw/EMA and scalar/Pareto routing."""

        report = {"noise_loss": 2., "mean_squared_error": 3., "total_accuracy": .7, 
                  "classifier_accuracy": .6, "accuracy": .5, "clf_accuracy": .4, 
                  "ensemble_accuracy": .8}
        continual = {"final_average_accuracy": .75, "average_incremental_accuracy": .775, 
                     "average_forgetting": .2, "backward_transfer": -.2}
        evaluations = {"valset_eval": report, "valset_network_eval": report, 
                       "valset_ema_eval": {key: value + .125 for key, value in report.items()}, 
                       "validation_continual_metrics": continual, 
                       "testset_eval": {key: 999. for key in report}}
        rows = 0
        for task, models in SEARCH_SPACES.items():
            for model, branch in product(models, ("raw", "ema")):
                # An umbrella study scores its resolved family rather than its selector name.
                family = "dit_classifier" if model == "diffusion_classifier" else model
                diffusion = family.startswith(("dit", "diffusion", "unet"))
                shift = .125 if diffusion and branch == "ema" else 0.
                expected = {"generation": (3. if family == "vae" else 2.) + shift, 
                            "classification": .5, "continual": .75, 
                            "joint": ((3. if family == "vae_classifier" else 2.) + shift, .7 + shift)}[task]
                with self.subTest(task=task, model=model, branch=branch):
                    value = _objective_values(task, family, {"val_loss": [-999.]}, evaluations, 
                                              diffusion_network_name=branch)
                    self.assertEqual(value, expected)
                    # Continual custom objectives preserve order and signed transfer values.
                    if task == "continual":
                        metrics = list(continual)
                        self.assertEqual(_objective_values(task, family, {}, evaluations, 
                            objective_metrics=metrics, diffusion_network_name=branch), tuple(continual.values()))
                    # Joint diffusion ensemble feedback replaces only the accuracy objective.
                    elif task == "joint" and diffusion:
                        self.assertEqual(_objective_values(task, family, {}, evaluations, 
                            use_ensemble_accuracy=True, diffusion_network_name=branch), (2. + shift, .8 + shift))
                    rows += 1
        self.assertEqual(rows, 48)
        print(f"HPO_MATRIX objective branches: {rows} model/task/network rows", flush=True)


# Run bounded configuration and numerical routing matrices under unittest.
if __name__ == "__main__":
    unittest.main()
