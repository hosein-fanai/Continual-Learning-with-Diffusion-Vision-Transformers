"""Offline and naive references using the thesis DiT and native training APIs.

These supplemental runs are separate from the frozen semantic-route campaign.
Both retain the shared diffusion/classification objective; neither uses a CL
retention mechanism. Their accuracy values are empirical references, not bounds.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from common.config import Config, resolve_continual_schedule, save_config
from common.continual_reporting import continual_metrics
from common.dataloader import get_dataset, load_cifar10, load_cifar100
from common.learner import _ensemble_accuracy_row, _load_continual_arrays, _predict_diffusion_classes
from common.model import get_model
from common.runtime import configure_runtime, derive_seed
from common.train import report, train_model
from notebooks.thesis.workflow import check_runtime
from semantic_consolidation.config import load_route_config
from semantic_consolidation.provenance import save_provenance, source_provenance


BENCHMARKS = ('offline_joint', 'naive_sequential')
OBJECTIVE = 'Shared thesis diffusion denoising + classification; no replay, distillation, or semantic phases.'


def _write_json(path: Path, value: dict) -> None:
    """Publish a JSON artifact atomically, rejecting nonfinite metric values."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def configure_reference(
    dataset: str,
    benchmark: str,
    seed: int = 17,
    evaluation_split: str = 'validation',
    *,
    config_path: str | Path | None = None,
    results_root: str | Path | None = None,
) -> Config:
    """Derive a separate reference config without changing the central recipe.

    Validation-only is the default. Explicit test runs are supplemental,
    unregistered references, not members of the frozen confirmation campaign.
    The class permutation matches ``workflow.load_development`` for this seed.
    """
    # Restrict references to the two maintained datasets and training protocols.
    if dataset not in ('cifar10', 'cifar100') or benchmark not in BENCHMARKS:
        raise ValueError('Choose cifar10/cifar100 and offline_joint/naive_sequential.')
    # Use an explicit seed supported by every shared random-number backend.
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    # Training rows cannot serve as held-out evaluation observations.
    if evaluation_split not in ('validation', 'test'):
        raise ValueError('evaluation_split must be validation or test.')
    path = Path(config_path) if config_path is not None else Path(__file__).parent / 'configs' / f'{dataset}.yaml'
    config = deepcopy(load_route_config(path).common)
    # A supplied recipe must describe the selected dataset.
    if config.dataset.name != dataset:
        raise ValueError('The recipe dataset differs from the requested dataset.')
    continual = config.continually_learn
    order, groups = resolve_continual_schedule(
        continual.class_num, continual.class_order, continual.task_groups,
        task_size=continual.task_size, seed=seed)
    order = np.random.default_rng(seed).permutation(order).tolist()
    boundaries = np.cumsum([0, *map(len, groups)])
    continual.class_order = order
    continual.task_groups = [order[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
    continual.class_order_mode = continual.task_order_mode = 'fixed'
    continual.seed = config.training.seed = seed
    config.dataset.indices = list(order)
    continual.baseline = 'joint_none' if benchmark == 'naive_sequential' else None
    continual.use_buffer = continual.use_generative_replay = continual.use_distillation = False
    continual.use_generative_model_classifier = True
    continual.remove_prev_classes = continual.keep_same_model = True
    continual.replay_budget_mode = 'fixed_total'
    continual.replay_old_examples = 0
    continual.replay_current_examples = None
    continual.replay_cache_mode = 'off'
    continual.mechanistic_metrics = False
    continual.plot_results = False
    continual.return_details = True
    continual.resume_from = continual.checkpoint_dir = None
    continual.experiment_manifest_path = continual.experiment_manifest_hash = continual.experiment_run_id = None
    continual.experiment_phase = 'development' if evaluation_split == 'validation' else 'legacy'
    continual.save_task_checkpoints = benchmark == 'naive_sequential'
    config.training.task = 'joint' if benchmark == 'offline_joint' else 'continual'
    config.training.fit_method = 'fit'
    config.training.project_tag = f'{benchmark}-{dataset}-seed-{seed}'
    config.training.results_path = str(results_root or './results/thesis_route_one/reference_benchmarks')
    config.training.report_every_epoch = config.training.show_images = config.training.save_gifs = False
    config.training.use_valset = True
    config.training.patience = 0
    config.model.kwargs['num_classes'] = len(order) if benchmark == 'offline_joint' else None
    config.model.wrapper_kwargs.update(
        clf_distil_loss_coef=0., noise_distil_loss_coef=0.,
        use_ema=False, test_network_name='raw', seen_classes={})
    config.model.wrapper_kwargs.pop('teacher_network', None)
    config.reporting.run_trainset_eval = config.reporting.run_valset_eval = False
    config.reporting.save_final_images = config.reporting.show_final_images = config.reporting.save_final_gifs = False
    config.reporting.show_history_plot = config.reporting.save_history_plot = False
    config.reporting.save_csv = True
    config.hpo['reference_benchmark'] = {
        'benchmark': benchmark, 'evaluation_split': evaluation_split,
        'objective': OBJECTIVE, 'recipe_path': str(path.resolve()),
        'confirmation_campaign_member': False,
        'bound_claim': 'Neither a guaranteed maximum nor a guaranteed minimum.'}
    return config


def _validate_controls(config: Config) -> dict:
    """Reject settings that would silently introduce a retention mechanism."""
    spec = config.hpo.get('reference_benchmark', {})
    benchmark = spec.get('benchmark')
    continual = config.continually_learn
    # Require the declared reference identity before allocating a run.
    if benchmark not in BENCHMARKS or spec.get('evaluation_split') not in ('validation', 'test'):
        raise ValueError('Use configure_reference before preparing a reference.')
    # Keep architecture and wrapper semantics aligned with the thesis platform.
    if config.model.name != 'dit_classifier' or config.model.wrapper_name != 'diffusion_classifier':
        raise ValueError('These references require the shared DiTClassifier/V1 platform.')
    # Reject retention mechanisms added after reference configuration.
    if continual.use_buffer or continual.use_generative_replay or continual.use_distillation \
            or config.model.wrapper_kwargs.get('clf_distil_loss_coef', 0) != 0 \
            or config.model.wrapper_kwargs.get('noise_distil_loss_coef', 0) != 0:
        raise ValueError('Reference replay, buffer and distillation controls must stay disabled.')
    expected_task = 'joint' if benchmark == 'offline_joint' else 'continual'
    # Each reference uses its declared native training route.
    if config.training.task != expected_task or config.training.fit_method != 'fit':
        raise ValueError('The selected reference training route was changed.')
    # Naive learning carries one model forward without revisiting previous rows.
    if benchmark == 'naive_sequential' and (
        continual.baseline != 'joint_none' or not continual.remove_prev_classes
        or not continual.keep_same_model or not continual.use_generative_model_classifier
    ):
        raise ValueError('Naive training must retain one expanding model and use current-task rows only.')
    # Prevent native evaluation from selecting a different held-out partition.
    if continual.experiment_phase != ('development' if spec['evaluation_split'] == 'validation' else 'legacy'):
        raise ValueError('Evaluation split and native experiment phase disagree.')
    # Preserve the shared pixel coordinates and sparse target representation.
    if config.dataset.preprocess != 'fixed-standardize' or config.dataset.return_features \
            or config.dataset.onehot_labels or config.dataset.pad:
        raise ValueError('The paired references require fixed-standardized, unpadded CIFAR pixels and sparse labels.')
    # Keep a nonempty validation partition separate from training in both routes.
    if not config.training.use_valset or not 0 < config.dataset.validation_ratio < 1:
        raise ValueError('A held-out validation partition is required in both references.')
    # Exact reference schedules count complete epochs over every permitted current row.
    if config.training.epochs < 1 or {'epochs', 'steps_per_epoch', 'initial_epoch'}.intersection(config.training.fit_kwargs) \
            or continual.optimizer_steps_per_epoch is not None or continual.replay_current_examples is not None:
        raise ValueError('Reference schedules require complete epochs without step or current-pool overrides.')
    return spec


def _offline_arrays(config: Config, loader: Callable[..., tuple] | None = None) -> tuple:
    """Use exactly the native CL split, caps, and schedule-position label mapping."""
    # A cached native loader lets schedule planning reuse the arrays consumed by training.
    if loader is None:
        loader = {'cifar10': load_cifar10, 'cifar100': load_cifar100}[config.dataset.name]
    arrays, _ = _load_continual_arrays(
        loader, config.continually_learn.class_order, False,
        {'preprocess': config.dataset.preprocess, 'onehot_labels': False,
         'validation_ratio': config.dataset.validation_ratio,
         'features_path': config.dataset.features_path, 'seed': config.training.seed},
        config.dataset.max_train_samples, config.dataset.max_val_samples,
        config.dataset.pad, config.training.seed)
    # Drop locked test contents before preparing a validation-only reference.
    if config.hpo['reference_benchmark']['evaluation_split'] == 'validation':
        arrays = (*arrays[:4], np.asarray(arrays[4])[:0].copy(), np.asarray(arrays[5])[:0].copy())
    return arrays


def _reference_training_budget(config: Config, labels: np.ndarray) -> dict:
    """Resolve full-epoch updates from native capped training labels before optimizer creation.

    Args:
        config (Config): Validated reference recipe; an inherited cosine horizon is replaced.
        labels (np.ndarray): Actual permitted sparse labels remapped to schedule positions.

    Returns:
        budget (dict): Training rows and partial-batch-inclusive counts for each fit/task,
            plus the exact planned optimizer applications across all epochs.

    Raises:
        ValueError: If a scheduled training stage has no permitted examples.
    """
    labels = np.asarray(labels).reshape(-1)
    # Offline learning fits one pooled training partition.
    if config.hpo['reference_benchmark']['benchmark'] == 'offline_joint':
        rows = [len(labels)]
    # Naive learning fits each task separately, retaining each task's partial batch.
    else:
        boundaries = np.cumsum([0, *map(len, config.continually_learn.task_groups)])
        rows = [int(np.count_nonzero((labels >= start) & (labels < stop)))
                for start, stop in zip(boundaries[:-1], boundaries[1:])]
    # Empty stages cannot supply their declared reference trajectory.
    if not all(rows):
        raise ValueError('Every reference training stage requires permitted training rows.')
    batches = [math.ceil(count / config.dataset.batch_size) for count in rows]
    updates = config.training.epochs * sum(batches)
    # Replay-platform horizons do not describe either reference's optimizer clock.
    if config.optimizer.schedule == 'cosine':
        config.optimizer.decay_steps = updates
    return {'training_rows_per_stage': rows, 'batches_per_epoch_per_stage': batches,
            'planned_optimizer_updates': updates, 'partial_batches_retained': True}


def prepare_reference(config: Config) -> dict:
    """Create one fresh run directory, paired inputs, and the existing model."""
    spec = _validate_controls(config)
    runtime = check_runtime()
    configure_runtime(seed=config.training.seed, dtype_policy=config.training.dtype_policy,
                      deterministic_ops=config.training.deterministic_ops)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = Path(config.training.results_path).resolve() / config.dataset.name / spec['benchmark'] / f'seed-{config.training.seed}-{run_id}'
    run_dir.mkdir(parents=True, exist_ok=False)
    config.training.results_path = str(run_dir / 'native')
    # Place native task checkpoints beneath this reference's fresh run directory.
    if config.continually_learn.save_task_checkpoints:
        config.continually_learn.checkpoint_dir = str(run_dir / 'checkpoints')
    context = {'run_dir': run_dir, 'config': config, 'spec': deepcopy(spec),
               'runtime': runtime, 'training_started': False, 'training_finished': False}
    # Offline training receives the complete native training partition at once.
    if spec['benchmark'] == 'offline_joint':
        x_train, y_train, x_val, y_val, x_test, y_test = _offline_arrays(config)
        context['trainset'] = get_dataset(
            x_train, y_train.reshape(-1), batch_size=config.dataset.batch_size,
            shuffle_buffer=config.dataset.shuffle_buffer, seed=config.training.seed, drop_remainder=False)
        context['valset'] = get_dataset(
            x_val, y_val.reshape(-1), batch_size=config.dataset.batch_size,
            shuffle_buffer=0, drop_remainder=False)
        config.dataset.trainset_len = math.ceil(len(x_train) / config.dataset.batch_size)
        context['evaluation_arrays'] = (x_val, y_val) if spec['evaluation_split'] == 'validation' else (x_test, y_test)
        context['split_counts'] = {'training': len(x_train), 'validation': len(x_val),
                                   'evaluated': len(context['evaluation_arrays'][0])}
        context['training_budget'] = _reference_training_budget(config, y_train)
    # Naive training defers task selection to the native continual loader.
    else:
        source_loader = {'cifar10': load_cifar10, 'cifar100': load_cifar100}[config.dataset.name]
        cached_arrays, cached_options = None, None

        def cached_loader(**options: object) -> tuple:
            """Reuse one native uncapped split; the learner still owns capping and its RNG."""
            nonlocal cached_arrays, cached_options
            # Load once for both budget resolution and the subsequent native task runner.
            if cached_arrays is None:
                cached_arrays, cached_options = source_loader(**options), deepcopy(options)
            # A changed loader contract cannot silently reuse another split or preprocessing.
            elif options != cached_options:
                raise ValueError('Reference data options changed after schedule preparation.')
            return cached_arrays

        arrays = _offline_arrays(config, loader=cached_loader)
        context['training_budget'] = _reference_training_budget(config, arrays[1])
        config.dataset.trainset_len = math.ceil(len(arrays[0]) / config.dataset.batch_size)
        context['trainset'], context['valset'] = cached_loader, None
    context['model'] = get_model(config)
    # No semantic adapter/controller is attached in either route.
    save_config(config, run_dir / 'reference_config.yaml')
    save_provenance(source_provenance(), run_dir)
    _write_json(run_dir / 'reference_plan.json', {
        **deepcopy(spec), 'runtime': runtime, 'dataset': config.dataset.name,
        'seed': config.training.seed, 'class_order': config.continually_learn.class_order,
        'task_groups': config.continually_learn.task_groups,
        'label_mapping': 'Original class IDs map to their positions in class_order.',
        'epochs': config.training.epochs, 'batch_size': config.dataset.batch_size,
        'training_budget': context['training_budget'], 'optimizer': {
            'schedule': config.optimizer.schedule, 'initial_learning_rate': config.optimizer.initial_learning_rate,
            'decay_steps': config.optimizer.decay_steps},
        'budget': 'Same epochs per current example; offline sees all classes up front. No claim of compute matching to replay methods.',
        'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    _write_json(run_dir / 'status.json', {'state': 'prepared'})
    return context


def train_reference(config: Config, context: dict) -> dict:
    """Run native fitting once; preserve failure status instead of reporting success."""
    _validate_controls(config)
    # A prepared context owns exactly one fitting attempt with its original config.
    if context.get('config') is not config or context.get('training_started'):
        raise RuntimeError('Prepare a new run in a fresh kernel before training again.')
    context['training_started'] = True
    start = time.monotonic()
    _write_json(context['run_dir'] / 'status.json', {'state': 'training'})
    try:
        history = train_model(config, context['model'], context['trainset'], valset=context['valset'])
    except BaseException as error:
        _write_json(context['run_dir'] / 'status.json', {'state': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    context['training_seconds'] = time.monotonic() - start
    context['training_finished'] = True
    context['history'] = history
    _write_json(context['run_dir'] / 'status.json', {'state': 'trained_pending_report'})
    return history


def finish_reference(config: Config, context: dict, history: dict) -> tuple[dict, pd.DataFrame]:
    """Save the configured ordinary/ensemble accuracy and applicable native CL metrics."""
    spec = _validate_controls(config)
    # Failed or mismatched runs cannot publish completed accuracy outcomes.
    if context.get('config') is not config or not context.get('training_finished'):
        raise RuntimeError('Complete training successfully before reporting this run.')
    # Reuse completed observations when the reporting cell is repeated.
    if context.get('summary') is not None:
        return context['summary'], context['per_task']
    report(config, history, context['model'], context['trainset'], context['valset'])
    groups = config.continually_learn.task_groups
    # Offline learning provides final task accuracies without temporal CL metrics.
    if spec['benchmark'] == 'offline_joint':
        x, y = context['evaluation_arrays']
        labels = np.asarray(y).reshape(-1)
        # An empty partition cannot provide an accuracy observation.
        if not len(labels):
            raise ValueError('The requested evaluation partition is empty.')
        boundaries = np.cumsum([0, *map(len, groups)])
        counts = [int(np.sum((labels >= start) & (labels < stop)))
                  for start, stop in zip(boundaries[:-1], boundaries[1:])]
        # Each scheduled task must contribute held-out rows to its accuracy.
        if not all(counts):
            raise ValueError('An evaluation task has no held-out examples.')
        # Match the primary inference policy of the continually trained references.
        if config.continually_learn.use_ensemble_accuracy:
            dense_groups = [list(range(start, stop)) for start, stop
                            in zip(boundaries[:-1], boundaries[1:])]
            final = _ensemble_accuracy_row(
                context['model'], x, y, dense_groups, len(groups),
                -1., 2., config.dataset.batch_size,
                config.continually_learn.ensemble_accuracy_kwargs,
                derive_seed(config.training.seed, 'ensemble', len(groups) - 1, spec['evaluation_split']),
                False)
            # Unavailable ensemble results cannot be published as final task scores.
            if not np.isfinite(final).all():
                raise ValueError('Offline ensemble evaluation returned unavailable accuracy.')
        # Preserve ordinary clean scoring when that is the configured primary endpoint.
        else:
            scores = _predict_diffusion_classes(context['model'], x, y, -1., 2., config.dataset.batch_size)
            # Every evaluated row needs finite scores for the entire class vocabulary.
            if scores.shape != (len(labels), len(config.continually_learn.class_order)) or not np.isfinite(scores).all():
                raise ValueError('Offline predictions lack complete, finite class support.')
            correct = np.argmax(scores, axis=1) == labels
            final = [float(np.mean(correct[(labels >= start) & (labels < stop)]))
                     for start, stop in zip(boundaries[:-1], boundaries[1:])]
        metrics = {'final_average_accuracy': float(np.mean(final)),
                   'average_incremental_accuracy': None, 'average_forgetting': None,
                   'backward_transfer': None, 'final_example_accuracy': float(np.average(final, weights=counts))}
    # Naive learning supplies the native matrix of learned-task observations.
    else:
        details = context['model']['continual_details']
        matrix = np.asarray(details['accuracy_matrix'], dtype=float)
        # Temporal metrics require every scheduled learned-task observation.
        if matrix.shape != (len(groups), len(groups)) or not np.isfinite(matrix[np.tril_indices(len(groups))]).all():
            raise ValueError('A complete finite learned-task trajectory is required.')
        final = matrix[-1].tolist()
        metrics = continual_metrics(matrix)
        pd.DataFrame(matrix, index=pd.RangeIndex(1, len(groups)+1, name='after_task'),
                     columns=[f'task_{i+1}' for i in range(len(groups))]).to_csv(context['run_dir'] / 'accuracy_matrix.csv')
    per_task = pd.DataFrame([
        {'task': index+1, 'original_classes': json.dumps(group),
         'accuracy': accuracy, 'accuracy_percent': 100 * accuracy}
        for index, (group, accuracy) in enumerate(zip(groups, final))])
    summary = {
        'benchmark': spec['benchmark'], 'dataset': config.dataset.name,
        'seed': config.training.seed, 'evaluation_split': spec['evaluation_split'],
        'metric_scale': 'fraction', **metrics,
        'accuracy_source': 'ensemble' if config.continually_learn.use_ensemble_accuracy else 'ordinary',
        'ensemble_accuracy_kwargs': deepcopy(config.continually_learn.ensemble_accuracy_kwargs)
        if config.continually_learn.use_ensemble_accuracy else {},
        'training_seconds': context['training_seconds'],
        'training_seconds_scope': 'Entire native train_model call, including any task evaluation, reporting and checkpoint writes; excludes preparation and final reference reporting.',
        'native_results_path': str(config.training.results_path),
        'objective': OBJECTIVE, 'confirmation_campaign_member': False,
        'temporal_metrics_note': 'Offline has no task-transition trajectory; its temporal metrics are unavailable.'
        if spec['benchmark'] == 'offline_joint' else 'Signed forgetting and backward transfer use the native learned-task matrix.'}
    per_task.to_csv(context['run_dir'] / 'final_per_task_accuracy.csv', index=False)
    _write_json(context['run_dir'] / 'summary.json', summary)
    _write_json(context['run_dir'] / 'status.json', {'state': 'completed', 'summary': 'summary.json'})
    context['summary'], context['per_task'] = summary, per_task
    return summary, per_task
