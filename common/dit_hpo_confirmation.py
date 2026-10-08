"""Replay a selected DiT HPO input through the public training APIs.

The caller owns remote GPU admission, process isolation, finalist selection and
completion receipts. The training stack is imported only inside the worker
function, and selected trials' trained weights are never restored.
"""

from copy import deepcopy
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path


def run_confirmation(
    input_config_path: str | Path, 
    output_root: str | Path, 
    training_seed: int, 
    expected_config_sha256: str
) -> dict[str, object]:
    """Train one finalist from scratch while retaining its original data split.

    Run in a fresh admitted remote worker after installing its GPU memory limit.
    Dataset preparation uses the study seed, including its explicit shuffle seed;
    initialization, dropout and diffusion streams use the supplied fresh seed.
    Final reporting uses that fresh seed. Reuse the same fresh seeds across
    finalists for paired comparisons. Different batch sizes need not produce
    identical corruption draws, just as in the original HPO engine.

    Args:
        input_config_path (str | pathlib.Path): Immutable pre-training trial YAML,
            normally study_root/configs/trial-NNNN.yaml.
        output_root (str | pathlib.Path): Dedicated attempt directory, which must
            not contain confirmation-input.yaml. The caller serializes access
            and chooses a new attempt directory after failure.
        training_seed (int): Fresh training seed shared across finalist pairs.
            Runtime seed validation remains owned by configure_runtime.
        expected_config_sha256 (str): SHA-256 of the frozen source YAML bytes.

    Returns:
        dict[str, object]: JSON-serializable finite final EMA noise loss, seeds,
        source identity and artifact paths. No live model or dataset is returned.

    Raises:
        ValueError: Source identity changed, the input is not ordinary
            teacher-free DiT generation, validation is absent, or its final
            objective is nonfinite.
        TypeError: The final validation objective is not a real scalar.
        KeyError: Reporting omitted the required EMA noise loss.
        FileExistsError: This attempt already has an immutable input config.
        OSError: Source or output files cannot be accessed. Training and runtime
            errors propagate without marking completion.
    """

    source_path = Path(input_config_path).resolve()
    source_digest = sha256(source_path.read_bytes()).hexdigest()
    # Refuse a finalist whose immutable input differs from the frozen plan.
    if source_digest != expected_config_sha256:
        raise ValueError("Finalist input config does not match its frozen SHA-256.")
    attempt_root = Path(output_root).resolve()
    confirmation_input = attempt_root / "confirmation-input.yaml"
    # Preserve artifacts from any previous attempt in this directory.
    if confirmation_input.exists():
        raise FileExistsError(f"Confirmation attempt already exists: {confirmation_input}")

    from common.config import load_config, save_config
    from common.dataloader import get_datasets
    from common.model import get_model
    from common.runtime import configure_runtime
    from common.train import report, train_model


    config = load_config(source_path)
    # Detect source replacement between authentication and YAML loading.
    if sha256(source_path.read_bytes()).hexdigest() != source_digest:
        raise ValueError("Finalist input config changed while it was being loaded.")
    # Only the ordinary teacher-free DiT generation protocol is supported.
    if (
        config.training.task != "generation"
        or config.model.name != "diffusion_transformer"
        or config.model.wrapper_name not in (None, "diffusion_model")
        or config.training.fit_method != "fit"
        or config.hpo.get("use_distillation", False)
        or config.model.wrapper_kwargs.get("swap_noise_image", False)
        or config.model.wrapper_kwargs.get("noise_distil_loss_coef", 0.) != 0.
    ):
        raise ValueError("Confirmation requires ordinary teacher-free DiT generation HPO.")
    # Keep the original seeded partition, EMA score and fresh-fit protocol.
    if (
        not config.model.wrapper_kwargs.get("use_ema", True)
        or config.model.wrapper_kwargs.get("test_network_name", "ema") != "ema"
        or config.dataset.validation_source != "split"
        or not config.training.use_valset
        or config.training.seed is None
        or config.training.fit_kwargs.get("initial_epoch", 0) != 0
    ):
        raise ValueError("Confirmation requires fresh fitting and seeded split/EMA validation.")

    split_seed = config.training.seed
    source_trial_number = config.hpo.get("trial_number")
    data_config = deepcopy(config)
    configure_runtime(
        dtype_policy=data_config.training.dtype_policy, 
        deterministic_ops=data_config.training.deterministic_ops, 
        seed=split_seed
    )
    trainset, valset = get_datasets(data_config)
    # A missing partition cannot be replaced by train or official test rows.
    if valset is None:
        raise ValueError("Confirmation requires the original explicit validation partition.")

    config.dataset.trainset_len = data_config.dataset.trainset_len
    config.dataset.split_metadata = deepcopy(data_config.dataset.split_metadata)
    config.training.seed = training_seed
    config.model.kwargs["seed"] = training_seed
    config.model.wrapper_kwargs["seed"] = training_seed
    config.model.weights_path = None
    config.continually_learn.resume_from = None
    config.continually_learn.checkpoint_dir = None
    config.training.results_path = str(attempt_root / "runs")
    config.training.project_tag = f"confirmation-t{source_trial_number}-s{training_seed}"
    config.training.tensorboard = True
    config.training.tensorboard_path = str(attempt_root / "tensorboard")
    config.training.tensorboard_run_name = config.training.project_tag
    # Fresh-seed confirmation never inherits search-time performance pruning.
    for key in (
        "objectives", "checkpoint_dir", "resume_original_trial_number", 
        "resolved_config_path", "classifier_weights_path", "execution", 
        "pruning", "pruning_exchange", "pruning_monitor"
    ):
        config.hpo.pop(key, None)
    config.hpo.update({
        "seed": training_seed, 
        "input_config_path": str(confirmation_input), 
        "confirmation": {
            "source_input_config_path": str(source_path), 
            "source_input_config_sha256": source_digest, 
            "source_trial_number": source_trial_number, 
            "split_seed": split_seed, 
            "dataset_shuffle_seed": split_seed, 
            "training_seed": training_seed, 
            "report_seed": training_seed, 
            "fresh_weights": True
        }
    })
    attempt_root.mkdir(parents=True, exist_ok=True)
    save_config(config, confirmation_input)

    # The public model factory installs the fresh runtime seed before building.
    model = get_model(config)
    history = train_model(config, model, trainset, valset=valset)
    evaluations = report(config, history, model, trainset, valset=valset)
    metric = evaluations["valset_ema_eval"]["noise_loss"]
    # Reject structured or boolean scores instead of changing the objective.
    if isinstance(metric, bool) or not isinstance(metric, Real):
        raise TypeError("Final EMA validation noise_loss must be a real scalar.")
    objective = float(metric)
    # Divergent final evaluations cannot count as successful confirmations.
    if not isfinite(objective):
        raise ValueError("Final EMA validation noise_loss is nonfinite.")

    config.hpo["objectives"] = [objective]
    resolved_path = Path(config.training.results_path) / "config.yaml"
    save_config(config, resolved_path)
    return {
        "objective": objective, 
        "objective_metric": "noise_loss", 
        "objective_direction": "minimize", 
        "objective_network": "ema", 
        "source_trial_number": source_trial_number, 
        "source_input_config_path": str(source_path), 
        "source_input_config_sha256": source_digest, 
        "confirmation_input_config_path": str(confirmation_input), 
        "resolved_config_path": str(resolved_path), 
        "results_path": str(config.training.results_path), 
        "weights_path": config.model.weights_path, 
        "split_seed": split_seed, 
        "dataset_shuffle_seed": split_seed, 
        "training_seed": training_seed, 
        "report_seed": training_seed, 
        "epochs_maximum": config.training.epochs, 
        "early_stopping_patience": config.training.patience, 
        "fresh_weights": True
    }