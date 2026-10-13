"""Separate frozen-feature TMCL linear and weighted-kNN evaluation.

These readouts use all supplied training labels, independently of the sparse
labels used to acquire modulations. The declared vocabulary may include future
representation tasks: the paper's offline full-benchmark readout uses their
training labels as well. The caller owns the train/evaluation split
and must supply clean images in wrapper model coordinates. No diffusion noise,
modulation, replay, classifier finetuning or test-time augmentation is applied.
The DiT endpoint is declared by the adapter rather than represented as ConViT.
Reference: Dendritic-Learning-Group/tmcl at
2cb306cc5cbe6d13a3dd4991998ee3e7c2948c8b, main_linear.py and main_knn.py.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from common.dataloader import get_dataset
from common.model import _get_classifier_model
from common.runtime import derive_seed
from common.train import train_model
from semantic_consolidation.tmcl_optimizers import TMCLSGD


def _state(adapter: object) -> list[tuple[object, np.ndarray]]:
    """Snapshot every native and modulation variable before frozen evaluation."""

    variables = list(adapter.network.variables) + adapter.bank.variables_for()
    return [(variable, variable.numpy().copy()) for variable in variables]


def _require_frozen(before: list[tuple[object, np.ndarray]], adapter: object) -> None:
    """Reject variable growth or mutations including nontrainable normalization state."""

    current = list(adapter.network.variables) + adapter.bank.variables_for()
    # A readout cannot change either the model state or the retained modulation bank.
    if (len(current) != len(before) or any(left is not right for (left, _), right in zip(before, current))
            or any(not np.array_equal(value, variable.numpy()) for variable, value in before)):
        raise RuntimeError("Frozen TMCL evaluation changed the backbone or modulation state.")


def extract_frozen_features(adapter: object, images: np.ndarray, last_layers: int = 4, 
                            batch_size: int = 256, flip: bool = False) -> np.ndarray:
    """Extract clean or deterministically flipped cached features without changing state."""

    images = np.asarray(images, dtype="float32")
    spec = adapter.readout_spec(last_layers)
    # Feature rows must correspond to an explicitly nonempty aligned image cohort.
    if images.ndim != 4 or len(images) == 0 or not np.isfinite(images).all():
        raise ValueError("Frozen feature extraction requires finite nonempty NHWC images.")
    before = _state(adapter)
    chunks = []
    try:
        for start in range(0, len(images), batch_size):
            inputs = tf.convert_to_tensor(images[start:start + batch_size])
            # Horizontal flips are cached separately and selected only during probe training.
            if flip:
                inputs = tf.reverse(inputs, axis=[2])
            output = np.asarray(adapter.readout_features(inputs, last_layers).numpy(), dtype="float32")
            # A malformed layer endpoint must not silently enter the linear/kNN protocol.
            if output.shape != (len(inputs), spec["width"]) or not np.isfinite(output).all():
                raise ValueError("Frozen readout returned nonfinite or misaligned feature vectors.")
            chunks.append(output)
    finally:
        _require_frozen(before, adapter)
    return np.concatenate(chunks, axis=0)


def _labels(values: np.ndarray, classes: list[int], count: int) -> np.ndarray:
    """Map exact external IDs to the shared declared benchmark vocabulary."""

    values = np.asarray(values)
    # Distinct integer vocabulary and aligned integer labels prevent task-aware relabelling.
    if (not classes or len(set(classes)) != len(classes) or any(type(value) is not int for value in classes)
            or values.shape != tuple([count]) or not np.issubdtype(values.dtype, np.integer)
            or not np.isin(values, classes).all()):
        raise ValueError("Readout labels must align with the distinct declared integer class vocabulary.")
    lookup = {value: index for index, value in enumerate(classes)}
    return np.asarray([lookup[int(value)] for value in values], dtype="int32")


def classification_metrics(predictions: np.ndarray, labels: np.ndarray, classes: list[int], 
                           task_classes: list[list[int]] | None = None) -> dict[str, object]:
    """Score global predictions; task membership selects rows, never classifier columns."""

    truth = _labels(labels, classes, len(labels))
    predicted = _labels(predictions, classes, len(labels))
    correct = predicted == truth
    # Empty endpoints cannot define an observed accuracy.
    if len(correct) == 0:
        raise ValueError("Readout evaluation requires a nonempty endpoint.")
    per_class = {}
    for index, value in enumerate(classes):
        selected = truth == index
        count = int(selected.sum())
        per_class[str(value)] = {"count": count, "correct": int(correct[selected].sum()), 
                                 "accuracy": float(correct[selected].mean()) if count else None}
    task_results = []
    # Optional tasks must form a disjoint partition of the same declared vocabulary.
    if task_classes is not None:
        flattened = [value for task in task_classes for value in task]
        # Duplicated/missing classes would change the meaning of average continual accuracy.
        if sorted(flattened) != sorted(classes) or any(not task for task in task_classes):
            raise ValueError("Readout task classes must partition the complete class vocabulary.")
        for task in task_classes:
            selected = np.isin(labels, task)
            count = int(selected.sum())
            task_results.append({"classes": list(task), "count": count, 
                                 "correct": int(correct[selected].sum()), 
                                 "accuracy": float(correct[selected].mean()) if count else None})
    return {"accuracy": float(correct.mean()), "correct": int(correct.sum()), 
            "count": len(correct), "per_class": per_class, "tasks": task_results, 
            "prediction_scope": "all_declared_classes"}


def paper_transfer_metrics(accuracy_matrix: np.ndarray, 
                           independent_baselines: np.ndarray) -> dict[str, object]:
    """Compute TMCL Appendix B equations 9 and 10 from a full global-vocabulary matrix.

    Rows of the complete T-by-T matrix are completed representation sessions;
    columns are evaluation tasks, including future tasks at earlier sessions.
    Every cell uses the same full-benchmark classifier vocabulary. Baseline i
    is the task-agnostic kNN accuracy of an independently trained, from-scratch
    task-i-only VI/Barlow encoder, with the same full-label evaluation bank.
    It is neither A[i,i] nor the random-initialization accuracy. Callers must
    authenticate those data/training identities; arrays alone cannot prove them.

    Inputs are finite accuracy fractions in [0,1]. BT averages each task's
    later-session differences from its independent baseline, then averages
    tasks. FT analogously uses each task's earlier sessions. No conventional
    final-minus-diagonal backward-transfer metric is substituted.
    Source: https://arxiv.org/html/2505.14125v3#A2.SS1
    """

    matrix = np.asarray(accuracy_matrix, dtype="float64")
    baseline = np.asarray(independent_baselines, dtype="float64")
    # Both off-diagonal directions and every independent baseline must be observed.
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2
            or baseline.shape != tuple([matrix.shape[0]])
            or not np.isfinite(matrix).all() or not np.isfinite(baseline).all()):
        raise ValueError("Paper transfer requires a complete finite T-by-T matrix and T independent baselines, T>=2.")
    # Accuracy normalization prevents mixing fractions and percentage-point inputs.
    if np.any((matrix < 0.) | (matrix > 1.)) or np.any((baseline < 0.) | (baseline > 1.)):
        raise ValueError("Paper transfer accuracies and baselines must be fractions in [0,1].")
    tasks = matrix.shape[0]
    backward = [float(np.mean(matrix[index + 1:, index] - baseline[index]))
                for index in range(tasks - 1)]
    forward = [float(np.mean(matrix[:index, index] - baseline[index]))
               for index in range(1, tasks)]
    bt = float(np.mean(backward))
    ft = float(np.mean(forward))
    return {"paper_bt": bt, "paper_ft": ft, 
            "paper_bt_percentage_points": 100. * bt, "paper_ft_percentage_points": 100. * ft, 
            "backward_by_task": backward + [None], "forward_by_task": [None] + forward, 
            "tasks": tasks, "units": "accuracy_fraction", 
            "baseline_definition": "independent_task_only_VI_Barlow_encoder_global_vocabulary_kNN", 
            "matrix_definition": "all_sessions_by_all_tasks_global_vocabulary", 
            "equations": "TMCL_v3_Appendix_B_9_10"}


class _EpochCosine(tf.keras.callbacks.Callback):
    """Apply reference CosineAnnealingLR's epoch clock with no minibatch decay."""

    def __init__(self, initial_rate: float, epochs: int) -> None:
        """Bind the fresh probe's scaled initial rate and total epoch horizon."""

        super().__init__()
        self.initial_rate = initial_rate
        self.epochs = epochs
        self.rates: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Set one rate shared by every complete batch in this epoch."""

        rate = self.initial_rate * .5 * (1. + np.cos(np.pi * epoch / self.epochs))
        self.model.optimizer.learning_rate.assign(rate)
        self.rates.append(float(rate))


def linear_evaluate(adapter: object, train_images: np.ndarray, train_labels: np.ndarray, 
                    eval_images: np.ndarray, eval_labels: np.ndarray, classes: list[int], 
                    task_classes: list[list[int]] | None = None, epochs: int = 100, 
                    batch_size: int = 1024, last_layers: int = 4, 
                    feature_batch_size: int = 256, base_learning_rate: float = .1, 
                    verbose: int | str = 0, seed: int = 101) -> dict[str, object]:
    """Fit a fresh native linear probe on all training labels and evaluate clean images.

    The fixed encoder permits exact caching of both horizontal-flip states;
    each shuffled training row chooses a new Bernoulli(.5) flip on each epoch.
    This preserves online flip semantics without rerunning a frozen backbone.
    ``base_learning_rate`` is scaled by batch_size/256, as in the reference.
    The last incomplete training batch is dropped; evaluation retains all rows.
    """

    train_labels = _labels(train_labels, classes, len(train_images))
    _labels(eval_labels, classes, len(eval_images))
    # Dropping every training row or using a fractional horizon cannot fit the stated probe.
    if type(epochs) is not int or epochs < 1 or len(train_images) < batch_size:
        raise ValueError("Linear evaluation requires positive integer epochs and at least one full batch.")
    # An absent declared class would silently replace the full-benchmark readout with a sparse one.
    if set(train_labels.tolist()) != set(range(len(classes))):
        raise ValueError("Linear training labels must cover every declared class.")
    before = _state(adapter)
    clean = extract_frozen_features(adapter, train_images, last_layers, feature_batch_size)
    flipped = extract_frozen_features(adapter, train_images, last_layers, feature_batch_size, flip=True)
    evaluation = extract_frozen_features(adapter, eval_images, last_layers, feature_batch_size)
    variants = tf.constant(np.stack([clean, flipped], axis=1))
    rng = tf.random.Generator.from_seed(derive_seed(seed, "tmcl_linear_flip"))

    def choose_flip(indices: tf.Tensor) -> tf.Tensor:
        """Select exactly one frozen feature vector per independently flipped image row."""

        choice = tf.cast(rng.uniform(tf.shape(indices), dtype=tf.float32) < .5, tf.int32)
        return tf.gather_nd(variants, tf.stack([tf.cast(indices, tf.int32), choice], axis=1))

    dataset = get_dataset(
        np.arange(len(clean), dtype="int32"), train_labels, 
        shuffle_buffer=len(clean), batch_size=batch_size, drop_remainder=True, 
        augment_fn=choose_flip, num_parallel_calls=1, prefetch=False, 
        seed=derive_seed(seed, "tmcl_linear_shuffle")
    )
    rate = base_learning_rate * batch_size / 256.
    model = _get_classifier_model(
        len(classes), model_type="DNN", dropout_rate=0., 
        compile_args={"optimizer": TMCLSGD(learning_rate=rate, momentum=.9), 
                      "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}, 
        architecture_kwargs={"input_shape": tuple([clean.shape[1]]), "hidden_dims": (), 
                             "use_batch_norm": False}, verbose=0, seed=seed
    )
    # Native Linear initialization uses independent uniform kernel and bias distributions.
    initializer = np.random.default_rng(derive_seed(seed, "tmcl_linear_initialization"))
    bound = 1. / np.sqrt(clean.shape[1])
    model.layers[-1].set_weights([
        initializer.uniform(-bound, bound, (clean.shape[1], len(classes))).astype("float32"), 
        initializer.uniform(-bound, bound, len(classes)).astype("float32")
    ])
    schedule = _EpochCosine(rate, epochs)
    try:
        history = train_model(
            None, model, dataset, save_config_=False, extra_callbacks=[schedule], 
            epochs=epochs, results_path=None, show_images=True, save_gifs=False, 
            report_every_epoch=False, save_weights=False, use_tensorboard=False, verbose=verbose
        )
        expected_updates = epochs * (len(clean) // batch_size)
        # Full-budget probes do not permit early stopping or missing training batches.
        if int(model.optimizer.iterations.numpy()) != expected_updates:
            raise RuntimeError("Linear probe optimizer count differs from the declared full budget.")
        predicted = []
        for start in range(0, len(evaluation), feature_batch_size):
            probabilities = model(evaluation[start:start + feature_batch_size], training=False).numpy()
            predicted.extend(np.asarray(classes)[np.argmax(probabilities, axis=1)].tolist())
    finally:
        _require_frozen(before, adapter)
    return {"method": "linear", "metrics": classification_metrics(
                np.asarray(predicted), np.asarray(eval_labels), classes, task_classes), 
            "feature_spec": adapter.readout_spec(last_layers), "classes": list(classes), 
            "epochs": epochs, "batch_size": batch_size, "optimizer_updates": expected_updates, 
            "learning_rates": schedule.rates, "history": history, "seed": seed, 
            "train_rows": len(clean), "train_labels": "all_supplied_training_labels", 
            "augmentation": "independent_horizontal_flip_p0.5", "l2_normalize": False, 
            "head_weights": [value.tolist() for value in model.get_weights()]}


def weighted_knn_predict(bank: np.ndarray, bank_labels: np.ndarray, queries: np.ndarray, 
                         classes: list[int], neighbors: int = 20, 
                         temperature: float = .07, batch_size: int = 256) -> np.ndarray:
    """Use cosine top-k and temperature-weighted votes over every labelled bank row."""

    bank = np.asarray(bank, dtype="float32")
    queries = np.asarray(queries, dtype="float32")
    labels = _labels(bank_labels, classes, len(bank))
    # A silently clipped neighborhood changes the published readout and is not permitted.
    if type(neighbors) is not int or neighbors < 1 or neighbors > len(bank):
        raise ValueError("kNN requires an integer neighborhood no larger than its complete bank.")
    # Malformed and nonfinite matrices cannot define the reference feature bank.
    if (bank.ndim != 2 or queries.ndim != 2 or len(queries) == 0 or bank.shape[1] != queries.shape[1]
            or not np.isfinite(bank).all() or not np.isfinite(queries).all()):
        raise ValueError("kNN features must be aligned finite vectors.")
    # A bank missing a declared class cannot represent the stated full-benchmark evaluation.
    if set(labels.tolist()) != set(range(len(classes))):
        raise ValueError("kNN bank must contain every declared class.")
    # F.normalize's norm floor 1e-12 is a squared-norm floor 1e-24 in TensorFlow.
    normalized = tf.math.l2_normalize(tf.constant(bank), axis=1, epsilon=1e-24)
    predictions = []
    for start in range(0, len(queries), batch_size):
        current = tf.math.l2_normalize(tf.constant(queries[start:start + batch_size]), axis=1, epsilon=1e-24)
        cosine = tf.linalg.matmul(current, normalized, transpose_b=True)
        similarity, indices = tf.math.top_k(cosine, k=neighbors, sorted=True)
        weights = tf.exp((similarity - similarity[:, :1]) / temperature)
        votes = tf.reduce_sum(tf.one_hot(tf.gather(labels, indices), len(classes)) * weights[..., None], axis=1)
        predictions.extend(np.asarray(classes)[tf.argmax(votes, axis=1).numpy()].tolist())
    return np.asarray(predictions, dtype="int64")


def knn_evaluate(adapter: object, train_images: np.ndarray, train_labels: np.ndarray, 
                 eval_images: np.ndarray, eval_labels: np.ndarray, classes: list[int], 
                 task_classes: list[list[int]] | None = None, neighbors: int = 20, 
                 temperature: float = .07, feature_batch_size: int = 256) -> dict[str, object]:
    """Evaluate clean last-layer features with a full-label bank and no learned probe."""

    bank = extract_frozen_features(adapter, train_images, 1, feature_batch_size)
    queries = extract_frozen_features(adapter, eval_images, 1, feature_batch_size)
    predictions = weighted_knn_predict(
        bank, train_labels, queries, classes, neighbors, temperature, feature_batch_size
    )
    return {"method": "weighted_knn", "metrics": classification_metrics(
                predictions, np.asarray(eval_labels), classes, task_classes), 
            "feature_spec": adapter.readout_spec(1), "classes": list(classes), 
            "neighbors": neighbors, "temperature": temperature, "bank_rows": len(bank), 
            "zero_norm_bank_rows": int(np.sum(np.linalg.norm(bank, axis=1) == 0)), 
            "zero_norm_query_rows": int(np.sum(np.linalg.norm(queries, axis=1) == 0)), 
            "train_labels": "all_supplied_training_labels", "augmentation": None, "l2_normalize": True}
