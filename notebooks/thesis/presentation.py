"""Small notebook views of existing measurements; no training or prediction."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

# Import annotation-only types without changing the runtime backend.
if TYPE_CHECKING:
    from semantic_consolidation.config import RouteConfig

import numpy as np
import pandas as pd


def describe_run(config: RouteConfig) -> None:
    """Show the few settings needed to identify this one stream.

    Args:
        config (RouteConfig): RouteConfig containing the live common and semantic settings for
            one stream.

    Returns:
        displayed (None): None; displays dataset/treatment, actual evaluation split, seed, work
            settings and native resume locator.

    Raises:
        KeyError: If the configuration lacks required stream fields.
    """
    from IPython.display import display
    project, route = config.common, config.route
    display(pd.Series({
        "dataset / condition": f"{project.dataset.name} / {route.condition}",
        "evaluation split": "test" if project.continually_learn.experiment_phase == "confirmation" else "validation",
        "recovery": project.continually_learn.resume_from or "fresh stream",
        "seed": project.training.seed,
        "classes / tasks": f"{project.continually_learn.class_num} / {len(project.continually_learn.task_groups)}",
        "joint epochs / batch": f"{project.training.epochs} / {project.dataset.batch_size}",
        "current rows": project.continually_learn.replay_current_examples or "all permitted training rows",
        "old replay rows": project.continually_learn.replay_old_examples,
        "configured semantic update budget": f"acquisition {route.acquisition_steps} / consolidation {route.consolidation_steps}",
    }, name="Selected stream").to_frame())


def show_learning_results(config: RouteConfig, bundle: dict, *, output_dir: str | Path | None=None, details: bool=True) -> tuple[Path, Path]:
    """Display native metrics and plot saved boundary/history observations.

    Args:
        config (RouteConfig): RouteConfig containing the live common and semantic settings for
            one stream.
        bundle (dict): Dictionary returned by common.model.get_model, with the live model and
            native continual results.
        output_dir (str | Path | None): Separate directory for saved view artifacts. None uses
            results/thesis_route_one/notebook_views/<run-name> below the repository root.
        details (bool): True includes saved diagnostic figures and extended views; False keeps
            the compact scalar presentation.

    Returns:
        locations (tuple[Path, Path]): Original run and separate view paths. Always displays
            native scalar metrics; details=True adds trajectory/history plots, while False
            leaves the compact outcome table.

    Raises:
        ValueError: If output overlaps original evidence or the saved accuracy matrix is
            invalid.
        OSError: If views cannot be saved.
    """
    from IPython.display import display
    from common.config import resolve_continual_schedule
    from semantic_consolidation.study import _completed_metrics
    project, result_details = config.common, bundle["continual_details"]
    run = Path(project.training.results_path).resolve()
    views = (Path(output_dir).resolve() if output_dir is not None else
             Path(__file__).resolve().parents[2] / "results/thesis_route_one/notebook_views" / run.name)
    # Notebook views must be separate from original run artifacts.
    if views == run or run in views.parents or views in run.parents:
        raise ValueError("Notebook views must be separate from original run artifacts.")
    continual = project.continually_learn
    confirmation = continual.experiment_phase == "confirmation"
    matrix = np.asarray(result_details["ordinary_accuracy_matrix" if confirmation else "validation_accuracy_matrix"], dtype=float)
    _, groups = resolve_continual_schedule(continual.class_num, continual.class_order,
                                          continual.task_groups, task_size=continual.task_size)
    # Final notebook values require every scheduled learned-task observation.
    # General reporting permits missing cells, which could hide an interrupted run.
    scalars = _completed_metrics(matrix, len(groups))
    metrics = pd.DataFrame([
        {"metric": name, "value": value * 100,
         "unit": "%" if "accuracy" in name else "percentage points",
         "split": "test" if confirmation else "validation", "completed_tasks": len(groups)}
        for name, value in scalars.items()
    ])
    views.mkdir(parents=True, exist_ok=True)
    display(metrics)
    metrics.to_csv(views / "metrics.csv", index=False)
    # Confirmation defaults to scalar outcomes; development retains learning plots.
    if not details:
        print("Native results:", run)
        return run, views
    from IPython.display import Image
    import matplotlib.pyplot as plt
    from common.continual_reporting import task_accuracy_summaries
    from common.utils import plot_history
    new, old = task_accuracy_summaries(matrix)
    trajectory = pd.DataFrame({"new": new, "old": old,
                               "all_seen": [np.mean(matrix[i, :i + 1]) for i in range(len(matrix))]},
                              index=pd.RangeIndex(1, len(matrix) + 1, name="completed_task")) * 100
    sizes = [len(group) for group in groups]
    trajectory["uniform_chance"] = 100. / np.cumsum(sizes)
    trajectory.to_csv(views / "accuracy_trajectory.csv")
    ax = trajectory[["new", "old", "all_seen"]].plot(marker="o", ylim=(0, 100), ylabel="Accuracy (%)")
    ax.plot(trajectory.index, trajectory.uniform_chance, ":", color="gray", label="Uniform all-seen chance")
    ax.set(title=f"{project.dataset.name} / {config.route.condition} / {'test' if confirmation else 'validation'}",
           xticks=trajectory.index)
    ax.legend()
    ax.figure.savefig(views / "accuracy_trajectory.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    # Native file-only reporting may select Agg; display the saved artifact
    # explicitly so notebook output does not depend on a global backend.
    display(Image(filename=str(views / "accuracy_trajectory.png")))
    histories = result_details.get("generative_histories", [])
    # Plot only measurements actually retained in the native history.
    if histories:
        last = histories[-1]
        keys = [key for key in ("loss", "classifier_loss", "noise_loss", "cls_token_accuracy") if key in last]
        # Plot only measurements actually retained in the native history.
        if keys:
            print("Final task joint history; all tasks remain in the native epoch_metrics.csv.")
            print("Classifier validation uses clean images; training uses masked/noisy views. "
                  "Validation total/noise losses are omitted here: clean zero-target noise loss "
                  "does not measure held-out denoising.")
            plotted_history = {key: values for key, values in last.items()
                               if key not in ("val_loss", "val_noise_loss")}
            plot_history(plotted_history, metrics=keys, col=2, figsize=(10, 6), show_all_x_ticks=False,
                         show_plots=False, plot_path=views / "last_task_joint_history.png")
            display(Image(filename=str(views / "last_task_joint_history.png")))
    print("Native results:", run)
    print("Notebook views:", views)
    return run, views


def show_diagnostics(run: str | Path, views: str | Path) -> dict[str, pd.DataFrame]:
    """Show concise phase/cost views; retain every available review table as CSV.

    Args:
        run (str | Path): Original saved run directory; no new prediction or training is
            requested.
        views (str | Path): Separate destination for notebook plots and CSV views.

    Returns:
        tables (dict[str, pd.DataFrame]): Saved validation phase, coverage and cost tables;
            detailed CSV remains separate from original native evidence.

    Raises:
        ValueError: If endpoints, task identity, validation split or resource evidence is
            invalid.
        OSError: If saved files cannot be read or views cannot be written.
    """
    from IPython.display import display
    from notebooks.thesis.development import review_development_run
    tables = review_development_run(run, output_dir=Path(views) / "diagnostics")
    names = {
        "gate_coverage": "Gate coverage (visits are updates, not distinct images)",
        "deployed_classifier_and_hidden_phase_changes": "Consolidation changes (after minus before)",
        "optimizer_work": "Optimizer work",
        "measured_task_runtime": "Measured runtime (disjoint task totals only)",
        "measured_checkpoint_io": "Measured recovery writes (separate from active task time)",
        "sampled_and_allocator_memory": "Sampled process RSS and allocator memory (bytes)",
        "replay_self_consistency": "Replay self-consistency (not independent image quality)",
    }
    for key, title in names.items():
        print(title)
        table = tables[key]
        if key == "deployed_classifier_and_hidden_phase_changes" and not table.empty:
            # Every column has one declared unit; no average over tasks or metrics.
            table = table.assign(measurement=table.measurement + " [" + table.change_unit + "]")
            table = table.pivot(index="task", columns="measurement", values="change_after_minus_before")
            table.columns = [name.replace("frozen_target_alignment.aggregates.selected_gates.", "")
                             .replace("representation.", "") for name in table.columns]
        display(table)
    print("Exposure, tensor storage and all detailed observations:", Path(views) / "diagnostics")
    return tables


def show_saved_replay(config: RouteConfig, bundle: dict, run: str | Path, views: str | Path) -> None:
    """Preview prespecified saved audit images without another diffusion pass.

    Args:
        config (RouteConfig): RouteConfig containing the live common and semantic settings for
            one stream.
        bundle (dict): Dictionary returned by common.model.get_model, with the live model and
            native continual results.
        run (str | Path): Original saved run directory; no new prediction or training is
            requested.
        views (str | Path): Separate destination for notebook plots and CSV views.

    Returns:
        displayed (None): None; displays a fixed saved final-task replay subset using native
            pixel inversion and original class IDs. A missing archive produces an unavailable
            message, never new sampling.

    Raises:
        ValueError: If saved arrays or conditioning labels cannot match the schedule.
        OSError: If the archive or separate image view cannot be read or written.
    """
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    count = len(config.common.continually_learn.task_groups)
    path = Path(run) / f"generated_examples_task_{count:03d}.npz"
    # Use existing evidence only when the corresponding artifact is present.
    if not path.is_file():
        print("No saved final-task generated examples are available.")
        return
    with np.load(path, allow_pickle=False) as archive:
        images, labels = archive["images"], archive["labels"]
    # Saved replay images and conditioning labels are inconsistent.
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) \
            or len(images) != len(labels) or not np.isfinite(images).all():
        raise ValueError("Saved replay images and conditioning labels are inconsistent.")
    unique = np.unique(labels)
    # An empty saved archive has no conditioning classes to preview.
    if not len(unique):
        print("The saved generated-example archive is empty.")
        return
    chosen = unique[np.linspace(0, len(unique) - 1, min(10, len(unique)), dtype=int)]
    indices = [int(np.flatnonzero(labels == label)[0]) for label in chosen]
    # Native capture saves 2 * display_sample - 1, independently of dataset
    # preprocessing. Undo that exact transform, not per-image extrema.
    pixels = (images[indices] + 1) / 2
    order = np.asarray(bundle["continual_details"]["class_order"], dtype=int)
    dense = labels[indices].astype(int)
    # Saved replay labels do not match the stream class order.
    if np.any(dense < 0) or np.any(dense >= len(order)):
        raise ValueError("Saved replay labels do not match the stream class order.")
    selected = pd.DataFrame({"saved_row": indices, "dense_conditioning_class": dense,
                             "original_class": order[dense], "source": path.name})
    selected.to_csv(Path(views) / "replay_preview_selection.csv", index=False)
    fig, axes = plt.subplots(1, len(indices), figsize=(max(4, len(indices) * 1.2), 2), squeeze=False,
                             constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        ax.imshow(np.clip(pixels[index], 0, 1).squeeze(), interpolation="nearest", cmap="gray")
        ax.set_title(f"Class {order[dense[index]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Saved generated examples — conditioning labels, qualitative only")
    fig.savefig(Path(views) / "saved_replay_preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(Path(views) / "saved_replay_preview.png")))
    display(selected.drop(columns="source"))
