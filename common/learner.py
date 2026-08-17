"""Legacy class-incremental learning experiment with replay alternatives."""


def continually_learn(class_num: int, load_dataset_fn: callable, 
                    load_dataset_fn_kwargs: dict = {},
                    remove_prev_classes: bool = True, keep_same_model: bool = True, 
                    tuned_model_path: str = "", compile_args: dict = {}, 
                    use_loaded_opt: bool = False, batch_size: int = 128, 
                    epochs: int = 100, use_buffer: bool = False, 
                    buffer_kwargs: dict = {}, use_vae: bool = False, 
                    vae_init_kwargs: dict = {}, vae_kwargs: dict[str, int] = {}, 
                    plot_results: bool = True, verbose: bool = True) -> list[float]:
    """Run class-incremental classifier training from two through N classes.

    A new output head is created at each task.  Optional replay comes either
    from a fixed-size sample buffer or a conditional dense VAE; the two modes
    are mutually exclusive.

    Args:
        class_num (int): Total class count ``N``.  The loop produces tasks for
            classes ``[0, 1]``, ``[0, 1, 2]``, ..., ``range(N)`` and therefore
            returns ``max(N - 1, 0)`` scores.
        load_dataset_fn (Callable[..., tuple[numpy.ndarray, ...]]): Loader called
            with ``indices``, ``return_features``, ``preprocess``,
            ``onehot_labels``, and ``verbose``.  It must return exactly
            ``(x_train, y_train, x_val, y_val, x_test, y_test)``.  The built-in
            :func:`common.dataloader.load_cifar10` and ``load_cifar100`` satisfy
            this interface.
        load_dataset_fn_kwargs (Mapping[str, object]): Loader options merged
            over ``{"preprocess": "", "onehot_labels": False}``.  For built-in
            loaders the optional additional key is ``features_path``.  Valid
            ``preprocess`` values are ``"normalize"``, ``"min-max"``, or
            ``""``/``None`` for no scaling; ``onehot_labels`` is bool.
            Do not include ``indices``, ``return_features``, or ``verbose``
            because this function passes them explicitly.  A custom loader may
            accept other keys.  Example: ``{"preprocess": "normalize",
            "onehot_labels": True}`` is required by VAE replay.
        remove_prev_classes (bool): If true, training receives classes 0 and 1
            at the first task and only the newly introduced class thereafter;
            validation/test still contain all seen classes.  If false, every
            split contains all classes seen so far.
        keep_same_model (bool): Copy learned non-head weights and old class-head
            columns into the next, one-class-wider model when true.  False uses
            a freshly cloned tuned architecture at each task.
        tuned_model_path (str | os.PathLike): Nonempty saved Keras model path.
            If its case-sensitive text contains ``"dnn"``, the loader is asked
            for saved 2,048-wide features; otherwise it is asked for images.
        compile_args (Mapping[str, object]): Overrides the classifier defaults
            accepted by ``tf.keras.Model.compile``, such as ``optimizer``,
            ``loss``, ``metrics``, ``loss_weights``, or ``run_eagerly``.
            Example: ``{"optimizer": "adam", "metrics": ["accuracy"]}``.
        use_loaded_opt (bool): Inherit the optimizer deserialized from the tuned
            model instead of ``compile_args["optimizer"]``.
        batch_size (int): Positive NumPy-input batch size for each classifier
            fit; defaults to 128.
        epochs (int): Positive maximum epochs per task; defaults to 100.
        use_buffer (bool): Enable fixed-capacity replay.  It must not be true
            together with ``use_vae``.
        buffer_kwargs (Mapping[str, object]): Replay controls merged over
            ``{"maxlen": 10000, "sample_num": 1000, "insert_num": 1000,
            "seed": None}``.  ``maxlen`` is deque capacity; ``sample_num`` is
            the maximum prior pairs concatenated before a task; ``insert_num``
            is the number sampled from that task's augmented training arrays
            after fitting; and ``seed`` is accepted by ``random.seed``.  Extra
            keys are retained but unused.  Example: ``{"maxlen": 5000,
            "sample_num": 500, "insert_num": 500, "seed": 42}``.
        use_vae (bool): Enable conditional-VAE replay.  This requires
            ``load_dataset_fn_kwargs["onehot_labels"]`` to be true and cannot
            be combined with ``use_buffer``.
        vae_init_kwargs (Mapping[str, object]): Options forwarded to
            :class:`autoencoder.variational_autoencoder.VariationalAutoencoder`
            after this function fixes ``conditioned=True`` and ``class_num``.
            Allowed project keys are ``data_dim``, ``latent_dim``,
            ``hiddens_dims``, ``hiddens_kwargs``, ``last_activation``, ``beta``,
            ``compile``, and ``compile_args``; Keras keys such as ``name``,
            ``dtype``, and ``trainable`` are also accepted.  Do not repeat
            ``conditioned`` or ``class_num``.  Within ``hiddens_kwargs``, only
            ``actv``, ``use_batch_norm``, and ``kernel_init`` are valid.  For
            example: ``{"data_dim": 2048, "latent_dim": 16,
            "hiddens_dims": (256, 64), "hiddens_kwargs":
            {"actv": "relu", "use_batch_norm": False}}``.
        vae_kwargs (Mapping[str, int]): VAE action controls merged over
            ``{"train_num": 1000, "samples_per_class": 1000}``.
            ``samples_per_class`` sets prior generations per seen class.
            ``train_num=-1`` fits current data without resampling; any other
            value triggers with-replacement resampling of
            ``max(train_num, len(x_train))`` rows, so a smaller positive value
            does not downsample.  Extra keys are retained but unused.
        plot_results (bool): Plot accuracy against the number of seen classes
            after all tasks.
        verbose (bool): Print task summaries, Keras progress, history figures,
            classification reports, and confusion matrices when true.

    Returns:
        list[float]: Test accuracy for each two-through-``class_num`` task, in
        order.  Each value is in ``[0.0, 1.0]``.

    Raises:
        AssertionError: If ``tuned_model_path`` is empty or both replay modes
            are enabled.
        TypeError: If a forwarded dictionary contains a conflicting or
            unsupported keyword.
        ValueError: If dataset shapes/labels cannot support the requested
            task, replay, or classifier loss.
    """


    from sklearn.metrics import (accuracy_score, 
                            classification_report, 
                            ConfusionMatrixDisplay)

    import numpy as np

    from matplotlib import pyplot as plt

    from common.utils import CL_plot, plot_history
    from common.model import get_model, copy_model, get_callbacks
    from common.replay_buffer import ReplayBuffer
    from autoencoder.variational_autoencoder import VariationalAutoencoder


    assert len(tuned_model_path) > 0, "tuned_model_path cannot be empty."
    assert (not use_buffer and not use_vae) or (use_buffer and not use_vae) or (not use_buffer and use_vae), "Both of the replay buffer and VAE cannot be used."


    load_dataset_fn_kwargs_default = {
        "preprocess": "", 
        "onehot_labels": False, 
    }
    load_dataset_fn_kwargs = {**load_dataset_fn_kwargs_default, **load_dataset_fn_kwargs}

    buffer_kwargs_default = {
        "maxlen": 10_000, 
        "sample_num": 1_000,
        "insert_num": 1_000,
        "seed": None,
    }
    buffer_kwargs = {**buffer_kwargs_default, **buffer_kwargs}

    vae_kwargs_default = {
        "train_num": 1_000,
        "samples_per_class": 1_000
    }
    vae_kwargs = {**vae_kwargs_default, **vae_kwargs}

    return_features = True if "dnn" in tuned_model_path else False

    if use_buffer:
        buffer = ReplayBuffer(maxlen=buffer_kwargs["maxlen"], seed=buffer_kwargs["seed"])

    if use_vae:
        vae = VariationalAutoencoder(conditioned=True, class_num=class_num, **vae_init_kwargs)

    prev_model = get_model(
        1, 
        model_type="hp-tuned", 
        model_path=tuned_model_path, 
        compile_args=compile_args, 
        use_loaded_opt=use_loaded_opt, 
        verbose=0
    )

    acc_list = []
    for i in range(class_num-1):
        if verbose:
            print(75*'-'+" Classes:", list(range(i+2)))

        new_model = get_model(
            i+2, model_type="hp-tuned", 
            model_path=tuned_model_path, 
            compile_args=compile_args, 
            use_loaded_opt=use_loaded_opt, 
            verbose=0
        )

        if keep_same_model:
            copy_model(prev_model, new_model)

        if remove_prev_classes:
            *_, x_val, y_val, x_test, y_test = load_dataset_fn(
                indices=list(range(0, i+2)), 
                return_features=return_features, 
                **load_dataset_fn_kwargs, 
                verbose=0,
            )
            if i == 0:
                x_train, y_train, *_ = load_dataset_fn(
                    indices=[i, i+1], 
                    return_features=return_features, 
                    **load_dataset_fn_kwargs,
                    verbose=0,
                )
            else:
                x_train, y_train, *_ = load_dataset_fn(
                    indices=[i+1], 
                    return_features=return_features, 
                    **load_dataset_fn_kwargs, 
                    verbose=0,
                )
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_fn(
                indices=list(range(0, i+2)), 
                return_features=return_features, 
                **load_dataset_fn_kwargs,
                verbose=0,
            )

        if use_buffer:
            x_buffer, y_buffer = buffer.sample_buffer_and_prepare_dataset(buffer_kwargs["sample_num"])
            # buffer.sample_dataset_and_extend_buffer((x_train, y_train), buffer_kwargs["insert_num"])

            if len(x_buffer) > 0:
                x_train = np.concatenate([x_train, x_buffer], axis=0)
                y_train = np.concatenate([y_train, y_buffer], axis=0)

        if use_vae:
            x_buffer, y_buffer = vae.generate(
                samples_per_class=vae_kwargs["samples_per_class"], 
                onehot_y_output=load_dataset_fn_kwargs["onehot_labels"], 
            )

            if len(x_buffer) > 0:
                x_train = np.concatenate([x_train, x_buffer], axis=0)
                y_train = np.concatenate([y_train, y_buffer], axis=0)

        history = new_model.fit(
            x_train, y_train, 
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val), 
            callbacks=get_callbacks(verbose=verbose),
            verbose=verbose,
        ).history
        prev_model = new_model

        if verbose:
            plot_history(history, indices=[1])

        if use_buffer:
            buffer.sample_dataset_and_extend_buffer((x_train, y_train), buffer_kwargs["insert_num"])

        if use_vae:
            vae.train(
                x_train, y_train, 
                vae_kwargs["train_num"], 
                clf=new_model,
                verbose=verbose
            )

        if load_dataset_fn_kwargs["onehot_labels"]:
            y_test = np.argmax(y_test, axis=-1)

        preds = new_model.predict(x_test, verbose=verbose)
        preds = np.argmax(preds, axis=-1)

        acc = accuracy_score(y_test, preds)
        acc_list.append(acc)

        if verbose:
            print(classification_report(y_test, preds, digits=4))
            ConfusionMatrixDisplay.from_predictions(y_test, preds)
            plt.show()

            print(75*'-'+'\n')

    if plot_results:
        CL_plot(class_num, [(acc_list, " ")])

    return acc_list
