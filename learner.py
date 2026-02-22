def continually_learn(class_num: int, load_dataset_fn: callable, 
                    load_dataset_fn_kwargs: dict = {},
                    remove_prev_classes: bool = True, keep_same_model: bool = True, 
                    tuned_model_path: str = "", compile_args: dict = {}, 
                    use_loaded_opt: bool = False, batch_size: int = 128, 
                    epochs: int = 100, use_buffer: bool = False, 
                    buffer_kwargs: dict = {}, use_vae: bool = False, 
                    vae_init_kwargs: dict = {}, vae_kwargs: dict[str, int] = {}, 
                    plot_results: bool = True, verbose: bool = True) -> list[float]:
    """
    Runs a continual learning scenario with an optional solution.

    Args:
        class_num: 
            The number of total classes in the given dataset.

        load_dataset_fn:
            A function to load the designated dataset.

        load_dataset_fn_kwargs:
            A dictionary with the following keys to pass to the load_dataset_fn:
                preprocess:
                    What type of preprocessing to apply on the dataset (image dataset or features dataset). It can be one of the following: "", "normalize", or "min-max".
                onehot_labels:
                    Whether or not to onehot the labels in the dataset.

        remove_prev_classes:
            Whether to keep the previous tasks' classes or remove them for the current task.

        keep_same_model:
            Whether to use the same model for each continual task or reinstantiate it for each task.

        tuned_model_path: 
            Path to the hyperparameter-optimized model to run the learning process on. If the path contains "dnn", return_feature argument of load_dataset_fn will be True.

        compile_args:
            Complie arguments for the model being continually learned.

        use_loaded_opt:
            Whether to use the loaded optimizer or to use the optimizer from comple_args.

        batch_size:
            The batch size for the CL process.

        epochs:
            The number of epochs for the CL process.

        use_buffer:
            Whether or not to use a replay buffer to mitigate catastrophic forgetting. It cannot be used with the VAE.

        buffer_kwargs:
            A dictionary to be used for replay-buffer-related actions. Its keys are as follows:
                maxlen:
                    The maximum capacity for the buffer.

                sample_num:
                    The number of samples to be drawn from the buffer.

                insert_num:
                    The number of new samples to be inserted to the buffer after each task.

                seed: 
                    The random seed for the buffer to draw the samples for the CL process.

        use_vae:
            Whether or not to use VAE to mitigate catastrophic forgetting. It cannot be used with the replay buffer.

        vae_init_kwargs:
            Keyword arguments passed to the VAE's initializer.

        vae_kwargs:
            A dictionary to be used for VAE-related actions. Its keys are as follows:
                train_num:
                    The number of instances to train the VAE from the given input. 

                    If -1 is provided, all of the data is used and any other number makes the function to sample only that number of instances, then, train the VAE.

                samples_per_class:
                    The number of samples to be drawn from each class that VAE has seen before.

        plot_results:
            Whether to plot the accuracies list.

        verbose:
            Whether to print anything about the learning process.

    Returns:
        A list of accuracy scores corresponding to each continual-learning task.
    """


    from sklearn.metrics import (accuracy_score, 
                            classification_report, 
                            ConfusionMatrixDisplay)

    import numpy as np

    from matplotlib import pyplot as plt

    from common.utils import CL_plot, plot_history
    from model import get_model, copy_model, get_callbacks
    from replay_buffer import ReplayBuffer
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

