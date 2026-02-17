from typing import Literal


def continually_learn(class_num: int, load_dataset_fn: callable, 
                    keep_same_model: bool, remove_prev_classes: bool, 
                    tuned_model_path: str, onehot_labels: bool = False, 
                    preprocess: Literal["", "normalize", "min-max"] = None, 
                    compile_args: dict = None, use_loaded_opt: bool = False, 
                    batch_size: int = 128, epochs: int = 100, buffer_maxlen: int = 0, 
                    buffer_sample_num: int = 1_000, buffer_insert_num: int = 1_000,
                    buffer_seed: int = None, use_vae: bool = False, vae_train_num: int = 1_000, 
                    vae_per_class_num: int = 1_000, verbose: bool = True) -> list[float]:
    """
    Runs a continual learning scenario with an optional solution.

    Args:
        class_num: number of total classes

        load_dataset_fn:

        keep_same_model:

        remove_prev_classes:

        tuned_model_path:

        onehot_labels:

        preprocess:

        compile_args:

        use_loaded_opt:

        batch_size:

        epochs:

        buffer_maxlen:

        buffer_sample_num:

        buffer_insert_num:

        buffer_seed: 

        use_vae:

        vae_train_num:

        vae_per_class_num:

        verbose: 

    Returns:
        A list of accuracy scores corresponding each continual-learning task.
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


    return_features = True if "dnn" in tuned_model_path else False

    if buffer_maxlen > 0:
        buffer = ReplayBuffer(maxlen=buffer_maxlen, seed=buffer_seed)

    if use_vae:
        vae = VariationalAutoencoder(conditioned=True, class_num=class_num)

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
            copy_model(prev_model, new_model) # , copy_opt_states=copy_opt_states

        if remove_prev_classes:
            *_, x_val, y_val, x_test, y_test = load_dataset_fn(
                indices=list(range(0, i+2)), 
                preprocess=preprocess,
                return_features=return_features, 
                onehot_labels=onehot_labels, 
                verbose=0
            )
            if i == 0:
                x_train, y_train, *_ = load_dataset_fn(
                    indices=[i, i+1], 
                    preprocess=preprocess, 
                    return_features=return_features, 
                    onehot_labels=onehot_labels, 
                    verbose=0
                )
            else:
                x_train, y_train, *_ = load_dataset_fn(
                    indices=[i+1], 
                    preprocess=preprocess, 
                    return_features=return_features, 
                    onehot_labels=onehot_labels, 
                    verbose=0
                )
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_fn(
                indices=list(range(0, i+2)), 
                preprocess=preprocess,
                return_features=return_features, 
                onehot_labels=onehot_labels, 
                verbose=0
            )

        if buffer_maxlen > 0:
            x_buffer, y_buffer = buffer.sample_buffer_and_prepare_dataset(buffer_sample_num)

            if len(x_buffer) > 0:
                x_train = np.concatenate([x_train, x_buffer], axis=0)
                y_train = np.concatenate([y_train, y_buffer], axis=0)

        if use_vae:
            x_buffer, y_buffer = vae.generate(
                samples_per_class=vae_per_class_num, 
                onehot_labels=onehot_labels, 
                verbose=verbose
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

        plot_history(history, indices=[1])

        if buffer_maxlen > 0:
            buffer.sample_dataset_and_extend_buffer((x_train, y_train), buffer_insert_num)

        if use_vae:
            vae.train(
                x_train, y_train, 
                vae_train_num, 
                validation_data=(x_val, y_val),
                clf=new_model
            )

        if onehot_labels:
            y_test = np.argmax(y_test, axis=-1)

        preds = new_model.predict(x_test)
        preds = np.argmax(preds, axis=-1)

        acc = accuracy_score(y_test, preds)
        acc_list.append(acc)

        if verbose:
            print(classification_report(y_test, preds, digits=4))
            ConfusionMatrixDisplay.from_predictions(y_test, preds)
            plt.show()

            print(75*'-'+'\n')

    CL_plot(class_num, [(acc_list, " ")])

    return acc_list

