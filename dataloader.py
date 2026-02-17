def sort_filter_labels(labels_list, indices):
    import numpy as np


    labels_set_list = []
    for labels_set in labels_list:
        allowed_instances_list = []
        for index in indices:
            allowed_instances = np.where(labels_set == index)[0].tolist()
            allowed_instances_list.extend(allowed_instances)

        labels_set_list.append(allowed_instances_list)
    
    return labels_set_list


def get_data(x_train, y_train, x_test, y_test, 
            class_num, indices, preprocess, 
            return_features, features_path, 
            onehot_labels, verbose):
    from tensorflow.keras.utils import to_categorical

    from sklearn.model_selection import train_test_split

    import numpy as np

    from common.utils import load_samples


    if return_features:
        if "cifar10_" in features_path:
            class_num = 10
        elif "cifar100_" in features_path:
            class_num = 100
        else:
            raise Exception("features_path has to contain cifar10_ or cifar100_.")

        labels_set_list = sort_filter_labels([y_train, y_test], list(range(class_num)))
        y_train = y_train[labels_set_list[0]]
        y_test = y_test[labels_set_list[1]]

        y_train, y_val = train_test_split(y_train, test_size=0.2, 
                                        stratify=y_train, random_state=42)

        x_train, x_val, x_test = load_samples(features_path, ".npy")

        x_train = np.concatenate([x_train, x_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)

    labels_set_list = sort_filter_labels([y_train, y_test], indices)
    x_train, y_train = x_train[labels_set_list[0]], y_train[labels_set_list[0]]
    x_test, y_test = x_test[labels_set_list[1]], y_test[labels_set_list[1]]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                    stratify=y_train, random_state=42)

    if preprocess == "min-max":
        min_ = x_train.min()
        max_ = x_train.max()

        x_train = (x_train - min_) / (max_ - min_)
        x_val = (x_val - min_) / (max_ - min_)
        x_test = (x_test - min_) / (max_ - min_)
    elif preprocess == "normalize":
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)

        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
    else: # no preprocess
        if not return_features:
            x_train = x_train.astype("uint8")
            x_test = x_test.astype("uint8")

    if onehot_labels:
        y_train = to_categorical(y_train, num_classes=class_num)
        y_val = to_categorical(y_val, num_classes=class_num)
        y_test = to_categorical(y_test, num_classes=class_num)

    if verbose:
        print("Trainset:", x_train.shape, y_train.shape)
        print("Validation set:", x_val.shape, y_val.shape)
        print("Testset:", x_test.shape, y_test.shape)

        for set_id, dataset in enumerate((y_train, y_val, y_test)):
            print(f"---{set_id}")
            for clss_id in np.unique(dataset):
                print(clss_id, sum(dataset == clss_id) / len(dataset))
            
            print()

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10(indices=list(range(10)), preprocess=None, 
                features_path="./data/cifar10_xception_gavgpooled_features_train_val_test", 
                return_features=False, onehot_labels=False, verbose=1):
    from tensorflow.keras.datasets import cifar10


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 10, 
                indices, preprocess, return_features, 
                features_path, onehot_labels, verbose)


def load_cifar100(indices=list(range(100)), preprocess=None, 
                features_path="./data/cifar100_xception_gavgpooled_features_train_val_test", 
                return_features=False, onehot_labels=False, verbose=1):
    from tensorflow.keras.datasets import cifar100


    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 100,
                indices, preprocess, return_features, 
                features_path, onehot_labels, verbose)


def get_dataset(X, Y, conv_base=None, batch_size=128):
    import tensorflow as tf


    def preprocess_func(x, y):
        if conv_base is not None:
            x = conv_base(x)

        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_func)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()

    return dataset

