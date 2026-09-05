"""Regression checks for component seeds and saved-feature label alignment.

Small callbacks, layers, and model factories expose derived random streams. Synthetic
feature archives verify split metadata, seeded reconstruction, and legacy compatibility
against known label-coded arrays. Temporary artifacts keep persistence checks local to each
test.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, mock

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from autoencoder.decoder_accuracy_callback import DecoderAccuracyCallback
from autoencoder.variational_autoencoder import VariationalAutoencoder
from common.config import Config
from common.dataloader import preprocess_dataset
from common.model import get_model
from common.runtime import configure_runtime, derive_seed
from common.train import report
from common.utils import (
    load_feature_split_metadata,
    save_feature_split_metadata,
    save_samples,
)
from diffusion.layers.block.vision_transformer_block import (
    VisionTransformerBlock,
)
from diffusion.layers.convolution.residual_block import ResidualConvStack
from diffusion.layers.drop_path import DropPath
from diffusion.layers.embedding.patch_embedding import PatchEmbedding
from diffusion.layers.single_token_layer import SingleTokenLayer


class SeedPropagationTests(TestCase):
    """Exercise every newly explicit component seed path.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def setUp(self: SeedPropagationTests) -> None:
        """Install a stable float32 baseline for each test.

        Returns:
            None.
        """

        configure_runtime(seed=101, dtype_policy="float32")

    def test_decoder_callback_uses_distinct_epoch_streams(
        self: SeedPropagationTests,
    ) -> None:
        """Forward stable, non-repeating child seeds into VAE generation.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        generated_seeds: list[int | None] = []

        def generate(
            samples_per_class: int,
            onehot_y_output: bool,
            seed: int | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            """Return class-coded samples while recording the received seed.

            Args:
                samples_per_class (int): Requested samples for each class.
                onehot_y_output (bool): Whether one-hot labels were requested.
                seed (int | None): Derived callback sampling seed. Defaults to ``None``.

            Returns:
                tuple[tf.Tensor, tf.Tensor]: Class-coded samples and labels.
            """

            self.assertEqual(samples_per_class, 1)
            self.assertFalse(onehot_y_output)
            generated_seeds.append(seed)
            labels = tf.constant([0, 1], tf.int64)
            return tf.cast(labels[:, None], tf.float32), labels

        def classifier(inputs: tf.Tensor) -> tf.Tensor:
            """Decode the class ID stored in each sample's first feature.

            Args:
                inputs (tf.Tensor): Class-coded sample tensor.

            Returns:
                tf.Tensor: Two-class one-hot scores.
            """

            return tf.one_hot(tf.cast(inputs[:, 0], tf.int32), depth=2)

        callback = DecoderAccuracyCallback(classifier, 1, seed=37)
        callback.set_model(SimpleNamespace(generate=generate))
        for epoch in (0, 1, 0):
            logs: dict[str, object] = {}
            callback.on_epoch_end(epoch, logs)
            self.assertEqual(float(logs["decoder_accuracy"]), 1.0)

        self.assertEqual(generated_seeds, [
            derive_seed(37, "decoder_accuracy", 0),
            derive_seed(37, "decoder_accuracy", 1),
            derive_seed(37, "decoder_accuracy", 0),
        ])
        self.assertNotEqual(generated_seeds[0], generated_seeds[1])
        self.assertEqual(
            DecoderAccuracyCallback(classifier, seed=True).seed,
            1,
        )

    def test_vae_uses_explicit_reparameterization_and_generation_streams(
        self: SeedPropagationTests,
    ) -> None:
        """Keep latent training and replay sampling reproducible and separate.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        vae = VariationalAutoencoder(
            data_dim=2,
            latent_dim=2,
            hiddens_dims=(),
            last_activation=None,
            compile=False,
            seed=41,
        )
        self.assertEqual(vae.seed, 41)
        self.assertEqual(
            vae.reparameterization_seed,
            derive_seed(41, "vae", "reparameterization"),
        )

        first = vae.generate(samples_per_class=4, seed=43)
        second = vae.generate(samples_per_class=4, seed=43)
        np.testing.assert_array_equal(first, second)

        means = tf.zeros((3, 2), tf.float32)
        log_variances = tf.zeros_like(means)
        tf.random.set_seed(47)
        first_z = VariationalAutoencoder.compute_z(
            means,
            log_variances,
            seed=53,
        )
        tf.random.set_seed(47)
        second_z = VariationalAutoencoder.compute_z(
            means,
            log_variances,
            seed=53,
        )
        tf.debugging.assert_equal(first_z, second_z)

    def test_seeded_stochastic_layers_serialize_child_streams(
        self: SeedPropagationTests,
    ) -> None:
        """Preserve initializer/dropout seeds through nested layer configs.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        first_token = SingleTokenLayer(
            dim=4,
            with_pos_embed=False,
            seed=59,
        )
        second_token = SingleTokenLayer(
            dim=4,
            with_pos_embed=False,
            seed=59,
        )
        tf.debugging.assert_equal(first_token.token, second_token.token)
        self.assertEqual(first_token.get_config()["seed"], 59)

        drop_path = DropPath(drop_prob=0.25, seed=61)
        self.assertEqual(drop_path.get_config()["seed"], 61)

        block = VisionTransformerBlock(
            dim=4,
            num_heads=1,
            mlp_ratio=1.0,
            drop_prob=0.25,
            seed=67,
        )
        self.assertEqual(
            block.mha_drop_path.seed,
            derive_seed(67, "mha_drop_path"),
        )
        self.assertEqual(
            block.mlp_drop_path.seed,
            derive_seed(67, "mlp_drop_path"),
        )

        patches = PatchEmbedding(
            dim=4,
            grid_size=2,
            patch_size=2,
            shift_right_token=True,
            seed=71,
        )
        self.assertEqual(
            patches.shift_right_token.seed,
            derive_seed(71, "bos_token"),
        )

        stack = ResidualConvStack(
            filters=4,
            depth=2,
            dropout_rate=0.25,
            seed=73,
        )
        self.assertEqual(stack.blocks[0].seed, derive_seed(73, "block", 0))
        self.assertEqual(
            stack.blocks[0].dropout.seed,
            derive_seed(stack.blocks[0].seed, "spatial_dropout"),
        )
        self.assertNotEqual(stack.blocks[0].seed, stack.blocks[1].seed)

    def test_factory_replaces_none_with_effective_seed(
        self: SeedPropagationTests,
    ) -> None:
        """Route runtime seeds into legacy Dropout and typed raw networks.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        classifier = get_model(
            3,
            model_type="DNN",
            dropout_rate=0.25,
            seed=79,
            verbose=0,
        )
        # Inspect the Dropout layer, which owns this stochastic seed.
        dropout = next(
            layer
            for layer in classifier.layers
            if isinstance(layer, tf.keras.layers.Dropout)
        )
        self.assertEqual(
            dropout.seed,
            derive_seed(79, "classifier", "dnn", "legacy", "dropout"),
        )

        config = Config()
        config.training.task = "generation"
        config.training.seed = 83
        config.dataset.trainset_len = 100
        config.model.name = "diffusion_transformer"
        config.model.diffusion_transformer.timesteps = 4
        config.model.diffusion_transformer.dim = 4
        config.model.diffusion_transformer.cond_dim = 4
        config.model.diffusion_transformer.depth = 0
        config.model.diffusion_transformer.build = False
        configured = get_model(config)
        self.assertEqual(configured.seed, 83)
        self.assertEqual(configured.network.seed, 83)

    def test_final_vae_report_passes_derived_seed(
        self: SeedPropagationTests,
    ) -> None:
        """Make final sample images repeatable without consuming training RNG.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        vae = VariationalAutoencoder(
            data_dim=28 * 28,
            latent_dim=2,
            hiddens_dims=(),
            compile=False,
            seed=89,
        )
        generated = np.zeros((10, 28 * 28), np.float32)
        with mock.patch.object(
            vae,
            "generate",
            return_value=generated,
        ) as generate, mock.patch("common.train.plot_images"):
            report(
                history={"loss": [1.0]},
                model=vae,
                trainset=object(),
                run_trainset_eval=False,
                run_valset_eval=False,
                show_history_plot=False,
                show_final_images=True,
                save_final_images=False,
                dataset_name="mnist",
                seed=97,
                verbose=0,
            )

        self.assertEqual(
            generate.call_args.kwargs["seed"],
            derive_seed(97, "final_report", "vae_generation"),
        )


class FeatureSplitMetadataTests(TestCase):
    """Verify new metadata and legacy feature archives remain label-aligned.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    @staticmethod
    def _save_archive(path: Path, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Write a tiny label-coded train/validation/test feature archive.

        Args:
            path (pathlib.Path): Archive base path without an extension.
            split_seed (int): Stratified train/validation split seed.

        Returns:
            tuple[np.ndarray, np.ndarray]: Original training and test labels.
        """

        y_train = np.repeat(np.arange(2), 10)
        train_ids, validation_ids = train_test_split(
            np.arange(len(y_train)),
            test_size=0.2,
            stratify=y_train,
            random_state=split_seed,
        )
        y_test = np.asarray([0, 1, 0, 1])
        features = np.empty(3, dtype=object)
        features[0] = np.column_stack((train_ids, y_train[train_ids])).astype(
            np.float32
        )
        features[1] = np.column_stack((
            validation_ids,
            y_train[validation_ids],
        )).astype(np.float32)
        # Saved feature archives follow the loader's class-grouped test order.
        test_order = np.concatenate([
            np.flatnonzero(y_test == class_id)
            for class_id in np.unique(y_test)
        ])
        features[2] = np.column_stack((
            test_order,
            y_test[test_order],
        )).astype(np.float32)
        save_samples(features, path, ".npy")
        return y_train, y_test

    def _assert_archive_alignment(
        self: FeatureSplitMetadataTests,
        path: Path,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Load an archive and assert that every feature retains its label.

        Args:
            path (pathlib.Path): Archive base path without an extension.
            y_train (np.ndarray): Original training labels.
            y_test (np.ndarray): Original test labels.

        Returns:
            None.
        """

        prepared = preprocess_dataset(
            np.empty((len(y_train), 1), np.float32),
            y_train,
            np.empty((len(y_test), 1), np.float32),
            y_test,
            class_num=10,
            indices=[0, 1],
            validation_ratio=0.0,
            preprocess=None,
            return_features=True,
            features_path=str(path),
            onehot_labels=False,
            seed=999,
            verbose=False,
        )
        x_train, returned_y_train, _, _, x_test, returned_y_test = prepared
        np.testing.assert_array_equal(x_train[:, 1], returned_y_train)
        np.testing.assert_array_equal(x_test[:, 1], returned_y_test)

    def test_metadata_seed_controls_label_reconstruction(
        self: FeatureSplitMetadataTests,
    ) -> None:
        """Read a nonlegacy split seed from the JSON archive sidecar.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mnist_seeded_features"
            y_train, y_test = self._save_archive(path, split_seed=7)
            metadata_path = save_feature_split_metadata(path, 7, 0.2)
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(load_feature_split_metadata(path), (7, 0.2))
            self._assert_archive_alignment(path, y_train, y_test)

    def test_legacy_archive_retains_seed_42_alignment(
        self: FeatureSplitMetadataTests,
    ) -> None:
        """Keep existing NPY-only archives readable without relabeling rows.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mnist_legacy_features"
            y_train, y_test = self._save_archive(path, split_seed=42)
            self.assertIsNone(load_feature_split_metadata(path))
            self._assert_archive_alignment(path, y_train, y_test)


# Execute only this focused suite when invoked directly.
if __name__ == "__main__":
    main()
