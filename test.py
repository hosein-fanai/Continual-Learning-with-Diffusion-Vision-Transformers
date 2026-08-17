
PROJECT_SELF_TEST_CLASSES = {
    "autoencoder.classifier_variational_autoencoder": ("ClassifierVAE",), 
    "autoencoder.decoder_accuracy_callback": ("DecoderAccuracyCallback",), 
    "autoencoder.variational_autoencoder": ("VariationalAutoencoder",), 
    "common.argument_saver": (
        "ArgumentSaver", 
        "ArgumentSaverLayer", 
        "ArgumentSaverModel", 
    ), 
    "common.config": (
        "KwargsMixin", 
        "DiffusionTransformerConfig", 
        "DiTClassifierConfig", 
        "DiffusionModelConfig", 
        "DiffusionClassifierConfig", 
        "DatasetConfig", 
        "ModelConfig", 
        "OptimizerConfig", 
        "TrainingConfig", 
        "ReportingConfig", 
        "Config", 
    ), 
    "common.lr_logger_callback": ("LrLoggerCallback",), 
    "common.masked_loss": ("MaskedLoss",), 
    "common.replay_buffer": ("ReplayBuffer",), 
    "diffusion.callbacks.batch_loss_plateau": ("BatchLossPlateau",), 
    "diffusion.callbacks.image_generator_callback": ("ImageGeneratorCallback",), 
    "diffusion.callbacks.raw_network_validation_callback": (
        "RawNetworkValidationCallback", 
    ), 
    "diffusion.layers.adaptive_layer_normalization_zero": ("AdaLNZero",), 
    "diffusion.layers.base_layer": ("BaseLayer",), 
    "diffusion.layers.block.di_t_decoder_block": ("DiTDecoderBlock",), 
    "diffusion.layers.block.vision_transformer_block": (
        "VisionTransformerBlock", 
    ), 
    "diffusion.layers.convolution.downsample": ("ImageDownsample",), 
    "diffusion.layers.convolution.residual_block": (
        "ResidualConvBlock", 
        "ResidualConvStack", 
    ), 
    "diffusion.layers.convolution.stage": ("LayerDict",), 
    "diffusion.layers.convolution.upsample": ("ImageUpsample",), 
    "diffusion.layers.convolution.variational_reshaper": (
        "VariationalReshaper", 
    ), 
    "diffusion.layers.drop_path": ("DropPath",), 
    "diffusion.layers.embedding.base_embedding": ("BaseEmbedding",), 
    "diffusion.layers.embedding.condition_embedding": ("ConditionEmbedding",), 
    "diffusion.layers.embedding.patch_embedding": ("PatchEmbedding",), 
    "diffusion.layers.feature_handler": ("FeatureHandler",), 
    "diffusion.layers.manipulation.downsample": ("Downsample",), 
    "diffusion.layers.manipulation.local_mixer": ("LocalMixer",), 
    "diffusion.layers.manipulation.upsample": ("Upsample",), 
    "diffusion.layers.single_token_layer": ("SingleTokenLayer",), 
    "diffusion.metrics.ensemble_accuracy": ("EnsembleAccuracy",), 
    "diffusion.models.convolution.unet": ("UNet",), 
    "diffusion.models.convolution.unet_classifier": ("UNetClassifier",), 
    "diffusion.models.transformer.di_t_classifier": ("DiTClassifier",), 
    "diffusion.models.transformer.di_t_decoder": ("DiTDecoder",), 
    "diffusion.models.transformer.di_t_encoder_decoder": (
        "DiTEncoderDecoder", 
    ), 
    "diffusion.models.transformer.di_t_encoder_decoder_classifier": (
        "DiTEncoderDecoderClassifier", 
    ), 
    "diffusion.models.transformer.diffusion_transformer": (
        "DiffusionTransformer", 
    ), 
    "diffusion.models.wrapper.diffusion_classifier": ("DiffusionClassifier",), 
    "diffusion.models.wrapper.diffusion_classifier_v2": (
        "DiffusionClassifierV2", 
    ), 
    "diffusion.models.wrapper.diffusion_encoder_decoder_model": (
        "DiffusionEncoderDecoderModel", 
    ), 
    "diffusion.models.wrapper.diffusion_model": ("DiffusionModel",), 
    "diffusion.old.schedulers": ("ScheduleKind", "ScheduleConfig"), 
    "diffusion.schedulers": ("ScheduleKind", "ScheduleConfig"), 
}
"""Classes that must be covered by each module's ``run_self_tests`` result."""


def run_project_self_tests(
    verbose: bool = True, 
) -> dict[str, dict[str, str]]:
    """Run and coverage-audit every class self-test in the repository.

    Each registered module is imported and its ``run_self_tests`` function is
    called in-process.  Before accepting a result, this utility discovers the
    classes whose ``__module__`` matches that module and compares them with the
    fixed registry.  It then requires the self-test result to contain exactly
    those class names and the value ``"passed"`` for each one.  Consequently a
    newly added or silently omitted class makes the project check fail instead
    of producing an incomplete success message.

    The runner deliberately continues after ordinary exceptions so one call
    reports every failing module.  Python, NumPy, and TensorFlow random seeds
    are reset before each module; Keras state and garbage are cleared after
    each module to keep the full suite deterministic and memory-efficient.

    Args:
        verbose (bool): Print one PASS/FAIL line per module and a final class
            count.  ``False`` suppresses progress output but does not suppress
            exceptions.

    Returns:
        dict[str, dict[str, str]]: Ordered-by-registration module results.  A
        successful result covers all 58 registered classes and every inner
        value is ``"passed"``.

    Raises:
        AssertionError: After all modules have run if a module is missing its
            runner, defined-class coverage differs from the registry, a result
            has missing/extra/non-passing entries, or any self-test raises.
    """

    import tensorflow as tf

    import numpy as np

    import gc
    import importlib
    import inspect
    import random
    import time
    import traceback


    results = {}
    failures = {}
    started = time.perf_counter()

    for module_name, expected_names_tuple in PROJECT_SELF_TEST_CLASSES.items():
        module_started = time.perf_counter()
        expected_names = set(expected_names_tuple)
        try:
            random.seed(1729)
            np.random.seed(1729)
            tf.random.set_seed(1729)

            module = importlib.import_module(module_name)
            defined_names = {
                name
                for name, value in vars(module).items()
                if inspect.isclass(value) and value.__module__ == module_name
            }
            assert defined_names == expected_names, (
                f"Class registry mismatch for {module_name}: defined="
                f"{sorted(defined_names)}, expected={sorted(expected_names)}"
            )

            runner = getattr(module, "run_self_tests", None)
            assert callable(runner), f"{module_name} has no callable run_self_tests"

            module_result = runner()
            assert isinstance(module_result, dict), (
                f"{module_name}.run_self_tests() returned "
                f"{type(module_result).__name__}, not dict"
            )
            assert set(module_result) == expected_names, (
                f"Self-test coverage mismatch for {module_name}: reported="
                f"{sorted(module_result)}, expected={sorted(expected_names)}"
            )
            assert all(value == "passed" for value in module_result.values()), (
                f"Non-passing self-test result from {module_name}: {module_result}"
            )
            results[module_name] = module_result

            if verbose:
                elapsed = time.perf_counter() - module_started
                print(
                    f"[PASS] {module_name}: {len(module_result)} "
                    f"class(es), {elapsed:.3f}s"
                )
        except Exception as error:
            failures[module_name] = {
                "error": f"{type(error).__name__}: {error}", 
                "traceback": traceback.format_exc(), 
            }

            if verbose:
                elapsed = time.perf_counter() - module_started
                print(
                    f"[FAIL] {module_name}: {type(error).__name__}: "
                    f"{error} ({elapsed:.3f}s)"
                )
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    if failures:
        details = "\n\n".join(
            f"{module_name}\n{failure['traceback']}"
            for module_name, failure in failures.items()
        )
        raise AssertionError(
            f"{len(failures)} of {len(PROJECT_SELF_TEST_CLASSES)} module "
            f"self-test suites failed:\n\n{details}"
        )

    tested_classes = sum(len(module_result) for module_result in results.values())
    expected_classes = sum(
        len(class_names) for class_names in PROJECT_SELF_TEST_CLASSES.values()
    )
    assert tested_classes == expected_classes == 58

    if verbose:
        elapsed = time.perf_counter() - started
        print(
            f"[PASS] Project self-tests: {tested_classes} classes across "
            f"{len(results)} modules in {elapsed:.3f}s; safe and sound."
        )

    return results


if __name__ == "__main__":
    run_project_self_tests()
