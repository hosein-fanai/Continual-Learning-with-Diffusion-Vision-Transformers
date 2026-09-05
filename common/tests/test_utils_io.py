"""Regression checks for numeric sample archives, CSV output, and GIF inputs.

Temporary files exercise ordinary numeric NPY/NPZ storage, explicit trusted legacy-pickle
migration, feature metadata, and output-shape/error contracts. Object arrays and empty image
sequences verify that unsupported input is not silently accepted.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

import numpy as np

from common.utils import create_gif, load_samples, save_samples


class SampleArchiveTests(unittest.TestCase):
    """Verify safe defaults and explicit legacy-pickle migration.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_numeric_npy_round_trip_never_needs_pickle(self) -> None:
        """Keep homogeneous arrays in ordinary non-pickled NPY format.

        Returns:
            None: Numeric values and format magic are asserted.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric"
            save_samples(values, path, ".npy")
            self.assertEqual((path.with_suffix(".npy")).read_bytes()[:4], b"\x93NUM")
            loaded = load_samples(path, ".npy")

        np.testing.assert_array_equal(loaded, values)
        self.assertEqual(loaded.dtype, np.float32)

    def test_heterogeneous_numeric_bundle_uses_safe_ordered_container(self) -> None:
        """Persist unequal numeric feature splits without a pickle payload.

        Returns:
            None: Container format, ordering, shapes, and values are asserted.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        bundle = np.empty(3, dtype=object)
        bundle[0] = np.arange(6, dtype=np.float32).reshape(3, 2)
        bundle[1] = np.arange(2, dtype=np.int16).reshape(1, 2)
        bundle[2] = np.ones((4, 2), dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features"
            save_samples(bundle, path, ".npy")
            self.assertEqual((path.with_suffix(".npy")).read_bytes()[:4], b"PK\x03\x04")
            loaded = load_samples(path, ".npy")

        self.assertEqual(loaded.dtype, object)
        self.assertEqual(loaded.shape, (3,))
        # Member order is the train/validation/test compatibility contract.
        for actual, expected in zip(loaded, bundle):
            np.testing.assert_array_equal(actual, expected)

    def test_legacy_object_npy_requires_opt_in_and_can_be_migrated(self) -> None:
        """Never unpickle by default; warn on trust opt-in and migrate safely.

        Returns:
            None: Rejection, warning, and safe re-save are asserted.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        legacy = np.empty(2, dtype=object)
        legacy[0] = np.asarray([1, 2], dtype=np.int32)
        legacy[1] = np.asarray([3.5], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy"
            with open(legacy_path.with_suffix(".npy"), "wb") as file:
                np.save(file, legacy, allow_pickle=True)

            with self.assertRaisesRegex(ValueError, "disabled"):
                load_samples(legacy_path, ".npy")
            with self.assertWarnsRegex(RuntimeWarning, "execute code"):
                trusted = load_samples(legacy_path, ".npy", allow_pickle=True)

            migrated_path = Path(directory) / "migrated"
            save_samples(trusted, migrated_path, ".npy")
            migrated = load_samples(migrated_path, ".npy")

        # The explicitly trusted legacy data survives conversion member by member.
        for actual, expected in zip(migrated, legacy):
            np.testing.assert_array_equal(actual, expected)

    def test_malformed_or_object_valued_containers_are_rejected(self) -> None:
        """Validate member names and dtypes before returning a safe bundle.

        Returns:
            None: Both malformed container forms raise clear errors.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with tempfile.TemporaryDirectory() as directory:
            named_path = Path(directory) / "named"
            with open(named_path.with_suffix(".npy"), "wb") as file:
                np.savez(file, train=np.ones((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "arr_0"):
                load_samples(named_path, ".npy")

            object_path = Path(directory) / "object-member"
            with open(object_path.with_suffix(".npy"), "wb") as file:
                np.savez(file, np.asarray([{"unsafe": True}], dtype=object))
            with self.assertRaisesRegex(ValueError, "non-object"):
                load_samples(object_path, ".npy")

    def test_invalid_bundle_and_pickle_flag_normalization(self) -> None:
        """Reject nested objects and accept truthy pickle compatibility flags.

        Returns:
            None: Save/load boundary failures are asserted.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        bundle = np.empty(1, dtype=object)
        bundle[0] = np.asarray([object()], dtype=object)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid"
            save_samples(np.ones(1), path, ".npy")
            original_bytes = path.with_suffix(".npy").read_bytes()
            with self.assertRaisesRegex(ValueError, "non-object"):
                save_samples(bundle, path, ".npy")
            self.assertEqual(path.with_suffix(".npy").read_bytes(), original_bytes)
            np.testing.assert_array_equal(
                load_samples(path, ".npy", allow_pickle=1),
                np.ones(1),
            )

    def test_create_gif_rejects_an_empty_frame_sequence(self) -> None:
        """Report the documented error before indexing an absent first frame.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "empty.gif"
            with self.assertRaisesRegex(ValueError, "At least one GIF frame"):
                create_gif(output_path, [], verbose=0)


# Support direct execution in addition to unittest discovery.
if __name__ == "__main__":
    unittest.main()
