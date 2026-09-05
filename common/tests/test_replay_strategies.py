"""Regression checks for FIFO, reservoir, and class-balanced replay policies.

Deterministic streams and local reference recurrences verify retained samples, aligned
labels, sample dtypes, quotas, and restorable RNG/state. Invalid state dictionaries exercise
structural rejection without changing the live buffer.

Inputs are fixtures constructed by the test methods and their helpers. Tests return no
application result: unittest records assertion outcomes and errors. Run this module directly
or through ``python -m unittest`` discovery. Importing it defines fixtures and cases; it
does not itself start a test run.
"""

from __future__ import annotations

from copy import deepcopy

import random
import tempfile
import unittest

import numpy as np

from common.recovery import (
    load_task_checkpoint,
    restore_replay_buffer,
    save_task_checkpoint,
)
from common.replay_buffer import ReplayBuffer


class ReplayStrategyTests(unittest.TestCase):
    """Verify FIFO compatibility and deterministic reservoir policies.

    The unittest runner executes the selected test method with its local fixtures;
    individual methods describe the configurations and failure cases they exercise. There is
    no application model or experiment result returned by constructing this test case.

    Args:
        methodName (str): Test method selected by unittest. Defaults to ``"runTest"``;
            discovery supplies each named ``test_*`` method.

    Attributes:
        _testMethodName (str): Selected method name maintained by unittest.
    """

    def test_fifo_remains_the_exact_default(self) -> None:
        """Default construction retains historical deque and RNG behavior.

        Returns:
            None: FIFO contents and sampling are compared with Python directly.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        replay = ReplayBuffer(maxlen=3, seed=17)
        replay.extend(range(6))
        self.assertEqual(replay.strategy, "fifo")
        self.assertEqual(list(replay.buffer), [3, 4, 5])
        expected_rng = random.Random(17)
        self.assertEqual(
            replay.sample_buffer(2),
            expected_rng.sample([3, 4, 5], 2),
        )

    def test_sampled_sparse_labels_keep_their_dtype(self) -> None:
        """Sparse class IDs above 255 must not wrap during replay sampling.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        replay = ReplayBuffer(maxlen=2, seed=23)
        labels = np.asarray([256, 511], dtype=np.int32)
        replay.extend([
            (np.asarray([1.], dtype=np.float32), labels[0]),
            (np.asarray([2.], dtype=np.float32), labels[1]),
        ])

        sampled_x, sampled_labels = replay.sample_buffer_and_prepare_dataset(3)

        self.assertEqual(sampled_x.dtype, np.float32)
        np.testing.assert_array_equal(sampled_x, [[1.], [2.]])
        self.assertEqual(sampled_labels.dtype, labels.dtype)
        np.testing.assert_array_equal(sampled_labels, labels)

    def test_reservoir_matches_algorithm_r(self) -> None:
        """Reservoir insertion exactly matches the Algorithm R recurrence.

        Returns:
            None: Retained items and deterministic continuation are asserted.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        capacity = 5
        seed = 29
        stream = list(range(40))
        expected = []
        expected_rng = random.Random(seed)
        for seen, item in enumerate(stream, start=1):
            # Fill the reservoir directly until it reaches capacity.
            if len(expected) < capacity:
                expected.append(item)
            # Apply Algorithm R replacement after the initial fill.
            else:
                replacement = expected_rng.randrange(seen)
                # Retain the item only when its sampled slot is in the reservoir.
                if replacement < capacity:
                    expected[replacement] = item

        replay = ReplayBuffer(
            maxlen=capacity,
            seed=seed,
            strategy="reservoir",
        )
        replay.extend(stream)
        self.assertEqual(list(replay.buffer), expected)
        self.assertEqual(replay.state_dict()["items_seen"], len(stream))

    def test_reservoir_state_continues_deterministically(self) -> None:
        """Direct state restoration must reproduce subsequent replacements.

        Args:
            None. The unittest instance owns the fixtures used by this case.

        Returns:
            None: Assertions verify the stated regression; failures are reported to the
            unittest runner.
        """

        source = ReplayBuffer(3, seed=11, strategy="reservoir")
        source.extend(range(20))
        restored = ReplayBuffer(3, strategy="reservoir")
        restored.load_state_dict(source.state_dict())

        source.extend(range(20, 25))
        restored.extend(range(20, 25))
        self.assertEqual(restored.state_dict(), source.state_dict())

    def test_reservoir_has_uniform_long_run_inclusion(self) -> None:
        """Each stream position approaches the common ``capacity / n`` rate.

        Returns:
            None: A broad deterministic frequency bound audits inclusion bias.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        inclusions = np.zeros(20, dtype=np.int32)
        for seed in range(500):
            replay = ReplayBuffer(4, seed=seed, strategy="reservoir")
            replay.extend(range(20))
            inclusions[list(replay.buffer)] += 1
        # The expectation is 100 appearances per item. This fixed, generous
        # interval detects positional bias while remaining non-flaky.
        self.assertTrue(np.all((inclusions >= 70) & (inclusions <= 130)))

    def test_class_balanced_storage_uses_near_equal_quotas(self) -> None:
        """An imbalanced stream occupies equal feasible per-class reservoirs.

        Returns:
            None: Class counts, capacity, aliases, and one-hot parsing are tested.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        replay = ReplayBuffer(
            maxlen=7,
            seed=41,
            strategy="class-balanced",
        )
        stream = [
            ((label, occurrence), np.eye(3, dtype=np.float32)[label])
            for label, count in ((0, 100), (1, 20), (2, 5))
            for occurrence in range(count)
        ]
        replay.extend(stream)
        labels = [int(np.argmax(item[1])) for item in replay.buffer]
        counts = np.bincount(labels, minlength=3)
        self.assertEqual(replay.strategy, "class_balanced")
        self.assertEqual(len(replay), 7)
        self.assertEqual(sorted(counts.tolist()), [2, 2, 3])

        duplicate = ReplayBuffer(7, seed=41, strategy="class_balanced")
        duplicate.extend(stream)
        self.assertEqual(
            [item[0] for item in replay.buffer],
            [item[0] for item in duplicate.buffer],
        )

    def test_more_classes_than_slots_does_not_favor_first_classes(self) -> None:
        """Remainder priorities choose a deterministic subset of classes.

        Returns:
            None: Capacity and one-item-per-selected-class invariants are checked.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        replay = ReplayBuffer(3, seed=53, strategy="class_balanced")
        replay.extend(((label, 0), label) for label in range(10))
        labels = [item[1] for item in replay.buffer]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3)
        self.assertNotEqual(set(labels), {0, 1, 2})

    def test_strategy_state_round_trips_through_task_recovery(self) -> None:
        """Recovery preserves counters, allocation state, RNG, and next updates.

        Returns:
            None: Continued source and restored buffers remain exactly identical.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        source = ReplayBuffer(6, seed=67, strategy="class_balanced")
        source.extend(
            (np.asarray([index], dtype=np.float32), np.uint8(index % 3))
            for index in range(30)
        )
        with tempfile.TemporaryDirectory() as directory:
            save_task_checkpoint(
                directory,
                completed_task_index=0,
                state={
                    "class_order": [0, 1, 2],
                    "task_groups": [[0], [1], [2]],
                },
                replay_buffer=source,
            )
            loaded = load_task_checkpoint(directory)
            restored = ReplayBuffer(6, seed=0, strategy="class_balanced")
            restore_replay_buffer(restored, loaded.replay_state)

            future = [
                (np.asarray([index], dtype=np.float32), np.uint8(index % 3))
                for index in range(30, 55)
            ]
            source.extend(future)
            restored.extend(future)
            self.assertEqual(
                source.state_dict()["items_seen"],
                restored.state_dict()["items_seen"],
            )
            self.assertEqual(
                source.state_dict()["classes"],
                restored.state_dict()["classes"],
            )
            self.assertEqual(
                source.state_dict()["rng_state"],
                restored.state_dict()["rng_state"],
            )
            for expected, actual in zip(source.buffer, restored.buffer):
                np.testing.assert_array_equal(expected[0], actual[0])
                np.testing.assert_array_equal(expected[1], actual[1])

            incompatible = ReplayBuffer(6, seed=0)
            with self.assertRaisesRegex(ValueError, "strategy differs"):
                restore_replay_buffer(incompatible, loaded.replay_state)

    def test_invalid_strategy_and_class_item_fail_clearly(self) -> None:
        """Invalid policies and non-pair balanced items are rejected.

        Returns:
            None: Both public validation branches are exercised.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        with self.assertRaisesRegex(ValueError, "strategy"):
            ReplayBuffer(2, strategy="recent")
        replay = ReplayBuffer(2, strategy="class_balanced")
        with self.assertRaisesRegex(TypeError, "pair"):
            replay.append("not-a-pair")

    def test_restore_normalizes_counts_and_rejects_impossible_state(self) -> None:
        """Normalize counters and reject contents no insertion path can produce.

        Returns:
            None.

        Args:
            None. The unittest instance owns the fixtures used by this case.
        """

        source = ReplayBuffer(4, seed=79, strategy="class_balanced")
        source.extend(
            ((label, occurrence), label)
            for label in (0, 1)
            for occurrence in range(10)
        )
        valid_state = source.state_dict()

        normalized_count = deepcopy(valid_state)
        normalized_count["classes"][0]["seen"] += 0.5
        normalized_target = ReplayBuffer(4, strategy="class_balanced")
        normalized_target.load_state_dict(normalized_count)
        self.assertEqual(
            normalized_target.state_dict()["classes"][0]["seen"],
            valid_state["classes"][0]["seen"],
        )

        impossible_quota = deepcopy(valid_state)
        # Choose a class-zero row to overfill its storage quota.
        zero_item = next(
            item for item in impossible_quota["items"] if item[1] == 0
        )
        # Replace a class-one row so the class-zero quota becomes invalid.
        one_index = next(
            index
            for index, item in enumerate(impossible_quota["items"])
            if item[1] == 1
        )
        impossible_quota["items"][one_index] = zero_item
        with self.assertRaisesRegex(ValueError, "allocation quota"):
            ReplayBuffer(4, strategy="class_balanced").load_state_dict(
                impossible_quota
            )

        unsupported_schema = deepcopy(valid_state)
        unsupported_schema["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema version"):
            ReplayBuffer(4, strategy="class_balanced").load_state_dict(
                unsupported_schema
            )

        target = ReplayBuffer(4, seed=83, strategy="class_balanced")
        target.extend([(("retained", 0), 0), (("retained", 1), 1)])
        target_before = target.state_dict()
        invalid_rng = deepcopy(valid_state)
        invalid_rng["rng_state"] = ("invalid",)
        with self.assertRaisesRegex(ValueError, "RNG state"):
            target.load_state_dict(invalid_rng)
        self.assertEqual(target.state_dict(), target_before)


# Support direct execution in addition to unittest discovery.
if __name__ == "__main__":
    unittest.main()
