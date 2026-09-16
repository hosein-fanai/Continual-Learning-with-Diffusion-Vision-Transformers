"""Model accounting must not require the optional gist-memory research route."""

import unittest

import tensorflow as tf

from common.tensor_inventory import tensor_inventory


class TensorInventoryTests(unittest.TestCase):
    def test_shared_tensors_count_once_while_equal_independent_copies_count_twice(self):
        first = tf.Variable([1., 2., 3.], dtype=tf.float32)
        copy = tf.Variable([1., 2., 3.], dtype=tf.float32)
        step = tf.Variable(2, dtype=tf.int64)
        result = tensor_inventory({"raw": [first, first], "teacher": [copy],
                                   "optimizer": [step, first]})
        self.assertEqual(result["unique_tensor_bytes"], 32)
        self.assertEqual(result["unique_tensor_count"], 3)
        self.assertEqual(result["tensor_reference_count"], 5)
        self.assertEqual(result["shared_tensor_bytes"], 12)
        self.assertEqual(result["groups"]["optimizer"]["tensor_bytes"], 20)
        self.assertEqual(result["tensors"][0]["groups"], ["raw", "optimizer"])

    def test_keras_variables_and_empty_groups_keep_inventory_schema(self):
        layer = tf.keras.layers.Dense(3)
        layer(tf.zeros((1, 2)))
        result = tensor_inventory({"network": layer.weights, "absent_teacher": []})
        self.assertEqual(result["unique_tensor_bytes"], 36)
        self.assertEqual([entry["shape"] for entry in result["tensors"]], [[2, 3], [3]])
        self.assertEqual(result["groups"]["absent_teacher"], {"tensor_count": 0, "tensor_bytes": 0})


if __name__ == "__main__":
    unittest.main()
