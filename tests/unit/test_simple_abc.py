import unittest
import tensorflow as tf
from tests import test_utils


class TestSimpleAbc(unittest.TestCase):
    def setUp(self) -> None:
        self.lr = 0.001
        self.optimizer_params = {"learning_rate": self.lr, "clipvalue": 1}
        self.real_optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)

        base_optimizer_name = (
            self.real_optimizer._name
            if "legacy" in str(self.real_optimizer.__class__)
            else self.real_optimizer.name
        )

        self.custom_optimizer = test_utils.SimpleOptimizer(base_optimizer_name)

    def test_real_optimizer_class(self):
        print(type(self.real_optimizer))
        print(self.real_optimizer)
        print(self.real_optimizer.__class__)
        self.assertIsInstance(self.real_optimizer, tf.keras.optimizers.Optimizer)

    def test_custom_optimizer_class(self):
        print(type(self.custom_optimizer))
        print(self.custom_optimizer)
        print(self.custom_optimizer.__class__)
        self.assertIsInstance(self.custom_optimizer, tf.keras.optimizers.Optimizer)


if __name__ == "__main__":
    unittest.main()
