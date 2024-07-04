import json
import math
import unittest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from keras.optimizers import Optimizer as KerasBaseOptimizer
from ga_optimizer import make_ga_optimizer, Optimizer, Adam
from ga_optimizer import Optimizer as Ga_Optimizer
from ga_optimizer.utils.utils import (
    os_is_mac,
    is_tf_211_and_above,
    optimizer_has_legacy,
)


class TestGAOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.metrics = [tf.keras.metrics.BinaryAccuracy()]
        self.batch_size = 8
        self.steps_per_epoch = 100
        self.epochs = 5
        self.val_batch_size = self.batch_size
        self.val_batches = math.floor(self.steps_per_epoch / 4)
        self.lr = 0.001
        self.desired_batch_size = 64
        self.optimizer_params = {"learning_rate": self.lr, "clipvalue": 1}

        # Tested on MacOS with TF 2.13.0
        # self.base_optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)

        # Should work on older versions of TF by using the correct import path
        # Tested on MacOS with TF 2.13.0

        # Check if TF version is 2.11 or above
        if is_tf_211_and_above():
            # Check if the legacy Adam optimizer is available and OS is MacOS
            if optimizer_has_legacy() and os_is_mac():
                self.base_optimizer = tf.keras.optimizers.legacy.Adam(
                    **self.optimizer_params
                )
                print(
                    f"Test setup: Using legacy Adam optimizer: {self.base_optimizer.__class__}"
                )
            else:
                self.base_optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)
        else:
            self.base_optimizer = optimizers.Adam(**self.optimizer_params)

        if optimizer_has_legacy() and os_is_mac() and is_tf_211_and_above():
            self.expected_optimizer_type = keras.optimizers.legacy.Adam
            self.expected_base_class = keras.optimizers.legacy.Optimizer
        else:
            self.expected_optimizer_type = keras.optimizers.Adam
            self.expected_base_class = keras.optimizers.Optimizer

    def test_make_compile_adam_optimizer(self):
        """This test creates an Adam object directly from GA class.

        As long as GA.get_config() returns "steps", this will not succeed. If "steps" is not returned however, it will just create a base optimizer, sidestepping the GA wrapper.
        """
        # This will sidestep the GA wrapper. Not advised method.
        adam_optimizer = Adam(steps=10, **self.optimizer_params)

        # Print optimizer's attributes
        print(f"Type of adam_optimizer: {type(adam_optimizer)}")
        print(f"adam_optimizer: {adam_optimizer}")
        # print(f"adam_optimizer attributes: {dir(adam_optimizer)}")
        print(f"adam_optimizer.optimizer attributes: {dir(adam_optimizer.optimizer)}")

        with self.assertRaises(TypeError) as context:
            self.model.compile(
                optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics
            )

        self.assertIn(
            "Unexpected keyword argument passed to optimizer", str(context.exception)
        )

        return

        print(f"Model after compile: {self.model}")
        # print(f"Model config: {json.dumps(self.model.get_config(), indent=4)}")

        print(f"Type of self.model.optimizer: {type(self.model.optimizer)}")
        print(f"self.model.optimizer: {self.model.optimizer}")

        if not isinstance(self.model.optimizer, self.expected_optimizer_type):
            print(
                "Error: self.model.optimizer is not an instance of tf.keras.optimizers.Optimizer"
            )

        # This should be the custom optimizer class, not the Keras one
        self.assertIsInstance(adam_optimizer, Optimizer)

        # Assert that the custom optimizer is an instance of the Keras base optimizer class because it indicates that the custom optimizer adheres to the interface defined by the Keras base class
        self.assertIsInstance(self.model.optimizer, self.expected_optimizer_type)
        self.assertIsInstance(adam_optimizer, self.expected_base_class)

        # Check if the wrapped optimizer is an instance of Adam
        self.assertIsInstance(
            self.model.optimizer.optimizer, self.expected_optimizer_type
        )

        self.assertEqual(type(self.model.optimizer), self.expected_optimizer_type)

    def test_make_ga_optimizer(self):
        print(f"Type of base_optimizer: {type(self.base_optimizer)}")
        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=self.base_optimizer,
            base_optimizer_params=self.optimizer_params,
            log_level=Ga_Optimizer.LOG_DEBUG,
        )
        ga_params: dict = ga_optimizer.get_config()
        print(ga_params)

        # Assert ga_optimizer got the expected params
        self.assertEqual(
            ga_params.get("learning_rate"), self.optimizer_params.get("learning_rate")
        )
        self.assertEqual(
            ga_params.get("clipvalue"), self.optimizer_params.get("clipvalue")
        )
        print(f"Type of optimizer before compile: {type(ga_optimizer)}")
        print(
            f"Accumulating gradients over {ga_optimizer.steps} steps/batches to simulate an effective batch size of {self.desired_batch_size} with an actual batch size of {self.batch_size}."
        )
        print(f"Type of optimizer: {type(ga_optimizer)}")
        ga_optimizer_super_class = ga_optimizer.__class__.__bases__
        print(f"Optimizer's super class: {ga_optimizer_super_class}")

        self.model.compile(optimizer=ga_optimizer, loss=self.loss, metrics=self.metrics)

        if not isinstance(self.model.optimizer, self.expected_optimizer_type):
            print(
                f"Error: self.model.optimizer is not an instance of {self.expected_optimizer_type}"
            )

        self.assertIsInstance(self.model.optimizer, Ga_Optimizer)

        for super_class in ga_optimizer_super_class:
            self.assertTrue(
                issubclass(super_class, keras.optimizers.Optimizer),
                f"{super_class} is not a subclass of keras.optimizers.Optimizer",
            )

        # Check if the wrapped optimizer is an instance of Adam
        print("self.model.optimizer.optimizer:", self.model.optimizer.optimizer)
        self.assertIsInstance(
            self.model.optimizer.optimizer, self.expected_optimizer_type
        )

        self.assertEqual(type(self.model.optimizer), type(ga_optimizer))
        self.assertEqual(self.model.optimizer.get_config(), ga_optimizer.get_config())

        self.assertEqual(
            float("{0:.4f}".format(ga_optimizer.get_config()["learning_rate"])), self.lr
        )

    def test_ga_optimizer_accumulation_steps(self):
        optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=self.base_optimizer,
        )
        # Check that the optimizer has the correct number of accumulation steps (// is int division)
        self.assertEqual(optimizer.steps, self.desired_batch_size // self.batch_size)

    def test_model_compile_with_base_optimizer(self):
        print(f"Type of optimizer before compile: {type(self.base_optimizer)}")

        # Check if optimizer has the necessary attributes and methods
        if not hasattr(self.base_optimizer, "apply_gradients"):
            print("Optimizer is missing 'apply_gradients' method")

        self.model.compile(
            optimizer=self.base_optimizer, loss=self.loss, metrics=self.metrics
        )

        print(f"Model after compile: {self.model}")
        # print(f"Model config: {json.dumps(self.model.get_config(), indent=4)}")

        print(f"Type of self.model.optimizer: {type(self.model.optimizer)}")
        print(f"self.model.optimizer: {self.model.optimizer}")

        if not isinstance(self.model.optimizer, tf.keras.optimizers.Optimizer):
            print(
                "Error: self.model.optimizer is not an instance of tf.keras.optimizers.Optimizer"
            )

        # Check and print the optimizer types and configurations
        print(f"self.model.optimizer: {self.model.optimizer}")
        print(f"optimizer: {self.base_optimizer}")
        print(
            f"self.model.optimizer class name: {self.model.optimizer.__class__.__name__}"
        )
        print(f"optimizer class name: {self.base_optimizer.__class__.__name__}")
        print(f"self.model.optimizer.get_config(): {self.model.optimizer.get_config()}")
        print(f"optimizer.get_config(): {self.base_optimizer.get_config()}")
        self.model.compile(
            optimizer=self.base_optimizer, loss=self.loss, metrics=self.metrics
        )
        self.assertEqual(type(self.model.optimizer), type(self.base_optimizer))
        self.assertEqual(
            self.model.optimizer.get_config(), self.base_optimizer.get_config()
        )

    def test_model_compile_with_ga_optimizer_optimizer_tf_keras(self):
        """This test uses tf.keras.optimizers to make a ga_optimizer.
        It then tests if the model can compile with ga_optimizer.optimizer.
        """
        # Check if TF version is 2.11 or above
        if is_tf_211_and_above():
            # Check if the legacy Adam optimizer is available and OS is MacOS
            if optimizer_has_legacy() and os_is_mac():
                base_optimizer = tf.keras.optimizers.legacy.Adam(
                    **self.optimizer_params
                )
            else:
                base_optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)
        else:
            # This test is specific for tf.keras and not guaranteed to work if path is anything else
            base_optimizer = optimizers.Adam(**self.optimizer_params)
        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=base_optimizer,
        )
        print(f"Type of optimizer before compile: {type(ga_optimizer.optimizer)}")

        # Check if optimizer has the necessary attributes and methods
        if not hasattr(ga_optimizer, "apply_gradients"):
            print("Optimizer is missing 'apply_gradients' method")

        self.model.compile(
            optimizer=ga_optimizer.optimizer, loss=self.loss, metrics=self.metrics
        )

        print(f"Model after compile: {self.model}")
        print(f"Model config: {json.dumps(self.model.get_config(), indent=4)}")

        print(f"Type of self.model.optimizer: {type(self.model.optimizer)}")
        print(f"self.model.optimizer: {self.model.optimizer}")

        if not isinstance(self.model.optimizer, tf.keras.optimizers.Optimizer):
            print(
                "Error: self.model.optimizer is not an instance of tf.keras.optimizers.Optimizer"
            )

        # Check and print the optimizer types and configurations
        print(f"self.model.optimizer: {self.model.optimizer}")
        print(f"optimizer: {ga_optimizer.optimizer}")
        print(
            f"self.model.optimizer class name: {self.model.optimizer.__class__.__name__}"
        )
        print(f"optimizer class name: {ga_optimizer.optimizer.__class__.__name__}")
        print(f"self.model.optimizer.get_config(): {self.model.optimizer.get_config()}")
        print(f"optimizer.get_config(): {ga_optimizer.optimizer.get_config()}")

        self.assertEqual(type(self.model.optimizer), type(ga_optimizer.optimizer))
        self.assertEqual(
            self.model.optimizer.get_config(), ga_optimizer.optimizer.get_config()
        )

    def test_model_compile_with_local_ga_optimizer_tensorflow_keras(self):
        """This test uses tensorflow.keras.optimizers to make a ga_optimizer.
        It then passes the ga_optimizer directly into the model.
        """
        optimizer_params = {"learning_rate": 0.001, "clipvalue": 1}
        base_optimizer = optimizers.Adam(**optimizer_params)
        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=base_optimizer,
            base_optimizer_params=optimizer_params,
        )

        ga_params: dict = ga_optimizer.get_config()
        print(ga_params)

        # Assert ga_optimizer got the expected params
        self.assertEqual(ga_params.get("learning_rate"), 0.001)
        self.assertEqual(ga_params.get("clipvalue"), 1)
        print(f"Type of optimizer before compile: {type(ga_optimizer)}")

        # Check if optimizer has the necessary attributes and methods
        if not hasattr(ga_optimizer, "apply_gradients"):
            print("Optimizer is missing 'apply_gradients' method")

        self.model.compile(optimizer=ga_optimizer, loss=self.loss, metrics=self.metrics)

        print(f"Model after compile: {self.model}")
        print(f"Model config: {json.dumps(self.model.get_config(), indent=4)}")

        print(f"Type of self.model.optimizer: {type(self.model.optimizer)}")
        print(f"self.model.optimizer: {self.model.optimizer}")

        if not isinstance(self.model.optimizer, tf.keras.optimizers.Optimizer):
            print(
                "Error: self.model.optimizer is not an instance of tf.keras.optimizers.Optimizer"
            )

        # Check and print the optimizer types and configurations
        print(f"self.model.optimizer: {self.model.optimizer}")
        print(f"optimizer: {ga_optimizer}")
        print(
            f"self.model.optimizer class name: {self.model.optimizer.__class__.__name__}"
        )
        print(f"optimizer class name: {ga_optimizer.__class__.__name__}")
        if hasattr(self.model.optimizer, "get_config"):
            print(
                f"self.model.optimizer.get_config(): {self.model.optimizer.get_config()}"
            )
        print(f"optimizer.get_config(): {ga_optimizer.get_config()}")

        self.assertEqual(type(self.model.optimizer), type(ga_optimizer))
        self.assertEqual(self.model.optimizer.get_config(), ga_optimizer.get_config())

    def test_model_compile_with_ga_optimizer_optimizer_tensorflow_keras(self):
        """This test uses tensorflow.keras.optimizers to make a ga_optimizer.
        It then tests if model can compile with ga_optimizer.optimizer.
        """
        # Check if TF version is 2.11 or above
        if is_tf_211_and_above():
            # Check if the legacy Adam optimizer is available and OS is MacOS
            if optimizer_has_legacy() and os_is_mac():
                base_optimizer = optimizers.legacy.Adam(**self.optimizer_params)
            else:
                base_optimizer = optimizers.Adam(**self.optimizer_params)
        else:
            base_optimizer = optimizers.Adam(**self.optimizer_params)

        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=base_optimizer,
        )
        print(f"Type of optimizer before compile: {type(ga_optimizer.optimizer)}")

        # Check if optimizer has the necessary attributes and methods
        if not hasattr(ga_optimizer, "apply_gradients"):
            print("Optimizer is missing 'apply_gradients' method")

        self.model.compile(
            optimizer=ga_optimizer.optimizer, loss=self.loss, metrics=self.metrics
        )

        print(f"Model after compile: {self.model}")
        print(f"Model config: {json.dumps(self.model.get_config(), indent=4)}")

        print(f"Type of self.model.optimizer: {type(self.model.optimizer)}")
        print(f"self.model.optimizer: {self.model.optimizer}")

        if not isinstance(self.model.optimizer, tf.keras.optimizers.Optimizer):
            print(
                "Error: self.model.optimizer is not an instance of tf.keras.optimizers.Optimizer"
            )

        # Check and print the optimizer types and configurations
        print(f"self.model.optimizer: {self.model.optimizer}")
        print(f"optimizer: {ga_optimizer.optimizer}")
        print(
            f"self.model.optimizer class name: {self.model.optimizer.__class__.__name__}"
        )
        print(f"optimizer class name: {ga_optimizer.optimizer.__class__.__name__}")
        print(f"self.model.optimizer.get_config(): {self.model.optimizer.get_config()}")
        print(f"optimizer.get_config(): {ga_optimizer.optimizer.get_config()}")

        self.assertEqual(type(self.model.optimizer), type(ga_optimizer.optimizer))
        self.assertEqual(
            self.model.optimizer.get_config(), ga_optimizer.optimizer.get_config()
        )

    def test_train_static_dummy_data_with_ga_optimizer(self):
        """Test training a model using a GA optimizer with static dummy data.

        This test uses the self.base_optimizer to make a ga_optimizer.
        It then passes the ga_optimizer directly into the model.

        The dummy data is created once and used directly for training.
        No validation dataset is used in this test.
        """
        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=self.base_optimizer,
        )
        self.model.compile(optimizer=ga_optimizer, loss=self.loss, metrics=self.metrics)

        # Dummy data
        x = tf.random.uniform(
            (self.batch_size * 4, 10)
        )  # Create enough data for multiple steps
        y = tf.random.uniform((self.batch_size * 4, 1))

        self.assertIsInstance(self.model, tf.keras.Model)
        assert isinstance(self.model, tf.keras.Model)

        # Train the model for a few epochs
        history: tf.keras.callbacks.History = self.model.fit(
            x, y, epochs=3, batch_size=self.batch_size
        )

        # Verify the training history
        self.assertIn("loss", history.history)
        self.assertIn("binary_accuracy", history.history)
        self.assertIsInstance(history, tf.keras.callbacks.History)
        assert isinstance(history, tf.keras.callbacks.History)
        self.assertGreater(len(history.history["loss"]), 0)

        # Verify the effective batch size
        effective_batch_size = self.batch_size * ga_optimizer.steps
        self.assertEqual(effective_batch_size, self.desired_batch_size)

    def test_train_dynamic_dummy_data_with_ga_optimizer_self_base(self):
        """Test training a model using a GA optimizer with dynamically generated dummy data.

        This test uses the self.base_optimizer to make a ga_optimizer.
        It then passes the ga_optimizer directly into the model.

        What separates this test from test_train_static_data_with_ga_optimizer is the way data is handled and fed into the model.
        In this test, we use a TensorFlow data generator to create training and validation datasets dynamically during training,
        whereas in the other test, static dummy data is used directly for training without a validation step.
        """

        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=self.base_optimizer,
            log_level=Optimizer.LOG_PARANOID,
        )
        self.model.compile(optimizer=ga_optimizer, loss=self.loss, metrics=self.metrics)

        print(f"GAOptimizer class: {Optimizer}")
        print(f"GAOptimizer bases: {Optimizer.__bases__}")
        print(f"Instance type: {type(ga_optimizer)}")
        print(
            f"Is instance of KerasBaseOptimizer: {isinstance(ga_optimizer, KerasBaseOptimizer)}"
        )
        print(
            f"Is instance of tf.keras.optimizers.Adam: {isinstance(ga_optimizer, tf.keras.optimizers.Adam)}"
        )

        # Create dummy datasets
        def dummy_data_generator(batch_size):
            while True:
                x = tf.random.uniform((batch_size, 10))
                y = tf.random.uniform((batch_size, 1))
                yield x, y

        dummy_training_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        validation_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.val_batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.val_batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.val_batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        # Train the model with the dummy dataset
        history = self.model.fit(
            dummy_training_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=self.val_batches,
        )

        # Verify the training history
        self.assertIn("loss", history.history)
        self.assertIn("binary_accuracy", history.history)
        self.assertGreater(len(history.history["loss"]), 0)

        # Verify the effective batch size
        effective_batch_size = self.batch_size * ga_optimizer.steps
        self.assertEqual(effective_batch_size, self.desired_batch_size)

    def test_train_dynamic_dummy_data_with_ga_optimizer_local_base(self):
        """Test training a model using a GA optimizer with dynamically generated dummy data.

        This test uses the base_optimizer to make a ga_optimizer.
        """
        base_optimizer = optimizers.Adam(**self.optimizer_params)

        ga_optimizer = make_ga_optimizer(
            desired_batch_size=self.desired_batch_size,
            batch_size=self.batch_size,
            base_optimizer=base_optimizer,
            log_level=Optimizer.LOG_PARANOID,
        )
        self.model.compile(optimizer=ga_optimizer, loss=self.loss, metrics=self.metrics)

        print(f"GAOptimizer class: {Optimizer}")
        print(f"GAOptimizer bases: {Optimizer.__bases__}")
        print(f"Instance type: {type(ga_optimizer)}")
        print(
            f"Is instance of KerasBaseOptimizer: {isinstance(ga_optimizer, KerasBaseOptimizer)}"
        )
        print(
            f"Is instance of tf.keras.optimizers.Adam: {isinstance(ga_optimizer, tf.keras.optimizers.Adam)}"
        )

        # Create dummy datasets
        def dummy_data_generator(batch_size):
            while True:
                x = tf.random.uniform((batch_size, 10))
                y = tf.random.uniform((batch_size, 1))
                yield x, y

        dummy_training_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        validation_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.val_batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.val_batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.val_batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        # Train the model with the dummy dataset
        history = self.model.fit(
            dummy_training_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=self.val_batches,
        )

        # Verify the training history
        self.assertIn("loss", history.history)
        self.assertIn("binary_accuracy", history.history)
        self.assertGreater(len(history.history["loss"]), 0)

        # Verify the effective batch size
        effective_batch_size = self.batch_size * ga_optimizer.steps
        self.assertEqual(effective_batch_size, self.desired_batch_size)

    def test_train_dynamic_dummy_data_with_ga_Adam_optimizer(self):
        """Test training a model using a GA optimizer with dynamically generated dummy data.

        This test creates an Adam object directly from GA class.

        As long as GA.get_config() returns "steps", this will not succeed. If "steps" is not returned however, it will just create a base optimizer, sidestepping the GA wrapper.
        """
        # This will sidestep the GA wrapper. Not advised method.
        adam_optimizer = Adam(steps=10, **self.optimizer_params)

        with self.assertRaises(TypeError) as context:
            self.model.compile(
                optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics
            )

        self.assertIn(
            "Unexpected keyword argument passed to optimizer", str(context.exception)
        )

        return

        print(f"GAOptimizer class: {Optimizer}")
        print(f"GAOptimizer bases: {Optimizer.__bases__}")
        print(f"Instance type: {type(adam_optimizer)}")
        print(
            f"Is instance of KerasBaseOptimizer: {isinstance(adam_optimizer, KerasBaseOptimizer)}"
        )
        print(
            f"Is instance of tf.keras.optimizers.Adam: {isinstance(adam_optimizer, tf.keras.optimizers.Adam)}"
        )

        # Create dummy datasets
        def dummy_data_generator(batch_size):
            while True:
                x = tf.random.uniform((batch_size, 10))
                y = tf.random.uniform((batch_size, 1))
                yield x, y

        dummy_training_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        validation_dataset = tf.data.Dataset.from_generator(
            lambda: dummy_data_generator(self.val_batch_size),
            output_signature=(
                tf.TensorSpec(shape=(self.val_batch_size, 10), dtype=tf.float32),
                tf.TensorSpec(shape=(self.val_batch_size, 1), dtype=tf.float32),
            ),
        ).repeat(self.epochs)

        # Train the model with the dummy dataset
        history = self.model.fit(
            dummy_training_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=self.val_batches,
        )

        # Verify the training history
        self.assertIn("loss", history.history)
        self.assertIn("binary_accuracy", history.history)
        self.assertGreater(len(history.history["loss"]), 0)

        # Verify the effective batch size
        effective_batch_size = self.batch_size * adam_optimizer.steps
        self.assertEqual(effective_batch_size, self.desired_batch_size)


if __name__ == "__main__":
    unittest.main()
