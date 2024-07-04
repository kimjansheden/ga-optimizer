import inspect
import json
import sys
from tensorflow import keras
import keras.backend as K
import tensorflow as tf


INSTANCE_ATTRIBUTES = {
    "optimizer",
    "steps",
    "log_level",
    "accumulated_gradients",
    "accumulation_counter",
}

CLASS_ATTRIBUTES = {
    "LOG_NONE",
    "LOG_INFO",
    "LOG_DEBUG",
    "LOG_PARANOID",
    "LOG_EXTREMELY_PARANOID",
}


class Optimizer(keras.optimizers.Optimizer):
    """Optimizer Wrapper for Keras Optimizers to support Gradient Accumulation"""

    # Log Levels
    LOG_NONE = 0  # No logs
    LOG_INFO = 1  # Informational messages
    LOG_DEBUG = 2  # Debug messages, more verbose
    LOG_PARANOID = 3  # Paranoid Debug messages, extremely verbose
    LOG_EXTREMELY_PARANOID = (
        4  # Extremely paranoid Debug messages, extremely, extremely verbose
    )

    def __init__(self, name, optimizer, steps, log_level=LOG_NONE):
        """
        Initialize the Optimizer.

        Args:
            name (str): Name of the optimizer.
            optimizer (keras.optimizers.Optimizer): Instance of a Keras optimizer to wrap.
            steps (int): Number of steps to accumulate gradients.
            log_level (int, optional): Logging level. Defaults to LOG_NONE.
        """
        super(Optimizer, self).__init__(name)
        self.optimizer = optimizer
        self.steps = steps
        self.log_level = log_level
        self.accumulated_gradients = None
        self.accumulation_counter = K.variable(
            0, dtype="int64", name="accumulation_counter"
        )

        print(f"Wrapped optimizer class: {self.optimizer.__class__.__name__}")
        print(f"Wrapped optimizer module: {self.optimizer.__class__.__module__}")
        print(f"Wrapped optimizer file: {inspect.getfile(self.optimizer.__class__)}")

        print(
            f"Wrapping '{optimizer.__class__.__name__}' Keras optimizer with GA of {steps} steps"
        )


    def apply_gradients(self, grads_and_vars, **kwargs):
        if self.accumulated_gradients is None:
            self.log(
                self.LOG_DEBUG,
                "No accumulated_gradients. Initializing accumulated_gradients",
            )
            # Initialize accumulated_gradients as tf.Variable
            self.accumulated_gradients = [
                tf.Variable(tf.zeros_like(g), trainable=False)
                for g, _ in grads_and_vars
            ]

        # Update iterations
        self.accumulation_counter.assign_add(1)

        # Determine if it's time to apply gradients
        apply_gradients = K.equal(self.accumulation_counter % self.steps, 0)

        def apply_and_reset_gradients():
            self.log(
                self.LOG_DEBUG,
                "Step",
                self.accumulation_counter,
                ": Ready to apply gradients",
            )

            # Accumulate gradients
            accumulate_gradients()

            if self.log_level >= self.LOG_DEBUG:
                # Log the gradients
                for grad, var in zip(
                    self.accumulated_gradients, [v for _, v in grads_and_vars]
                ):
                    grad_norm = tf.norm(grad)
                    self.log(
                        self.LOG_DEBUG,
                        "Gradient norm for variable",
                        var.name,
                        ":",
                        grad_norm,
                    )

            self.optimizer.iterations.assign_add(1)

            # Apply gradients after 'self.steps' accumulated steps
            # Here, each accumulated gradient is divided by the number of steps (self.steps).
            # This normalization ensures that the magnitude of the gradient updates is equivalent
            # to an average update for a larger batch size. This step is important for maintaining
            # the scale of the updates, preventing them from becoming too large when accumulated
            # over multiple steps.
            self.optimizer.apply_gradients(
                zip(
                    [ag / (self.steps) for ag in self.accumulated_gradients],
                    [v for _, v in grads_and_vars],
                ),
                **kwargs,
            )

            self.log(
                self.LOG_DEBUG,
                "Gradients have been applied",
                (self.optimizer.iterations - 1),
                "times.",
            )
            self.log(
                self.LOG_DEBUG, "self.optimizer.iterations:", self.optimizer.iterations
            )

            if self.log_level >= self.LOG_PARANOID:
                # Log the norm before reset
                for acc_grad in self.accumulated_gradients:
                    self.log(
                        self.LOG_PARANOID,
                        "Norm of accumulated gradient before reset:",
                        tf.norm(acc_grad),
                    )

            reset_accumulation()
            return tf.constant(True)

        def reset_accumulation():
            # Reset accumulated gradients
            for acc_grad in self.accumulated_gradients:
                acc_grad.assign(tf.zeros_like(acc_grad))
                self.log(
                    self.LOG_PARANOID,
                    "Norm of accumulated gradient after reset:",
                    tf.norm(acc_grad),
                )

            # Reset iterations
            self.accumulation_counter.assign(0)

        def accumulate_gradients():
            # Accumulate gradients
            self.log(
                self.LOG_DEBUG,
                "Accumulating gradients for step",
                self.accumulation_counter,
            )
            for i, (g, _) in enumerate(grads_and_vars):
                self.accumulated_gradients[i].assign_add(g)
            return tf.constant(False)

        # Use tf.cond to conditionally apply or accumulate gradients
        tf.cond(apply_gradients, apply_and_reset_gradients, accumulate_gradients)

        return None

    def get_gradients(self, loss, params):
        # Delegate to self.optimizer
        return self.optimizer.get_gradients(loss, params)

    def set_weights(self, weights):
        # Delegate to self.optimizer
        self.optimizer.set_weights(weights)

    def get_weights(self):
        # Delegate to self.optimizer
        return self.optimizer.get_weights()

    def get_config(self):
        # we have to support creating our optimizers from configurations in order to support being run with Horovod
        # Horovod dynamically creates a class that inherits the optimizer class it's wrapping (our optimizers), and
        # passes the dictionary returned from this very method as the kwargs for the initialization in __init__()
        #
        # our optimizers inherit from this very class, receive 'steps' as an argument, and do not receive 'optimizer'
        # as they create the one they mimic
        #
        # therefore, we do not save self.optimizer in the returned dictionary

        # Get the caller's information
        stack = inspect.stack()
        caller = stack[1]  # Index 1 to get the immediate caller
        caller_info = f"{caller.function} at {caller.filename}:{caller.lineno}"

        # Check if the call is coming from within the Keras package
        if "/keras/" in caller.filename:
            return self.get_keras_config()

        config = self.optimizer.get_config()
        config["steps"] = self.steps

        self.log(
            self.LOG_DEBUG,
            "Returning config:",
            json.dumps(config, indent=4, default=str),
        )

        return config

    def get_keras_config(self):
        """
        Here we can exclude "steps" from the returned dictionary and other attributes as needed, if they are not valid arguments for the optimizer we send to keras model.
        """
        config = self.optimizer.get_config()
        config["steps"] = self.steps

        return config

    def __setattr__(self, name, value):
        """
        Set an attribute on either the outer optimizer (wrapper) or the inner optimizer.

        This method ensures that attributes are set on the correct object. If the attribute
        name is in INSTANCE_ATTRIBUTES, it sets the attribute on the outer optimizer (the wrapper).
        Otherwise, it attempts to set the attribute on the inner optimizer. This prevents confusion
        and ensures that attributes are not inadvertently set on the wrong object, especially when
        there are overlapping attribute names between the wrapper and the inner optimizer.

        Args:
            name (str): Name of the attribute.
            value: Value to set for the attribute.
        """
        if name in INSTANCE_ATTRIBUTES:
            # Directly set the attribute on the current instance
            object.__setattr__(self, name, value)
        else:
            try:

                # Try to set the attribute on the inner optimizer object
                optimizer = object.__getattribute__(self, "optimizer")
                setattr(optimizer, name, value)
            except AttributeError:
                try:
                    object.__setattr__(self, name, value)
                    return
                except AttributeError:
                    pass

    def __getattr__(self, name):
        """
        Get an attribute of the optimizer.

        This method is called only when the attribute is not found in the usual places.
        It ensures that if an attribute is not found on the current instance, it will
        be retrieved from the internal optimizer object. This is useful for accessing
        attributes that might overlap between the wrapper and the internal optimizer.

        Args:
            name (str): Name of the attribute.

        Returns:
            The value of the attribute.
        """
        if name == "_weights":
            try:
                return self.optimizer.weights
            except Exception as e:
                try:
                    return self.optimizer._weights
                except Exception as e:
                    return []

    def __getattribute__(self, name):
        """
        Get an attribute of the optimizer.

        This method is always called when an attribute is accessed. It first tries to get the
        attribute from the current instance. If the attribute is not found, it will then try
        to retrieve it from the internal optimizer object. This ensures that attributes are
        accessed correctly, avoiding confusion between attributes of the wrapper and the
        internal optimizer.

        Args:
            name (str): Name of the attribute.

        Returns:
            The value of the attribute.
        """
        # users can query the optimizer to retrieve its attributes, such as 'lr'.
        # we rely on the fact that there are no mutual attribute names between our
        # implementation and the original optimizer implementation, and we get the
        # original optimizer's attribute in case our object does not have one.

        # Get the caller's information
        stack = inspect.stack()
        caller = stack[1]  # Index 1 to get the immediate caller
        caller_info = f"{caller.function} at {caller.filename}:{caller.lineno}"

        # Check if the call is coming from within the Keras package and name is "__class__"
        if (
            "/keras/" in caller.filename
            and caller.function != "__init__"
            and name == "__class__"
        ):
            # Trick Keras by returning the wrapped optimizer class' name when "name" is "__class__"
            return self.optimizer.__class__

        # First, try to get the attribute from 'self'
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass  # If the attribute is not found, proceed to the next step


        # Check if 'self.optimizer' is instantiated
        optimizer = object.__getattribute__(self, "optimizer")
        if optimizer is not None:
            try:
                return getattr(optimizer, name)
            except Exception as e:
                raise

        # If 'self.optimizer' is not instantiated, raise AttributeError
        raise AttributeError(
            f"Error in __getattribute__: '{type(self).__name__}' object has no attribute '{name}'"
        )

    def log(self, level, *args, **kwargs):
        """Logs a message if the given log level is high enough."""
        if level <= self.log_level:
            tf.print(*args, **kwargs)


def _optimizer(optimizer_name):
    """
    Create a new Optimizer subclass with gradient accumulation support. This method is run when the separate wrapped classes are created by the loop at the end of this file, not otherwise.

    Args:
        optimizer_name (str): Name of the Keras optimizer to wrap.

    Raises:
        TypeError: If there is an error creating the optimizer class.
    """
    try:

        def init_optimizer(self, steps, **kwargs):
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in INSTANCE_ATTRIBUTES and k not in CLASS_ATTRIBUTES
            }

            keras_object = keras.optimizers
            keras_optimizer = getattr(keras_object, optimizer_name)(**filtered_kwargs)

            Optimizer.__init__(
                self, name=optimizer_name, optimizer=keras_optimizer, steps=steps
            )

        optimizer_class = type(
            optimizer_name,
            (Optimizer,),
            {"__init__": init_optimizer},
        )
        setattr(sys.modules[__name__], optimizer_name, optimizer_class)
    except TypeError as e:
        print(f"Error creating optimizer class '{optimizer_name}': {e}")
        raise


for optimizer_name in [
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adam",
    "Adamax",
    "Nadam",
]:
    _optimizer(optimizer_name)
