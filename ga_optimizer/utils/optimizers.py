import inspect
import json
import sys
from tensorflow import keras
import keras.backend as K
import tensorflow as tf

## DEBUG START
print("keras ", keras.optimizers)
## DEBUG STOP

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
        ## DEBUG START
        print(
            f"GA Optimizer.__init__ called with name={name}, optimizer={optimizer} steps={steps}, log_level={log_level}"
        )
        ## DEBUG STOP
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

        ## DEBUG START
        print("type(self):", type(self))
        print("type(self.optimizer)", type(self.optimizer))
        ## DEBUG STOP

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
        ## DEBUG START
        # print(f"get_config called by {caller_info}")
        ## DEBUG STOP

        # Check if the call is coming from within the Keras package
        if "/keras/" in caller.filename:
            ## DEBUG START
            # print(
            #     f"Keras is calling get_config, using get_keras_config instead to return config to: {caller_info}"
            # )
            ## DEBUG STOP
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

        ## DEBUG START
        # print("get_keras_config: type(config):", type(config))

        # print(
        #     "get_keras_config: Returning config:",
        #     json.dumps(config, indent=4, default=str),
        # )
        ## DEBUG STOP
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
            ## DEBUG START
            # print(f"__setattr__: {name} is an instance attribute (it belongs to the wrapper). Setting attribute '{name}' to {value} on self")
            ## DEBUG STOP
            # Directly set the attribute on the current instance
            object.__setattr__(self, name, value)
            ## DEBUG START
            # print(
            #     f"__setattr__: Successfully set attribute '{name}' to {value} on self"
            # )
            ## DEBUG STOP
        else:
            try:
                ## DEBUG START
                # print(f"__setattr__: {name} is not an instance attribute. Trying to see if self.optimizer is instantiated and set {name} to {value} on self.optimizer")
                ## DEBUG STOP

                # Try to set the attribute on the inner optimizer object
                optimizer = object.__getattribute__(self, "optimizer")
                setattr(optimizer, name, value)
                ## DEBUG START
                # print(
                #     f"__setattr__: Successfully set attribute '{name}' to {value} on self.optimizer"
                # )
                ## DEBUG STOP
            except AttributeError:
                ## DEBUG START
                # print(
                #     f"__setattr__: Attribute '{name}' not found in self or self.optimizer"
                # )
                # print(f"__setattr__: Trying to set attribute '{name}' to {value} on self")
                ## DEBUG STOP
                try:
                    object.__setattr__(self, name, value)
                    ## DEBUG START
                    # print(
                    #     f"__setattr__: Successfully set attribute '{name}' to {value} on self"
                    # )
                    ## DEBUG STOP
                    return
                except AttributeError:
                    ## DEBUG START
                    # print(
                    #     f"__setattr__: Attribute '{name}' not found in self or self.optimizer either"
                    # )
                    ## DEBUG STOP
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
                    ## DEBUG START
                    # print(
                    #     f"Error retrieving weights from optimizer: {e}. Returning empty list"
                    # )
                    ## DEBUG STOP
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
        ## DEBUG START
        # print(f"__getattribute__ {name} called by {caller_info}")
        ## DEBUG STOP

        # Check if the call is coming from within the Keras package and name is "__class__"
        if (
            "/keras/" in caller.filename
            and caller.function != "__init__"
            and name == "__class__"
        ):
            # Trick Keras by returning the wrapped optimizer class' name when "name" is "__class__"
            ## DEBUG START
            # print(
            #     f"Keras is calling __getattribute__ with __class__. Return optimizer's class name instead to: {caller_info}"
            # )
            # print("Returning", self.optimizer.__class__)
            ## DEBUG STOP
            return self.optimizer.__class__

        # First, try to get the attribute from 'self'
        try:
            ## DEBUG START
            # print(
            #     f"Returning attribute '{name}' from self. Returning: {object.__getattribute__(self, name)}"
            # )
            ## DEBUG STOP
            return object.__getattribute__(self, name)
        except AttributeError:
            pass  # If the attribute is not found, proceed to the next step

        ## DEBUG START
        # print(
        #     f"Attribute {name} not found in self. Getting attribute '{name}' from self.optimizer instead"
        # )
        ## DEBUG STOP

        # Check if 'self.optimizer' is instantiated
        optimizer = object.__getattribute__(self, "optimizer")
        if optimizer is not None:
            ## DEBUG START
            # print(
            #     f"'self.optimizer' is instantiated. Getting attribute '{name}' from {optimizer.__class__.__name__}"
            # )
            ## DEBUG STOP
            try:
                return getattr(optimizer, name)
            except Exception as e:
                ## DEBUG START
                # print(
                #     f"Error returning attribute '{name}' from {optimizer.__class__.__name__}: {e}"
                # )
                ## DEBUG STOP
                raise

        # If 'self.optimizer' is not instantiated, raise AttributeError
        raise AttributeError(
            f"Error in __getattribute__: '{type(self).__name__}' object has no attribute '{name}'"
        )

    def log(self, level, *args, **kwargs):
        """Logs a message if the given log level is high enough."""
        ## DEBUG START
        # print(f"self.log_level: {self.log_level}")
        # print(f"level: {level}")
        ## DEBUG STOP
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
            ## DEBUG START
            # print(
            #     f"Creating {optimizer_name} optimizer with steps: {steps} and kwargs: {kwargs}"
            # )
            ## DEBUG STOP
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in INSTANCE_ATTRIBUTES and k not in CLASS_ATTRIBUTES
            }
            ## DEBUG START
            # print(f"Filtered kwargs for {optimizer_name}: {filtered_kwargs}")
            ## DEBUG STOP

            keras_object = keras.optimizers
            keras_optimizer = getattr(keras_object, optimizer_name)(**filtered_kwargs)

            ## DEBUG START
            # print(f"Keras optimizer: {keras_optimizer}")
            ## DEBUG STOP
            Optimizer.__init__(
                self, name=optimizer_name, optimizer=keras_optimizer, steps=steps
            )

        optimizer_class = type(
            optimizer_name,
            (Optimizer,),
            {"__init__": init_optimizer},
        )
        setattr(sys.modules[__name__], optimizer_name, optimizer_class)
        ## DEBUG START
        # print(
        #     # f"Successfully created optimizer class '{optimizer_name}' with attributes {dir(optimizer_class)}"
        # )
        ## DEBUG STOP
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
