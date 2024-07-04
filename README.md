# GA Optimizer

## Info

GA Optimizer is a wrapper for Keras/TensorFlow optimizers to enable gradient accumulation, allowing you to simulate larger batch sizes than your hardware can handle.

## Getting Started

### Installation
#### Pip
You can install the package using pip:

```bash
pip install ga-optimizer
```

#### Clone repo

Alternatively, you can clone the repository directly from GitHub and install it manually:

```bash
git clone https://github.com/kimjansheden/GAOptimizer.git
cd GAOptimizer
pip install .
```

### Usage
Here's an example of how to use GA Optimizer in your TensorFlow/Keras project:

```python
import tensorflow as tf
from tensorflow.keras import layers
from ga_optimizer import make_ga_optimizer

# Define a simple model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Define loss and metrics
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# Define base optimizer
base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Wrap the optimizer with GA Optimizer
ga_optimizer = make_ga_optimizer(
    desired_batch_size=64,
    batch_size=8,
    base_optimizer=base_optimizer,
    log_level=Ga_Optimizer.LOG_PARANOID
)

# Compile your model with GA optimizer
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Generate some dummy data
import numpy as np
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
```

## Functions and Classes

### `make_ga_optimizer`

The `make_ga_optimizer` function wraps a given Keras/TensorFlow optimizer to enable gradient accumulation.

#### Arguments

- `desired_batch_size` (int): The effective batch size you want to simulate.
- `batch_size` (int): The actual batch size that your hardware can handle.
- `base_optimizer` (tf.keras.optimizers.Optimizer): The base optimizer to wrap.
- `base_optimizer_params` (dict, optional): Parameters for the base optimizer. Only needed if you have any params in your base_optimizer and you're on a Mac where optimizer gets converted to legacy. Defaults to `None`.
- `log_level` (int, optional): Logging level. Defaults to `optimizers.Optimizer.LOG_NONE`.

### `Optimizer` Class

This class is a wrapper for Keras optimizers to support gradient accumulation.

## Logging Levels

The GA Optimizer supports different logging levels:

- `LOG_NONE`: No logs.
- `LOG_INFO`: Informational messages.
- `LOG_DEBUG`: Debug messages, more verbose.
- `LOG_PARANOID`: Paranoid debug messages, extremely verbose.
- `LOG_EXTREMELY_PARANOID`: Extremely paranoid debug messages, extremely, extremely verbose.

## Tests
To run the tests:
```sh
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Acknowledgements

I would like to acknowledge and thank the developers of the [RunAI GA Optimizer](https://github.com/bamps53/runai/tree/master/runai/ga) project, which served as the foundation for this work. Their innovative approach to gradient accumulation inspired and enabled the development of this extended version. 