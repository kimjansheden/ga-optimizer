# Create a simple class that inherits Keras Optimizer and just initiates all super stuff
from keras.optimizers import Optimizer


class SimpleOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
