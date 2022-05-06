import numpy as np
from activations import Activation

class Layer:
    def __init__(self, size):
        self.size = size
        self.value = None
        self.output = None

    def feed(self, x):
        pass

class Flatten(Layer):
    def __init__(self, size):
        super().__init__(np.prod(size))

    def feed(self, x):
        self.output = x.reshape(x.size, 1)

class FullyConnected(Layer):
    def __init__(self, size, activation=Activation('linear')):
        super().__init__(size)
        self.activation = activation
        self.weights = np.array([])
        self.biases = np.array([])

    def feed(self, x):
        self.value = np.dot(self.weights.T, x) + self.biases
        self.output = self.activation.func(self.value)
