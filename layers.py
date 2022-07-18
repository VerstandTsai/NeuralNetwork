import numpy as np
from activations import Activation

class Flatten():
    def __init__(self, size):
        self.size = np.prod(size)

    def feed(self, x):
        self.output = x.reshape(x.size, 1)

class FullyConnected():
    def __init__(self, size, activation=Activation('linear')):
        self.size = size
        self.activation = activation
        self.weights = np.array([])
        self.biases = np.array([])

    def feed(self, x):
        self.value = np.dot(self.weights.T, x) + self.biases
        self.output = self.activation.func(self.value)
