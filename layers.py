import numpy as np
from activations import Activation

class Flatten():
    def __init__(self, size):
        self.size = np.prod(size)

    def feed(self, x):
        self.output = x.reshape(x.size, 1)

    def feedback(self, dError, learning_rate):
        return dError

class FullyConnected():
    def __init__(self, size, activation=Activation('linear')):
        self.size = size
        self.activation = activation
        self.weights = np.array([])
        self.biases = np.array([])

    def feed(self, x):
        self.x = x
        self.value = np.dot(self.weights.T, x) + self.biases
        self.output = self.activation.func(self.value)

    def feedback(self, dError, learning_rate):
        dB = self.activation.prime(self.value) * dError
        dW = np.dot(self.x, dB.T)
        dError = np.dot(self.weights, dB)
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB
        return dError
