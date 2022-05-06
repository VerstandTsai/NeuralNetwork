import numpy as np

class Activation:
    def __init__(self, name):
        funcs = {
            'linear': [self.linear, self.linear_derivative],
            'sigmoid': [self.sigmoid, self.sigmoid_derivative],
            'relu': [self.relu, self.relu_derivative],
            'tanh': [self.tanh, self.tanh_derivative]
        }
        self.func = funcs[name][0]
        self.prime = funcs[name][1]

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones(x.shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derivative(self, x):
        return np.array([[1] if a[0] > 0 else [0] for a in x])

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
