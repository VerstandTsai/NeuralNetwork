import numpy as np
import matplotlib.pyplot as plt

class Activation:
    def __init__(self, name):
        funcs = {
            'sigmoid': [self.sigmoid, self.sigmoid_derivative],
            'relu': [self.relu, self.relu_derivative],
            'tanh': [self.tanh, self.tanh_derivative]
        }
        self.func = funcs[name][0]
        self.prime = funcs[name][1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.array([a if a > 0 else 0 for a in x])

    def relu_derivative(self, x):
        return np.array([1 if a > 0 else 0 for a in x])

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(np.tanh(x), 2)

class Layer:
    def __init__(self, size, activation=None):
        self.size = size
        self.activation = activation
        self.value = None
        self.activated = None
        self.weights = np.array([])
        self.biases = np.array([])
        self.output = None

    def feed(self, x):
        if self.weights.size == 0:
            self.output = x
            return

        self.value = np.dot(self.weights.T, x) + self.biases
        if self.activation == None:
            self.output = self.value
            return

        self.activated = self.activation.func(self.value)
        self.output = self.activated

class NeuralNetwork:
    def __init__(self, layers=[]):
        self.layers = []
        if len(layers) != 0:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if len(self.layers) != 0:
            layer.weights = np.random.randn(self.layers[-1].size, layer.size)
            layer.biases = np.random.randn(layer.size, 1)
        self.layers.append(layer)

    def forward(self, x):
        self.layers[0].feed(x)
        for i in range(1, len(self.layers)):
            self.layers[i].feed(self.layers[i-1].output)

    def backward(self, y):
        dA = 2 * (self.layers[-1].output - y)
        for i in range(-1, -len(self.layers), -1):
            dW = np.dot(self.layers[i-1].output, (self.layers[i].activation.prime(self.layers[i].value) * dA).T)
            dB = self.layers[i].activation.prime(self.layers[i].value) * dA
            dA = np.dot(self.layers[i].weights, self.layers[i].activation.prime(self.layers[i].value) * dA)
            self.layers[i].weights -= dW
            self.layers[i].biases -= dB

    def fit(self, data_x, data_y, epochs):
        for i in range(epochs):
            for j in range(len(data_x)):
                self.forward(data_x[j])
                self.backward(data_y[j])

    def predict(self, x):
        self.forward(x)
        return self.layers[-1].output

if __name__ == '__main__':
    nn = NeuralNetwork([
        Layer(2),
        Layer(4, Activation('sigmoid')),
        Layer(4, Activation('sigmoid')),
        Layer(1, Activation('sigmoid'))
    ])
    x = [
        np.array([[0], [0]]),
        np.array([[0], [1]]),
        np.array([[1], [0]]),
        np.array([[1], [1]])
    ]
    y = [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[1]]),
        np.array([[0]])
    ]
    nn.fit(x, y, 1000)
    print(nn.predict(np.array([[0], [0]])))
    print(nn.predict(np.array([[0], [1]])))
    print(nn.predict(np.array([[1], [0]])))
    print(nn.predict(np.array([[1], [1]])))
    