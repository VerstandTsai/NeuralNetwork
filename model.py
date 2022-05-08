import numpy as np

class Model:
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
        self.loss = (self.layers[-1].output - y) ** 2
        dA = self.layers[-1].output - y
        for i in range(-1, -len(self.layers), -1):
            dW = np.dot(self.layers[i-1].output, (self.layers[i].activation.prime(self.layers[i].value) * dA).T)
            dB = self.layers[i].activation.prime(self.layers[i].value) * dA
            dA = np.dot(self.layers[i].weights, self.layers[i].activation.prime(self.layers[i].value) * dA)
            self.layers[i].weights -= self.learning_rate * dW
            self.layers[i].biases -= self.learning_rate * dB

    def fit(self, data_x, data_y, learning_rate, epochs):
        self.learning_rate = learning_rate
        for i in range(epochs):
            self.losses = []
            for j in range(len(data_x)):
                self.forward(data_x[j])
                self.backward(data_y[j])
                self.losses.append(self.loss)
            print(f'\rEpoch: {i+1}/{epochs} Loss: {np.average(self.losses)}', end='')
        print()

    def predict(self, x):
        self.forward(x)
        return self.layers[-1].output
