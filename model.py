import numpy as np
import matplotlib.pyplot as plt

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
        self.loss = np.sum((self.layers[-1].output - y) ** 2)
        dError = self.layers[-1].output - y
        for i in range(-1, -len(self.layers), -1):
            dError = self.layers[i].feedback(dError, self.learning_rate)

    def fit(self, data_x, data_y, learning_rate, epochs, plot_losses=False):
        self.learning_rate = learning_rate
        self.loss_log = []
        for i in range(epochs):
            counter = 0
            losses = []
            for j in range(len(data_x)):
                if counter > 100:
                    avgloss = np.average(losses)
                    self.loss_log.append(avgloss)
                    counter = 0
                    losses = []
                    print(f'\rEpoch: {i+1}/{epochs} Loss: {avgloss}', end='')
                self.forward(data_x[j])
                self.backward(data_y[j])
                losses.append(self.loss)
                counter += 1
        print()
        if plot_losses:
            plt.plot(self.loss_log)

    def predict(self, x):
        self.forward(x)
        return self.layers[-1].output
