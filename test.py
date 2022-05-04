import nn
import numpy as np

if __name__ == '__main__':
    network = nn.Model([
        nn.layers.Input(2),
        nn.layers.FullyConnected(4, nn.Activation('sigmoid')),
        nn.layers.FullyConnected(4, nn.Activation('sigmoid')),
        nn.layers.FullyConnected(1, nn.Activation('sigmoid'))
    ])
    train_x = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])
    train_y = np.array([[[0]], [[1]], [[1]], [[0]]])
    network.fit(train_x, train_y, 10000)
    print(network.predict(np.array([[0], [0]])))
    print(network.predict(np.array([[0], [1]])))
    print(network.predict(np.array([[1], [0]])))
    print(network.predict(np.array([[1], [1]])))