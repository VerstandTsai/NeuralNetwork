import nn
from mnist_loader import LoadMNIST
import numpy as np
from time import time

if __name__ == '__main__':
    load_size = 10000
    train_ratio = 0.8
    train_size = int(load_size * train_ratio)
    train_images_temp = LoadMNIST('mnist/train-images-idx3-ubyte', load_size)
    train_images = []
    for img in train_images_temp:
        train_images.append(img / 255)
    train_labels_temp = LoadMNIST('mnist/train-labels-idx1-ubyte', load_size)
    train_labels = []
    for label in train_labels_temp:
        net_end = np.zeros((10, 1))
        net_end[label][0] = 1
        train_labels.append(net_end)

    network = nn.Model([
        nn.layers.Flatten((28, 28)),
        nn.layers.FullyConnected(16, nn.Activation('sigmoid')),
        nn.layers.FullyConnected(16, nn.Activation('sigmoid')),
        nn.layers.FullyConnected(10, nn.Activation('sigmoid')),
    ])

    rights = []
    for i in range(train_size, load_size):
        rights.append(1 if np.argmax(network.predict(train_images[i])) == train_labels_temp[i] else 0)
    print('Before training:')
    print(f'Accuracy: {np.average(rights) * 100}%')

    start = time()
    network.fit(train_images[:train_size], train_labels[:train_size], 1, 5, plot_losses=True)
    end = time()
    print(f'Time used: {end - start} seconds.')

    rights = []
    for i in range(train_size, load_size):
        rights.append(1 if np.argmax(network.predict(train_images[i])) == train_labels_temp[i] else 0)
    print('After training:')
    print(f'Accuracy: {np.average(rights) * 100}%')