import nn
from mnist_loader import LoadMNIST
import numpy as np

if __name__ == '__main__':
    train_images = LoadMNIST('mnist/train-images-idx3-ubyte')
    train_labels = LoadMNIST('mnist/train-labels-idx1-ubyte')