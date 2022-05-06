import numpy as np

def LoadMNIST(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        print(magic)