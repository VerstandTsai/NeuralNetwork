import numpy as np

def LoadMNIST(filename, load_size):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic == 2049:
            labels = []
            num = int.from_bytes(f.read(4), 'big')
            for i in range(min(num, load_size)):
                labels.append(int.from_bytes(f.read(1), 'big'))
            return labels
        elif magic == 2051:
            images = []
            num = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            for i in range(min(num, load_size)):
                img = np.zeros((rows, cols))
                for j in range(rows):
                    for k in range(cols):
                        img[j][k] = int.from_bytes(f.read(1), 'big')
                images.append(img)
                print(f'\rLoading images... {i+1}/{min(num, load_size)}', end='')
            print()
            return images