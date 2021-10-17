import matplotlib.pyplot as plt
from knn import Knn
import numpy as np
from utils import read_dataset


def run_for_dataset(dataset, k, examples):
    X, y = read_dataset(dataset)

    knn = Knn(k)
    knn.fit(X, y)
    for example in examples:
        r = knn.predict(example)
        print("{} predicted as {}".format(example.T, r))

if __name__ == "__main__":
    # For XOR dataset
    print('XOR dataset')
    dataset = './datasets/xor.csv'
    to_predict = []
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[0, 1]]).T)
    to_predict.append(np.array([[1, 0]]).T)
    to_predict.append(np.array([[1, 1]]).T)
    k = 3
    run_for_dataset(dataset, k, to_predict)

    # For blobs dataset
    print('blobs dataset')
    dataset = './datasets/blobs.csv'
    to_predict = []
    to_predict.append(np.array([[1, -9]]).T)
    to_predict.append(np.array([[-4, 7.8]]).T)
    to_predict.append(np.array([[-9, 4.5]]).T)
    k=3
    run_for_dataset(dataset, k, to_predict)

    
    # For moons dataset
    print('Moons dataset')
    dataset = './datasets/moons.csv'
    to_predict = []
    to_predict.append(np.array([[-0.5, 0.5]]).T)
    to_predict.append(np.array([[1, 0.5]]).T)
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[1.5, -0.5]]).T)

    run_for_dataset(dataset, k, to_predict)

    # For circles dataset
    print('circles dataset')
    dataset = './datasets/circles.csv'
    to_predict = []
    to_predict.append(np.array([[-0.6, -0.85]]).T)
    to_predict.append(np.array([[0.75, -0.06]]).T)

    run_for_dataset(dataset, k, to_predict)