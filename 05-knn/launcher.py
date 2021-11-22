"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 18/10/2021
"""

import matplotlib.pyplot as plt
from knn import Knn
import numpy as np
from utils import read_dataset

K = 3


def run_for_dataset(dataset, k, examples):
    X, y = read_dataset(dataset)

    knn = Knn(k)
    knn.fit(X, y)
    for example in examples:
        r = knn.predict(example)
        print("{} predicted as {}".format(example.T, r))

def runXor():
    # For XOR dataset
    print('XOR dataset')
    dataset = './datasets/xor.csv'
    to_predict = []
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[0, 1]]).T)
    to_predict.append(np.array([[1, 0]]).T)
    to_predict.append(np.array([[1, 1]]).T)
    run_for_dataset(dataset, K, to_predict)

def runBlobs():
    # For blobs dataset
    print('blobs dataset')
    dataset = './datasets/blobs.csv'
    to_predict = []
    to_predict.append(np.array([[1, -9]]).T)
    to_predict.append(np.array([[-4, 7.8]]).T)
    to_predict.append(np.array([[-9, 4.5]]).T)
    run_for_dataset(dataset, K, to_predict)

def runMoons():
    # For moons dataset
    print('Moons dataset')
    dataset = './datasets/moons.csv'
    to_predict = []
    to_predict.append(np.array([[-0.5, 0.5]]).T)
    to_predict.append(np.array([[1, 0.5]]).T)
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[1.5, -0.5]]).T)
    run_for_dataset(dataset, K, to_predict)

def runCircles():
    # For circles dataset
    print('circles dataset')
    dataset = './datasets/circles.csv'
    to_predict = []
    to_predict.append(np.array([[-0.6, -0.85]]).T)
    to_predict.append(np.array([[0.75, -0.06]]).T)
    run_for_dataset(dataset, K, to_predict)

if __name__ == "__main__":
    
    # Run Xor parts of the code
    runXor()
    
    # Run Blobs parts of the code
    runBlobs()
    
    # Run Moons parts of the code
    runMoons()
    
    # Run Circles parts of the code
    runCircles()
    