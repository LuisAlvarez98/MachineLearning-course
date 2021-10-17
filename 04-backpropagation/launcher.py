"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 05/10/2021
"""

import matplotlib.pyplot as plt
from ann import NeuralNetwork
import numpy as np
from utils import get_one_hot, plot_decision_boundary, read_dataset, create_structure


def run_for_dataset(dataset, hidden_layers_conf, examples, learning_rate=0.01, regularization_rate=0.01, epochs=100):
    X, y = read_dataset(dataset)
    unique_classes = len(np.unique(y))
    y = get_one_hot(y, unique_classes)
    num_features = X.shape[0]

    ann = NeuralNetwork(input_neurons=num_features,
                        hidden_layers_conf=hidden_layers_conf,
                        output_neurons=unique_classes,
                        learning_rate=learning_rate,
                        regularization_rate=regularization_rate,
                        epochs=epochs)
    ann.fit(X, y)

    for example in examples:
        r = ann.predict(example)
        print("{} predicted as {}".format(example.T, r.T))

    plot_decision_boundary(
        X.T, y.T, ann, title="ANN dataset={}".format(dataset))

    print('\n')


def get_config_for_example():
    X = np.array([[0.05], [0.10]])
    y = np.array([[0.01], [0.99]])

    hidden = [2]
    # el dos es por las neuronas en la capa de salida
    theta = create_structure(2, 2, hidden)

    theta_1 = theta[0]
    theta_1[0, 0] = 0.35
    theta_1[0, 1] = 0.15
    theta_1[0, 2] = 0.20

    theta_1[1, 0] = 0.35
    theta_1[1, 1] = 0.25
    theta_1[1, 2] = 0.30

    #####
    theta_2 = theta[1]
    theta_2[0, 0] = 0.60
    theta_2[0, 1] = 0.40
    theta_2[0, 2] = 0.45

    theta_2[1, 0] = 0.60
    theta_2[1, 1] = 0.50
    theta_2[1, 2] = 0.55

    return (X, y, hidden, theta)

def printTitle(s : str):
    print((' '+s+' ').upper().center(50,'+'))

def runExample():
    printTitle('Example dataset')
    X, y, hidden_layers_conf, theta = get_config_for_example()
    example = np.array([[0.05, 0.1]]).T

    ann = NeuralNetwork(input_neurons=2,
                        hidden_layers_conf=hidden_layers_conf,
                        output_neurons=2,
                        learning_rate=0.5,
                        regularization_rate=0,
                        epochs=1)
    ann.theta = theta
    r = ann.predict(example)
    print("{} predicted as {}".format(example.T, r.T))

    ann.fit(X, y, theta)

    r = ann.predict(example)
    print("{} predicted as {}\n\n".format(example.T, r.T))

def runXOR():
    printTitle('XOR dataset')
    np.random.seed(0)
    dataset = './datasets/xor.csv'
    hidden_layers_conf = [6,6]
    reg_factor = 0.0
    alpha=0.5
    epochs = 100000
    to_predict = []
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[0, 1]]).T)
    to_predict.append(np.array([[1, 0]]).T)
    to_predict.append(np.array([[1, 1]]).T)

    run_for_dataset(dataset, hidden_layers_conf, to_predict, learning_rate=alpha, regularization_rate=reg_factor, epochs=epochs)

def runBlobs():
    printTitle('blobs dataset')
    np.random.seed(5)
    dataset = './datasets/blobs.csv'
    hidden_layers_conf = [2,5]
    reg_factor = 0.0
    alpha=0.05
    epochs = 5000
    to_predict = []
    to_predict.append(np.array([[1, -9]]).T)
    to_predict.append(np.array([[-4, 7.8]]).T)
    to_predict.append(np.array([[-9, 4.5]]).T)

    run_for_dataset(dataset, hidden_layers_conf, to_predict, learning_rate=alpha, regularization_rate=reg_factor, epochs=epochs)

def runMoons():
    printTitle('Moons dataset')
    np.random.seed(0)
    dataset = './datasets/moons.csv'
    hidden_layers_conf = [4,4,4]
    reg_factor = 0.0
    alpha=0.1
    epochs = 500000
    to_predict = []
    to_predict.append(np.array([[-0.5, 0.5]]).T)
    to_predict.append(np.array([[1, 0.5]]).T)
    to_predict.append(np.array([[0, 0]]).T)
    to_predict.append(np.array([[1.5, -0.5]]).T)

    run_for_dataset(dataset, hidden_layers_conf, to_predict, learning_rate=alpha, regularization_rate=reg_factor, epochs=epochs)

def runCircles():
    printTitle('circles dataset')
    np.random.seed(0)
    dataset = './datasets/circles.csv'
    hidden_layers_conf = [9,9,9]
    reg_factor = 0.0
    alpha=0.05
    epochs = 500000
    to_predict = []
    to_predict.append(np.array([[-0.6, -0.85]]).T)
    to_predict.append(np.array([[0.75, -0.06]]).T)

    run_for_dataset(dataset, hidden_layers_conf, to_predict, learning_rate=alpha, regularization_rate=reg_factor, epochs=epochs)

if __name__ == "__main__":

    # For example dataset
    runExample()
        

    # For XOR dataset
    runXOR()


    # For blobs dataset
    runBlobs()

    
    # For moons dataset
    runMoons()

    # For circles dataset
    runCircles()

    plt.show()
