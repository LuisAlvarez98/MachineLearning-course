"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 05/10/2021
"""

from matplotlib.pyplot import legend
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
import numpy as np
import pandas as pd


def read_dataset(path):
    """
    Reads the specified dataset and returns the data as used by the Logistic Regressor
    """

    # Reading the dataset
    data = pd.read_csv(path)

    # all but first column. This gets an (n x m) array
    X = data.iloc[:, :-1].to_numpy().T

    # Process to select the y column
    y = data.iloc[:, -1].to_numpy()  # the last col is y

    # this is to get (1,m) rather than an (m,) array (2d instead of 1d)
    y = y.reshape(y.shape[0], -1).T
    return X, y


def create_structure(num_classes, num_features, hidden_layers_conf):
    weights = []
    out_neurons = num_classes

    h_layers = len(hidden_layers_conf)
    s_j = num_features
    for j in range(0, h_layers):
        s_jplus1 = hidden_layers_conf[j]
        cols = s_j + 1
        weights_layer = np.random.rand(s_jplus1, cols)  # between 0,1
        weights.append(weights_layer)
        s_j = s_jplus1

    weights.append(np.random.rand(out_neurons, s_j + 1))  # between 0-1

    return weights


def create_structure_for_ann(ann) -> list:
    """
    This function returns a structure to hold the weights or Deltas of the specified ann
    """
    num_features = ann.input_neurons
    num_classes = ann.output_neurons
    hidden_layers_conf = ann.hidden_layers
    structure = create_structure(num_classes, num_features, hidden_layers_conf)
    return structure


def get_one_hot(targets, nb_classes):
    """
    Performs a one hot encoding for the specified class values
    E.g., if [0,1,0] and 2 is received, then [[1, 0], [0, 1], [1, 0]] is returned
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.T


def plot_decision_boundary(X, y, ann, title):
    import matplotlib.pyplot as plt

    colormap = 'Greens'
    y_as_int = np.argmax(y, axis=1)  # this is to convert them to 0,1,2 etc.
    unique_classes = np.unique(y_as_int)

    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    # define the x and y scale
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)

    # create all of the columns and rows of the grid
    xx, yy = meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    # to obtain an Lx1 vector rather than (L,) vector
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = hstack((r1, r2))

    # make predictions for the grid
    yhat = ann.predict(grid.T)
    if yhat.shape[0] > 1: 
        # If there are more than one neuron, we need to get the index of the most suitable class
        yhat = np.argmax(yhat, axis=0)  # this is to convert them to 0,1,2 etc.
    else:
        # If there is only one output neuron, we need to transform it to either 0 or 1
        yhat = np.where(yhat > 0.5, 1, 0)

    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    plt.figure()
    plt.title(title)
    plt.contourf(xx, yy, zz, cmap=colormap)

    # create scatter plot for samples from each class
    for class_value in unique_classes:
        # get row indexes for samples with this class
        # need the first element of the tuple :S
        row_ix = where(y_as_int == class_value)[0]
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.5,
                    cmap=colormap, label="Class {}".format(class_value))
    plt.legend()
    plt.draw()


def plot_costs(ann, title):
    """
    Plot the values for the cost function in the specified LR

    ann: An instance of a neural network with the costs to plot
    title: the title for the plot
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ann.costs)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('J()')
    plt.draw()
