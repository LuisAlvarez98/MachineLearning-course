import numpy as np
import pandas as pd
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack


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


def add_ones(X):
    """
    Adds the row of 1's at the top of the dataset X
    """
    the_ones = np.ones((X.shape[1], 1)).T   # (1xm) additional row of 1's
    new_X = np.row_stack((the_ones, X))     # adding the row of 1's
    return new_X


def plot_costs(logisticRegressor, title):
    """
    Plot the values for the cost function in the specified LR

    logisticRegressor: An instance of LR with the costs to plot
    title: the title for the plot
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(logisticRegressor.costs)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('J()')
    plt.draw()


def plot_decision_boundary(X, y, classifier, title, is_polynomial=False):
    """
    Plots the decision boundary for a 2d dataset
    X: the dataset to plot, in this case is (m x n)
    y: the class labels
    classifier: a logistic regressor object that will be used to generate the surface
    title: the title for the plot
    is_polynomial: whether the dataset is using the polynomial feature engineering

    """
    import matplotlib.pyplot as plt
    # https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
    # decision surface for logistic regression on a binary classification dataset
    colormap = 'Greens'
    # unique classes:
    unique_classes = np.unique(y)
    # define bounds of the domain
    # We start at 1 because in 0 we have the bias
    min1, max1 = X[:, 1].min()-1, X[:, 1].max()+1
    min2, max2 = X[:, 2].min()-1, X[:, 2].max()+1
    # defining x and y scale
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)
    # creating cols and rows for the grid
    xx, yy = meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    # to obtain an Lx1 vector rather than (L,) vector
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    if is_polynomial:
        # before stacking, we need to convert r1, r2, to the polynomial representation
        polynomial_features = convert_to_polynomial(r1, r2)

        # Now we can horizontally stack vectors to create the grid
        grid = hstack(polynomial_features)
    else:
        # horizontal stack vectors to create x1,x2 input for the model
        grid = hstack((r1, r2))

    # make predictions for the grid
    the_ones = np.ones((grid.shape[0], 1))
    yhat = classifier.predict(np.column_stack((the_ones, grid)).T)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    plt.figure()
    plt.title(title)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap=colormap)

    # create scatter plot for samples from each class
    for class_value in unique_classes:
        # get row indexes for samples with this class
        # need the first element of the tuple :S
        row_ix = where(y == class_value)[0]
        # create scatter of these samples
        plt.scatter(X[row_ix, 1], X[row_ix, 2], alpha=0.5,
                    cmap=colormap, label=class_value)
    plt.draw()


def convert_to_polynomial(x1, x2):
    """
    generates a set of features according to a polynomial degree from the specified 2 features
    """
    degree = 6
    expanded_features = []

    for i in range(1, degree+1):
        for j in range(0, i+1):
            new_feature = x1 ** (i-j) * (x2 ** j)
            # print('Getting X1^{} * X2^{}'.format((i-j), j))
            expanded_features.append(new_feature)

    return expanded_features
