import numpy as np
import pandas as pd


def read_dataset(path):
    """
    Reads the specified dataset and returns the data as used by the Linear Regressor
    """

    # Reading the dataset
    data = pd.read_csv(path)

    X = data.iloc[:, :-1].to_numpy().T      # all but first column

    # Process to select the y column
    y = data.iloc[:, -1].to_numpy()  # the last col is y

    # this is to get (1,m) rather than an (m,) array (2d instead of 1d)
    y = y.reshape(y.shape[0], -1).T
    return X, y


def mean_normalization(X, y):
    """
    Performs mean normalization over the specified dataset.
    This will work only when the dataset has 1 dim
    """

    # min_x = np.amin(X)
    # max_x = np.amax(X)
    min_y = np.amin(y)
    max_y = np.amax(y)

    min_x = np.amin(X, axis=1)
    max_x = np.amax(X, axis=1)

    range_x = max_x - min_x
    range_y = max_y - min_y

    mean_x = np.mean(X, axis=1)
    mean_y = np.mean(y)

    normalized_X = (X - mean_x) / range_x
    normalized_y = (y - mean_y) / range_y

    return (normalized_X, normalized_y)


def plot_result(X, y, linearRegressor, title):
    """
    Plots the results of the linear regressor
    """

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X, y)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    x1 = np.amin(X)
    x2 = np.amax(X)

    y1 = linearRegressor.predict(np.array([[1], [x1]]))
    y2 = linearRegressor.predict(np.array([[1], [x2]]))

    # regression line
    plt.plot([x1, x2], [y1.flatten(), y2.flatten()], color='red')
    # plt.show()
    plt.draw()


def plot_costs(linearRegressor, title):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(linearRegressor.costs)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('J()')
    plt.draw()


def add_ones(X):
    """
    Adds the row of 1's at the top of the dataset X
    """
    the_ones = np.ones((X.shape[1], 1)).T   # (1xm) additional row of 1's
    new_X = np.row_stack((the_ones, X))     # adding the row of 1's
    return new_X
