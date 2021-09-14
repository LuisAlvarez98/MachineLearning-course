from utils import add_ones, plot_costs, plot_decision_boundary, read_dataset
from LogisticRegressor import LogisticRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # To generate same results
    np.random.seed(0)

    # First run (dataset-1)
    # Reading the dataset
    X, y = read_dataset('dataset-1.csv')
    X_with_ones = add_ones(X)

    # Setting hyperparameters
    alpha = 0.001
    epochs = 50000

    # Creating regressor
    lr = LogisticRegressor(alpha, epochs)

    # Fitting
    lr.fit(X_with_ones, y)

    # Associated plots
    plot_decision_boundary(
        X_with_ones.T, y.T, lr, "LogisticRegressor alpha={} epochs={}".format(alpha, epochs))
    plot_costs(lr, 'Cost function')

    # Predicting some examples
    idx = [1, 3, 7]
    X_test = X_with_ones[:, idx]
    y_pred = lr.predict(X_test)
    print('{} predicted as {} (was {})'.format(X_test, y_pred, y[:, idx]))


    # To generate same results
    np.random.seed(0)
    # Second run (dataset-2-adjusted.csv)
    
    # Reading the dataset
    X_with_ones, y = read_dataset('dataset-2-modified.csv')
    
    # Setting hyperparameters
    alpha = 0.001
    epochs = 5000
    
    # Creating regressor
    lr = LogisticRegressor(alpha, epochs, regularize=True)
    
    # Fitting
    lr.fit(X_with_ones, y)
    
    # Associated plots
    original_x, _ = read_dataset('dataset-2.csv')
    original_x = add_ones(original_x)
    plot_decision_boundary(
        original_x.T, y.T, lr, "LogisticRegressor alpha={} epochs={}".format(alpha, epochs), is_polynomial=True)
    plot_costs(lr, 'Cost function ')

    # Predicting some examples
    idx = [1, 70, 110]
    X_test = X_with_ones[:, idx]
    y_pred = lr.predict(X_test)
    print('{} predicted as {} (was {})'.format(X_test, y_pred, y[:, idx]))

    plt.show()
