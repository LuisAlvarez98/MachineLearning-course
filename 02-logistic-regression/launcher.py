"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 14/09/2021
"""

from utils import add_ones, plot_costs, plot_decision_boundary, read_dataset
from LogisticRegressor import LogisticRegressor
import numpy as np
import matplotlib.pyplot as plt

STANDARD_ALPHA = 0.001
STANDARD_EPOCHS = 50000

def part1(alpha : float, epochs : float, reg_factor = None):
    # To generate same results
    np.random.seed(0)

    # First run (dataset-1)
    # Reading the dataset
    X, y = read_dataset('./dataset-1.csv')
    X_with_ones = add_ones(X)

    reg = reg_factor != None

    # Creating regressor
    lr = LogisticRegressor(alpha, epochs, regularize=reg, reg_factor=(reg_factor if reg else -1.0))

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

def part2(alpha : float, epochs : float, reg_factor = None):
    # To generate same results
    np.random.seed(0)
    # Second run (dataset-2-adjusted.csv)
    
    # Reading the dataset
    X_with_ones, y = read_dataset('./dataset-2-modified.csv')
    
    # Setting hyperparameters
    reg = reg_factor != None
    
    # Creating regressor
    lr = LogisticRegressor(alpha, epochs, regularize=reg, reg_factor=(reg_factor if reg else -1.0))
    
    # Fitting
    lr.fit(X_with_ones, y)
    
    # Associated plots
    original_x, _ = read_dataset('./dataset-2.csv')
    original_x = add_ones(original_x)
    plot_decision_boundary(
        original_x.T, y.T, lr, "LogisticRegressor alpha={} epochs={}".format(alpha, epochs), is_polynomial=True)
    plot_costs(lr, 'Cost function ')

    # Predicting some examples
    idx = [1, 70, 110]
    X_test = X_with_ones[:, idx]
    y_pred = lr.predict(X_test)
    print('{} predicted as {} (was {})'.format(X_test, y_pred, y[:, idx]))

if __name__ == "__main__":
    

    ## Optimal Values
    part1(alpha=STANDARD_ALPHA, epochs=int(np.floor(STANDARD_EPOCHS * 2)))

    ## Experiments
    # part1(alpha=STANDARD_ALPHA, epochs=STANDARD_EPOCHS // 10, reg_factor=0.0001)

    separator = " experimento parte 2 ".upper().center(50,"=")

    print(f"\n{separator}\n")

    ## Optimal Values
    # part2(alpha=0.01, epochs=STANDARD_EPOCHS, reg_factor=0.1)

    ## Experiments
    part2(alpha=0.01, epochs=STANDARD_EPOCHS, reg_factor=0.1)
    

    # plt.show()
