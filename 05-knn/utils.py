"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 18/10/2021
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
