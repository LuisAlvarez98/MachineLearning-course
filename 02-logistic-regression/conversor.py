"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 14/09/2021
"""

import pandas as pd
import numpy as np


def ds_conversor():
    """
    Converts the 2d dataset to a new dataset with features from a polynomial degree
    """
    data = pd.read_csv('dataset-2.csv')
    X1 = data.iloc[:, 0].to_numpy()
    X2 = data.iloc[:, 1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    degree = 6
    expanded_features = [np.ones(X1.shape[0])]
    names = ["x0"]

    for i in range(1, degree+1):
        for j in range(0, i+1):
            new_feature = X1 ** (i-j) * (X2 ** j)
            col_name = 'x1^{}x2^{}'.format((i-j), j)
            names.append(col_name)
            print('Getting X1^{} * X2^{}'.format((i-j), j))
            expanded_features.append(new_feature)

    return expanded_features, y, names


if __name__ == "__main__":
    new_features, y, names = ds_conversor()
    zipped_list = list(zip(*new_features))
    new_ds = pd.DataFrame(data=zipped_list, columns=names)
    new_ds["y"] = y
    new_ds.to_csv('./dataset-2-modified.csv', index=False)
