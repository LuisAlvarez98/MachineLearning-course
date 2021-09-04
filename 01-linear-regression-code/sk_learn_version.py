"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 03/09/2021
"""

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

data = pd.read_csv('01-sea-temperature.csv')

X = np.array(data['salinity']).reshape(-1, 1)
y = np.array(data['temperature']).reshape(-1, 1)

regr = LinearRegression()

regr.fit(X, y)
y_pred = regr.predict([[33.5]])
print('Predicted is {}' .format(y_pred))
# print(regr.score(X, y))
