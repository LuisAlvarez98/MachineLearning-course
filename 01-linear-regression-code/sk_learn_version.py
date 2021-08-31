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
