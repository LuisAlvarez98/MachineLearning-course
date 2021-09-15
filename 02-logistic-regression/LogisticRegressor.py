"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 14/09/2021
"""

import numpy as np
from progressbar import progressbar, streams

# Setup Progressbar wrapper function
streams.wrap_stderr()


class LogisticRegressor():
    def __init__(self, alpha=0.1, epochs=1, regularize=False, reg_factor=0.1):
        """
        Constructor

        alpha: learning rate
        epochs: the number of epochs (how many times does the LR see the whole dataset)
        regularize: whether or not the LR should implement regularization during training
        reg_factor: the regularization factor to use (lambda); only used if regularize is set to True
        """
        self.alpha = alpha
        self.epochs = epochs
        self.regularize = regularize
        self.reg_factor = reg_factor
        self.costs = []
        self.theta = None

    def _cost_function(self, hyp, y) -> float:
        """
        Calculates the cost function (error) for the predicted values (hyp) when compared against the right labels (y).
        hyp: the values predicted by the current configuration of the LR
        y: the correct labels

        TODO: Implement this function to calculate the cost value for the LR. 
        Recall that the LR might be set to perform regularization, so you must have two cases: one for when regularization is required and one for when it is not.
        This returns an scalar
        """
        m = y.size

        cost = -1/m
        tempA = np.dot(y, np.log(hyp).T)
        tempB = np.dot((1 - y), np.log((1 - hyp)).T)
        cost *= (tempA + tempB).flatten()[0]

        if(self.regularize):
            regulator = (np.dot(self.theta, self.theta.T)).flatten()
            cost += (self.reg_factor/(2*m))*regulator[0]

        return cost

    def _cost_function_derivative(self, hyp, y, X, m):
        """
        Calculates the derivative (gradient) of the cost funcion. It supports the case for regularization.

        hyp: the predicted values
        y: the target class (right values) for the data
        X: the input dataset
        m: the number of examples in the dataset

        """
        return (np.dot(hyp - y,X.T)).T/m
        

    def _hypothesis(self, X) -> np.ndarray :
        """
        Calculates the hypothesis for the given dataset using the current LR configuration (theta parameters). This is the sigmoid function.
        X: the dataset to employ. It is an (n x m) array.
        """

        return 1/(1 + np.exp(-np.dot(self.theta.T,X)))
        
    def fit(self, X, y):
        """
        Fits the Logistic Regressor to the values in the dataset
        X: is an (n x m) vector, where n is the number of features and m is the number of samples/examples
        y: is an (1 x m) vector, where m is the number of samples/examples
        """

        # m is the number of samples, n is the number of features, y is (1 x m)
        n, m = X.shape[0], X.shape[1]

        # self.theta is an (n x 1) vector (one per feature)
        self.theta = np.random.uniform(-10, 10, (n, 1))

        for _ in progressbar(range(self.epochs)):
            # Get predictions
            hyp = self.predict(X)

            # Calculate cost
            cost = self._cost_function(hyp, y)

            # get gradient, an (nx1) array
            gradient : np.ndarray = self._cost_function_derivative(hyp, y, X, m)
            
            # delta/update rule
            self.theta = self.theta - self.alpha*gradient

            self.costs.append(cost)

        print("Final theta is {} (cost: {})".format(self.theta.T, cost))

    def predict(self, X):
        """
        Predicts the classes for the samples in the dataset X.

        X: an (n x m') array with the dataset to predict, m' samples of n dimensions.
        """

        THRESHOLD = 0.5

        hyp = self._hypothesis(X)

        # hyp[hyp > THRESHOLD] = 1
        # hyp[hyp <= THRESHOLD]= 0

        return hyp
        
