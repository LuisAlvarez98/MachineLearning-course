import numpy as np


class LinearRegressor():
    def __init__(self, alpha=0.1, epochs=1):
        self.alpha = alpha
        self.epochs = epochs
        self.costs = []
        self.theta = None



    def _cost_function(self, y_pred, y, m):
        """
        Gets the cost for the predicted values when contrasted with the correct ones.
        y_pred: An (1 x m) vector that corresponds to the values predicted by the Linear Regressor
        y: An (1 x m) vector that corresponds to the y (right) values in the dataset
        m: the number of samples (it could be also inferred from the shape of y or y_pred)

        TODO: You must implement the cost function and return an scalar that corresponds to the error produced by the Linear Regressor with its current configuration
        """
        cost = 0
        temp = y_pred - y 
        cost = np.dot(temp, temp.T)
        return 1/2*m * cost

    def _hypothesis(self, X):
        """
        Calculates the hypothesis for the given examples using the current self.theta values.
        X: an m x n array of m samples/examples with n features each.

        TODO: you must return a (1 x m) array, which corresponds to the estimated value for each of the m samples
        """
        # * is element wise multiplication
        # numpy.dot(), or @ operator will work

        return np.dot(self.theta, X)

    def _cost_function_derivative(self, y_pred, y, X, m):
        """
        Calculates the derivatives (gradient) of the cost function through the obtained/predicted values.
        y_pred: an (1 x m) array with the predicted values for X dataset
        y: an (1 x m) array with the right values for X dataset
        X: the input dataset
        m: the number of samples in the dataset

        TODO: You must implement the calculation of derivatives. An (n x 1) array that corresponds to the gradient of current theta values (the derivative per theta parameter) must be returned.
        """

        temp = y_pred - y # (1 x m)

        temp2 = np.dot(temp, X.T) # (1 x m) * (m x n) = (1 x n)

        gradient = self.theta - self.alpha * (temp2 / m)
        return gradient

    def fit(self, X, y):
        """
        Fits the linear regressor to the values in the dataset.
        X: is an (n x m) vector, where n is the number of features and m is the number of samples/examples
        y: is an (1 x m) vector, where m is the number of samples/examples

        TODO: You need to provide an implementation that in each epoch is updating the values for the theta parameters by using the hypothesis and cost function functions
        """

        n, m = X.shape[0], X.shape[1]

        # theta is (nx1) (one theta per dimension)
        self.theta = np.random.uniform(-10, 10, (n, 1))

        for i in range(self.epochs):
            # Get predictions
            y_pred = predict(X)

            # calculate cost
            cost = _cost_function(y_pred, y, m)
            

            # gradient is an (n) x 1 array, it refers to the derivate per theta
            gradient = _cost_function_derivative(y_pred, y, X, m):

            # delta/update rule
            # self.theta = ...

            self.costs.append(cost)
            pass

        print("Final theta is {} (cost: {})".format(self.theta.T, cost))

    def predict(self, X):
        """
        Predicts the values for the given X samples using the current configuration of the Linear Regressor.

        X: an (n x m') array with m' samples of n dimensions whose value must be predicted.

        TODO: You must return a (1 x m') array that includes the predictions for the given m' samples.
        """

        # ! You could simply call the hypothesis here
        empty_predictions = np.zeros((1,X.shape[1]))
        return empty_predictions
