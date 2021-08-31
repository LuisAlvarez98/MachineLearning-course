import numpy as np
from utils import add_ones


def predict_within_normalized_values(X_test, X, y, lr):
    """
    Predicts values for a Linear Regressor that is using a mean normalized dataset.
    X_test: an (n x m') array with m samples of n features that must be used for prediction. Their values ARE IN THE ORIGINAL DOMAIN.
    X: the original dataset as an (n x m) array. It should be used to extract the min, max, mean, range, etc. values to calculate normalization.
    y: the original/right labels in the dataset as an (1 x m) vector, where m is the number of samples/examples
    lr: a Linear Regressor object using mean normalized values that will be used to perform the prediction, it should be ready to use (already trained). Its internal values ARE IN THE NORMALIZED DOMAIN.

    TODO: You must implement the required code to take the X_test samples in the original domain, transform them to the normalized domain to perform the prediction, and then transform the predictions back to the original problem domain.
    The result must be a (1 x m') array that includes the m' predictions for the given m' samples.

    """

    min_x = np.amin(X, axis=1)
    max_x = np.amax(X, axis=1)
    mean_x = np.mean(X, axis=1)

    min_y = np.amin(y)
    max_y = np.amax(y)
    mean_y = np.mean(y)
    range_y = max_y - min_y

    range_x = max_x - min_x
    X_test_normalized = (X_test - mean_x) / range_x
    X_test_normalized = add_ones(X_test_normalized)

    normalized_y_test = lr.predict(X_test_normalized)
    denormalized_y_test = (normalized_y_test * range_y) + mean_y
    return denormalized_y_test
