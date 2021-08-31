from prediction import predict_within_normalized_values
from utils import add_ones, mean_normalization, plot_costs, plot_result, read_dataset
from LinearRegressor import LinearRegressor
import numpy as np

if __name__ == "__main__":
    # To generate same results
    np.random.seed(0)

    # Setting hyperparameters
    alpha = 0.001
    epochs = 50000

    # Reading the dataset
    X, y = read_dataset('01-sea-temperature.csv')

    # First run without mean normalization
    # Creating regressor
    lr1 = LinearRegressor(alpha, epochs)
    # Adding the 1's to Xs
    X_with_ones = add_ones(X)
    # Fitting
    lr1.fit(X_with_ones, y)
    # Associated plots
    plot_result(X, y, lr1, "Without mean normalization alpha={}, epochs={}".format(
        alpha, epochs))
    plot_costs(lr1, "Cost function without mean normalization")

    # Second run with mean normalization
    # Creating regressor
    lr2 = LinearRegressor(alpha, epochs)
    # Mean-normalizing dataset
    normalized_X, normalized_y = mean_normalization(X, y)
    # Adding the 1's to Xs
    norm_x_with_ones = add_ones(normalized_X)
    # Fitting
    lr2.fit(norm_x_with_ones, normalized_y)
    # Associated plots
    plot_result(normalized_X, normalized_y, lr2,
                "With mean normalization alpha={} epochs={}".format(alpha, epochs))
    plot_costs(lr2, "Cost function with mean normalization")

    # Predicting a value with the normalized dataset
    X_test = np.array([[30, 31, 33.5]])
    y_pred = predict_within_normalized_values(X_test, X, y, lr2)
    print('{} predicted as {}'.format(X_test, y_pred))

    import matplotlib.pyplot as plt
    plt.show()
