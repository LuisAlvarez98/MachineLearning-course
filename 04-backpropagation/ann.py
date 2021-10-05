"""
Modified by:
- Jesús Omar Cuenca Espino    A01378844
- Luis Felipe Alvarez Sanchez A01194173
- Juan José González Andrews  A01194101
- Rodrigo Montemayor Faudoa   A00821976

Date: 05/10/2021
"""

from utils import create_structure_for_ann
import numpy as np
from progressbar import progressbar, streams

# Setup Progressbar
streams.wrap_stderr()


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers_conf, output_neurons, learning_rate, regularization_rate, epochs) -> None:
        """
        Constructor
        """
        self.input_neurons = input_neurons
        self.hidden_layers = hidden_layers_conf
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.epochs = epochs
        self.costs = None
        self.theta = None
        self.activations = None

    def _sigmoid(self, z):
        """
        Calculates the sigmoid for the elements in the array z
        """
        return 1 / (1 + np.exp(-z))

    def _z(self, input, theta):
        """
        Calculates the net input for all neurons at a given layer
        """
        return theta @ input

    def _activation(self, z):
        """
        Calculates the activation for net input z (an array)
        """
        return self._sigmoid(z)

    def _initialize_weights(self):
        """
        Initializes the weights for all the ANN structure
        """
        self.theta = create_structure_for_ann(self)

    def _initialize_activations(self, X):
        """
        Initializes the activation matrices for neurons at all layers for handling all examples
        """
        m = X.shape[1]
        self.activations = []

        # activations at layer 1 are the inputs
        a_1 = X
        # we create and add the ones to input layer
        biases = np.ones(m)
        a_1 = np.vstack((biases, a_1))
        self.activations.append(a_1)

        # We create each hidden layer and add the bias on top of each one
        for i in range(len(self.hidden_layers)):
            a_i = np.zeros((self.hidden_layers[i]+1, m))
            # This set the bias
            a_i[0, :] = 1.0
            self.activations.append(a_i)

        self.activations.append(np.zeros((self.output_neurons, m)))

    def _forward(self):
        # OLD_TODO: Implement the forward step. Remember that this involves calculating the activations in the next layer, one by one.
        # After each calculation, you must leave in self.activations[someIndex] the calculated values.
        # Some points:
        # -- You start with the first layer (which already has your input X).
        # -- In each layer, you must calculate activations in a vectorized manner (i.e., the only for loop is for iterating the layers, not the examples m)
        # -- Make use of this class' activation function (which is linked to your implemented sigmoid)
        # Recall that the self.activations (a list of numpy arrays) is already configured to hold these values.
        #! DO NOT MODIFY THE FIRST POSITION AT EACH ACTIVATION LAYER, as we know it is the bias unit and it should be left equals to 1.
        for i in range(0, len(self.hidden_layers)):
            a_i = self.activations[i]  # a_0
            z_next = self._z(a_i, self.theta[i])  # a_0 x theta[0]
            a_next = self._activation(z_next)

            # update all activation but no bias
            self.activations[i+1][1:] = a_next

        a_i = self.activations[i+1]
        z_next = self._z(a_i, self.theta[i+1])
        a_next = self._activation(z_next)
        self.activations[i+2] = a_next

    def _backprop(self, y):
        # TODO Implement the back prop step. This fun does not produce output, it should at the end update the weights in self.theta
        # Hints: 
        # - You can get m from y parameter
        # - predictions are in the last element of the list self.activations
        # - You can (and it is recommended) create auxiliary lists or matrices for the individual errors (delta),  Δ (Delta) 
        #   and D (refer to slides to remember what they are)
        # ---- i.e., create as many tmp and auxiliary variables as needed
        # - To create the Delta matrix and match the same sizes of the weights, you can use the create_structure_for_ann method:
        # ---- Delta = create_structure_for_ann(self)
        # - You can use for loops that go in reverse
        # - Once you get D, you can use it with the update rule to update theta parameters
        # - Remember, here we don't update the weights coming from bias unit, you can use slicing [1:], [:,:1], etc to select the 
        #   portions of the arrays you want to update.
        # - Some code and comments are provided here as a starting point, but you can delete it and use your own
        
        
        # m = y.shape[1]
        # delta = []

        # calculate the error in output layer and append it to delta
        # delta_i = something
        # delta.append(delta_i)

        # Iterate backwards for each layer, we stop at layer 1 (excluding it)
        # for each layer going backwards:
        #   get corresponding thetas and caused errors in next layer
        #   multiply the previous together and also by the sigmoid derivative
        #     (that is a * (1 - a), you must select the correct activations here)
        #   at this point you got the deltas for this layer, appendit to delta list
        

        # Create your Delta Δ matrix
        # Delta = create_structure_for_ann(self)
        # Compute Δ = Δ + activations*delta_next_layer
        

        # Create matrix D from Δ matrix, there is some regularization here

        # Use D for gradient descent's update rule, no update for weights from bias units to match PDF results
        pass

    def predict(self, X):
        """
        Performs predictions for the given X.
        X is of shape n x m, that is n features and m examples
        You must return a vector, in this case 1xm array
        """
        # Put the X dataset in the input layer
        self._initialize_activations(X)
        # OLD_TODO Implement the following steps:
        # -- Perform the forward pass and
        # -- Return the last element in the list of activations, that is,
        # --   the numpy array that corresponds to the activations in the output layer
        # --   remember that the list of activations is in self.activations

        # Do the forward
        self._forward()
        # Gets the final activations as the predictions
        predicted = self.activations[-1]
        # return np.argmax(predicted, axis=0)
        return predicted

    def _cost_function(self, y):
        # TODO: Implement the cost function seen in class, indeed it is with regularization. 
        # TODO: You should implement a vectorized version, for loops are not needed.
        # TODO: Cost function always returns a scalar
        # Hints: 
        # - the predictions (y_pred in previous assignments) are in the activations of last layer, you can access them via self.activations
        # - you can get the value of m (number of examples) from the y parameter
        # - do not forget: We do not regularize the bias units weights
        return 0

    def fit(self, X, y, initialTheta=None):
        """
        Performs the training of the ANN. Optionally, it can load the initialTheta parameter with a configuration for out theta weights
        """
        # We initialize the weights, randomly
        self._initialize_weights()
        # We initialize the activations, and put X in the input layer as activations_1
        self._initialize_activations(X)

        # We copy an initial theta (if there is any).
        if initialTheta is not None:
            self.theta = initialTheta

        self.costs = []

        # Perform the training
        for _ in progressbar(range(self.epochs)):
            self._forward()
            cost = self._cost_function(y)
            self.costs.append(cost)
            self._backprop(y)

        print('final theta is {}'.format(self.theta))
        print('final cost is {} '.format(cost))
