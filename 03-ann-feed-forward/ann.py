"""
Modified/Updated by:
- Jesús Omar Cuenca Espino (JOmarCuenca)
-     jesomar.cuenca@gmail.com

Date: 24/09/2021
"""
from utils import create_structure_for_ann
import numpy as np


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
        return 1/(1 + np.exp(-z))

    def _z(self, input, theta):
        """
        Calculates the net input for all neurons at a given layer
        """
        return np.dot(theta,input)

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
        # print(self.activations)

    def _forward(self):
        # TODO: Implement the forward step. Remember that this involves calculating the activations in the next layer, one by one. 
        # After each calculation, you must leave in self.activations[someIndex] the calculated values.
        # Some points:
        # -- You start with the first layer (which already has your input X).
        # -- In each layer, you must calculate activations in a vectorized manner (i.e., the only for loop is for iterating the layers, not the examples m)
        # -- Make use of this class' activation function (which is linked to your implemented sigmoid)
        # Recall that the self.activations (a list of numpy arrays) is already configured to hold these values.
        #! DO NOT MODIFY THE FIRST POSITION AT EACH ACTIVATION LAYER, as we know it is the bias unit and it should be left equals to 1.
        
        for layer in range(len(self.hidden_layers)):
            layer_weights = self.theta[layer]
            x = self.activations[layer]
            self.activations[layer + 1][1:,:] = self._activation(self._z(x,layer_weights))

        layer_weights = self.theta[-1]
        x = self.activations[-2]

        self.activations[-1] = self._activation(self._z(x,layer_weights))

    def predict(self, X):
        """
        Performs predictions for the given X.
        X is of shape n x m, that is n features and m examples
        You must return a vector, in this case 1xm array
        """
        # Put the X dataset in the input layer
        self._initialize_activations(X)

        self._forward()

        # return np.where(self.activations[-1] < 0.5, 0, 1)
        return self.activations[-1]
