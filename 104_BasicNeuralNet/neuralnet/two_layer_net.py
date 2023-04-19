import numpy as np

from classifier import Classifier
from layers import fc_forward, fc_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """

    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################

        # Initialize with a seed
        np.random.seed(2)
        self.params = {}

        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(num_classes)

        self.W1 = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.W2 = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))

        self.params['b1'] = self.b1
        self.params['b2'] = self.b2

        self.params['W1'] = self.W1
        self.params['W2'] = self.W2

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        params = None
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################
        params = self.params
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return params

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # FC layer
        hidden, cache1 = fc_forward(X, W1, b1)
        # ReLU
        pre_relu = np.copy(hidden)
        hidden[hidden < 0] = 0
        # FC Layer
        scores, cache2 = fc_forward(hidden, W2, b2)
        cache = [hidden, cache1, cache2]
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################

        # W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden, cache1, cache2 = cache
        grad_x2, grad_w2, grad_b2 = fc_backward(grad_scores, cache2)

        hidden1 = np.matmul(grad_scores, W2.T)
        hidden1[hidden <= 0] = 0

        grad_x1, grad_w1, grad_b1 = fc_backward(hidden1, cache1)

        grads = {
            'W2': grad_w2,
            'b2': grad_b2,
            'W1': grad_w1,
            'b1': grad_b1,
        }
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads
