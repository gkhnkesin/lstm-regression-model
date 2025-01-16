import numpy as np
from math import exp


def sigmoid(x: float, derivative: bool = False) -> float:
    """
    Sigmoid activation function.
    
    Args:
        x (float): Input value.
        derivative (bool): Indicates whether to compute the derivative.
        
    Returns:
        float: The output of the sigmoid function or its derivative.
    """
    f = 1 / (1 + exp(-x))
    return f * (1 - f) if derivative else f


def tanh(x: float, derivative: bool = False) -> float:
    """
    Tanh activation function.
    
    Args:
        x (float): Input value.
        derivative (bool): Indicates whether to compute the derivative.
        
    Returns:
        float: The output of the tanh function or its derivative.
    """
    f = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    return 1 - f**2 if derivative else f


class LSTMModel:
    """
    LSTM model class.
    
    Attributes:
        weights (np.ndarray): Model weights.
        h_t1 (float): Previous hidden state.
        c_t1 (float): Previous cell state.
        learning_rate (float): Learning rate.
    """
    
    def __init__(self, input_dim: int, learning_rate: float = 0.1):
        """
        Initializes the LSTM model.
        
        Args:
            input_dim (int): The dimension of the input.
            learning_rate (float): Learning rate.
        """
        self.weights = 0.1 * np.random.randn(12)
        self.h_t1 = 0.1 * np.random.randn()
        self.c_t1 = 0.1 * np.random.randn()
        self.learning_rate = learning_rate

    def forward_pass(self, x: float) -> float:
        """
        Performs the forward pass of the model.
        
        Args:
            x (float): Input value.
            
        Returns:
            float: The model's output.
        """
        w = self.weights

        z_g = w[0] * x + w[1] * self.h_t1 + w[2]
        g = tanh(z_g)

        z_i = w[3] * x + w[4] * self.h_t1 + w[5]
        i = sigmoid(z_i)

        z_f = w[6] * x + w[7] * self.h_t1 + w[8]
        f = sigmoid(z_f)

        z_o = w[9] * x + w[10] * self.h_t1 + w[11]
        o = sigmoid(z_o)

        c_t = self.c_t1 * f + g * i
        h_t = o * tanh(c_t)

        self.c_t1, self.h_t1 = c_t, h_t
        return h_t

    def _backward_pass(self, x: float, h_t: float) -> None:
        """
        Performs the backward pass and updates the weights.
        
        Args:
            x (float): Input value.
            h_t (float): The model's output.
        """
        w = self.weights
        e_delta = h_t - x

        # Example gradient calculations for some weights
        dE_dwxo = e_delta * tanh(self.c_t1) * sigmoid(w[11], derivative=True) * x

        # Gradients for other weights can be added here
        gradients = [dE_dwxo]
        self._update_weights(gradients)

    def _update_weights(self, gradients: list) -> None:
        """
        Updates the model weights based on the gradients.
        
        Args:
            gradients (list): List of computed gradients.
        """
        for i, grad in enumerate(gradients):
            self.weights[i] -= self.learning_rate * grad

    def train(self, data: np.ndarray, iterations: int) -> None:
        """
        Trains the model on the given data.
        
        Args:
            data (np.ndarray): Training data.
            iterations (int): Number of training iterations.
        """
        for epoch in range(iterations):
            for x in data:
                h_t = self.forward_pass(x)
                self._backward_pass(x, h_t)
