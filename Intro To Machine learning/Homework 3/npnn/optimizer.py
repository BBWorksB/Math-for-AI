"""18-661 HW5 Optimization Policies."""

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Simple SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    """

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        #  implement the SGD.apply gradients method
        for param in params:
            if param.grad is not None:
                param.value -= self.learning_rate * param.grad
        # raise NotImplementedError()


class Adam(Optimizer):
    """Adam (Adaptive Moment) optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the demoniator for numerical stability.
    """

    def __init__(
            self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self, params):
        """Initialize any optimizer state needed.

        params : np.array[]
            List of parameters that will be used with this optimizer.
        """
        # Initialize first and second moment vectors
        self.m = [np.zeros_like(param.value) for param in params]
        self.v = [np.zeros_like(param.value) for param in params]
        # raise NotImplementedError()

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        # raise NotImplementedError()
        # Increment timestep
        self.t += 1
        
        for i, param in enumerate(params):
            if param.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            

            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
