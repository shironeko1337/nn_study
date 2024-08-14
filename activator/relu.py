import numpy as np
from activator.activator import Activator


class ReluActivator(Activator):
    partial_derivative_source = 'input'

    def forward(self, x):
        return np.maximum(0, x)

    def backward_x(self, x):
        return np.where(x > 0, 1, 0)
