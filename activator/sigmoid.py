import numpy as np
from activator.activator import Activator


class SigmoidActivator(Activator):
    partial_derivative_source = 'output'

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    # using y is easier to get
    def backward_y(self, y):
        return y * (1 - y)
