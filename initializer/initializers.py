import numpy as np

# kaiming initialization
def he(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

def xavier(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (output_size + input_size))

# random number from 0 to 1
def uniform(input_size, output_size):
    return np.random.rand((input_size, output_size))

# all zero
def zeros(input_size, output_size):
    return np.zeros((input_size, output_size))
