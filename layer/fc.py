import numpy as np


class FullConnectedLayer:
    # For all layers
    # self.activator
    # self.input_size
    # self_output_size
    # self.W - weight matrix (input_size,output_size)
    # self.b - bias (output_size,1)
    # self.W_grad - gradient of weight
    # self.B_grad - gradient of bias
    # self.delta: 输入节点的误差项，用于计算连接到该列节点的网络权重矩阵的梯度
    #   1. 节点的列数会比神经网络的层数多1，所以最后一层对应两列节点，一列是特殊的输出节点
    #   2. 对于每一层，输出都完全等于下一层的输入，这是一种冗余
    #   3. 对于第一层其实不用计算误差项, added is_first for flag
    # Only for the last layer
    # is_last: true
    # self.label - labels (output_size_of_network,1)
    # output_delta - delta for the output layer, only exists if it's the last layer
    def __init__(self, activator, input_size, output_size, is_first = False, is_last = False):
        self.activator = activator
        self.input_size = input_size
        self.output_size = output_size
        self.is_first = is_first
        self.is_last = is_last
        self.initialize()
        self.label = None
        self.output_delta = None
        self.delta = None

    def initialize(self):
        # Here W must be initialized to 0 for mnist dataset, because a 0-1 randomized matrix
        # would cause result of sigmoid function to intially be 1 (1/(1 + e^(-x))) when x is large
        self.W = np.zeros((self.input_size, self.output_size))
        self.b = np.zeros((self.output_size,1))

    # predict
    def forward(self):
        self.output =  self.activator.forward(np.dot(self.input.T, self.W).T + self.b)

    # back propagation
    def backward(self,next_delta):
        # For the last layer, we calculate two type of delta,
        # one is from the final output, which directly calculates the loss,
        # one is from the delta from next layer, for calculating the weight
        # gradient of current layer.
        #
        # For the other layers, we only calculate the later one.
        if self.is_last:
          self.output_delta = self.activator.backward_y(self.output) * (self.label - self.output)
          next_delta = self.output_delta
        if not self.is_first:
          self.delta = self.activator.backward_y(self.input) * np.dot(self.W,next_delta)
        self.W_grad = np.dot(self.input, next_delta.T) # ∇w = xi ⊗ δj
        self.B_grad = next_delta # x = 1
        return self.delta

    def update(self,learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.B_grad

