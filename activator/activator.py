import numpy as np
from abc import ABC, abstractmethod

class Activator:
    # the easiest way to get derivative
    # if by y then use input
    # if by x then use output
    partial_derivative_source = 'input'

    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def backward(self,input,output):
      if self.partial_derivative_source == 'input':
        return self.backward_x(input)
      return self.backward_y(output)

    @abstractmethod
    def backward_x(self,y):
        pass

    @abstractmethod
    def backward_y(self,y):
        pass

