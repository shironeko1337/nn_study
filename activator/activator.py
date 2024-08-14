import numpy as np
from abc import ABC, abstractmethod

class Activator:
    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def backward_y(self,y):
        pass
