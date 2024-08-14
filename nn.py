from layer.fc import FullConnectedLayer
from activator.sigmoid import SigmoidActivator

class Network:
    def __init__(self, nodes = []):
        self.layers = []
        for i in range(len(nodes) - 1):
            self.layers += [
                FullConnectedLayer(
                  SigmoidActivator(),
                  nodes[i],
                  nodes[i + 1],
                  is_first = i == 0,
                  is_last = i == len(nodes) - 2
                )
            ]

    def set_sample(self,sample):
        self.layers[0].input = sample

    def set_label(self,label):
        self.layers[-1].label = label

    # train the network once using sample and label with a specific learning rate
    def train_once(self, sample = [], label = [],learning_rate = 0.1):
        self.set_sample(sample)
        self.set_label(label)

        self.forward()
        self.backward()
        self.update(learning_rate)

    def forward(self):
        # forward and set the next layer's input by the current layer's
        # output if there is one
        for i in range(len(self.layers)):
            self.layers[i].forward()
            if i < len(self.layers) - 1:
              self.layers[i + 1].input = self.layers[i].output

    def backward(self):
        delta = None
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def update(self,learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)



