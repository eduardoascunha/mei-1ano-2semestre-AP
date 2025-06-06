####
#### Código baseado no material da UC Aprendizagem Profunda 24-25
####

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        input = np.clip(input, -500, 500) # anti overflows
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        sigmoid = self.activation_function(input)
        return sigmoid * (1 - sigmoid)


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)

    
class TanhActivation(ActivationLayer):
    
    def activation_function(self, input):
        return np.tanh(input)
    
    def derivative(self, input):
        return 1 - np.tanh(input) ** 2