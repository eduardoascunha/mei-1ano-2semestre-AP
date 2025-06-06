#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####
#### Código baseado no material da UC Aprendizagem Profunda 24-25
####

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer (Layer):
    
    def __init__(self, n_units, input_shape = None, l1_lambda = 0.0, l2_lambda = 0.0,init_weights="he"): 
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.init_weights = init_weights

        self.l1_lambda = l1_lambda  # Coeficiente de regularização L1
        self.l2_lambda = l2_lambda  # Coeficiente de regularização L2

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None


    def initialize(self, optimizer): # init_weights = "he" ou "xavier"
        # otimizacao de pesos iniciais

        if self.init_weights == "he":
            limit = np.sqrt(2 / self.input_shape()[0])  # He initialization
            self.weights = np.random.normal(0, limit, (self.input_shape()[0], self.n_units))
        
        elif self.init_weights == "xavier":
            limit = np.sqrt(6 / (self.input_shape()[0] + self.n_units))  # Xavier/Glorot initialization
            self.weights = np.random.uniform(-limit, limit, (self.input_shape()[0], self.n_units))
        
        else:
            print("Função de ativação passada, errada!")
            raise ValueError

        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)  # dE/dX
        weights_error = np.dot(self.input.T, output_error)  # dE/dW
        bias_error = np.sum(output_error, axis=0, keepdims=True)  # dE/dB

        # regularização l1
        if self.l1_lambda > 0:
            weights_error += self.l1_lambda * np.sign(self.weights)

        # regularização L2
        if self.l2_lambda > 0: 
            weights_error += self.l2_lambda * self.weights

        # Update parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error
 
    def output_shape(self):
         return (self.n_units,) 
    

class DropoutLayer(Layer):
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self._input_shape = None # isto foi só pra nao dar erro, nao percebi o motivo disto 
        self.dropout_rate = dropout_rate
        self.mask = None
        self.input = None
        self.output = None
    
    #def initialize(self, optimizer):
    #    """
    #    nada para inicializar
    #    """
    #    return self
    
    def parameters(self):
        # nao ha parametros treinaveis, tem de ser iniciada pra nao dar erro
        return 0
    
    def forward_propagation(self, inputs, training=True):
    
        self.input = inputs
        
        # apenas aplicavel em modo treino
        if training:
            # gera uma mascara aleatoria de 0s e 1s confrome a dropout rate
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            
            # aplica a mascara e normaliza os outputs
            self.output = inputs * self.mask / (1 - self.dropout_rate)
        
        else:
            self.output = inputs
            
        return self.output
    
    def backward_propagation(self, output_error):
        # aplica o output error à mascara da camada anterior
        return output_error * self.mask / (1 - self.dropout_rate)
    
    def output_shape(self):     
        return self.input_shape()

class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, trainable=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        self.w_opt = None

        # Initialize embedding weights
        if embedding_matrix is not None:
            self.weights = embedding_matrix  # Pre-trained GloVe embeddings
        else:
            self.weights = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    def initialize(self, optimizer):
        self.w_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) 

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        batch_size = inputs.shape[0]
        
        # Handle both one-hot encoded vectors and indices
        if len(inputs.shape) > 1 and inputs.shape[1] > 1 and np.sum(inputs) > batch_size:
            # This is likely one-hot encoded input
            self.input_indices = np.argmax(inputs, axis=1)
        else:
            # This is likely already indices
            self.input_indices = inputs.flatten().astype(int)
            
        # Get embeddings for these indices - output should be (batch_size, embedding_dim)
        self.output = self.weights[self.input_indices]
        return self.output
        
    def backward_propagation(self, output_error):
        if self.trainable:
            # Create a gradient matrix for embeddings
            weights_error = np.zeros_like(self.weights)
            
            # Accumulate gradients for each used embedding
            for i, idx in enumerate(self.input_indices):
                weights_error[idx] += output_error[i]
            
            # Update embeddings with optimizer
            if self.w_opt:
                self.weights = self.w_opt.update(self.weights, weights_error)
            
        # For backward compatibility, return error matrix with same shape as input
        return np.zeros_like(self.input)
    
    def output_shape(self):
        # The output shape will be (embedding_dim,) since we're outputting
        # one embedding vector per input sample
        return (self.embedding_dim,)

class EmbeddingLayerRNN(Layer):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix=None, trainable=False, input_shape=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        self.w_opt = None
        self._input_shape = input_shape

        # Initialize weights (embedding matrix)
        if embedding_matrix is not None:
            self.weights = embedding_matrix
        else:
            self.weights = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    def initialize(self, optimizer):
        if self.trainable:
            self.w_opt = copy.deepcopy(optimizer)
        return self
    
    def forward_propagation(self, inputs, training=True):
        """
        Inputs: (batch_size, sequence_length)
        Outputs: (batch_size, sequence_length, embedding_dim)
        """
        self.input = inputs  # Store input for backpropagation
        batch_size, sequence_length = inputs.shape
        self.output = self.weights[inputs]  # Efficient indexing for embeddings
        return self.output
    
    def backward_propagation(self, output_error):
        """
        Inputs: (batch_size, sequence_length, embedding_dim)
        """
        if self.trainable:
            grad_w = np.zeros_like(self.weights)
            batch_size, sequence_length, _ = output_error.shape
            
            for i in range(batch_size):
                for t in range(sequence_length):
                    idx = self.input[i, t]
                    grad_w[idx] += output_error[i, t]
            
            self.weights = self.w_opt.update(self.weights, grad_w)
        
        return np.zeros_like(self.input)
    
    def output_shape(self):
        return (self._input_shape[0], self._input_shape[1], self.embedding_dim)
    
    def parameters(self):
        return np.prod(self.weights.shape) if self.trainable else 0