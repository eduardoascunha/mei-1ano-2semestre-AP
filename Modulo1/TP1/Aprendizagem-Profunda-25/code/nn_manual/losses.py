#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####
#### Código baseado no material da UC Aprendizagem Profunda 24-25
####

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (p - y_true) / (p * (1 - p))

    

    
    
