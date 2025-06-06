#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####
#### Código baseado no material da UC Aprendizagem Profunda 24-25
####

import numpy as np

class Optimizer:

    def __init__(self, learning_rate = 0.01,  momentum = 0.90):
        self.retained_gradient = None
        self.learning_rate = learning_rate
        self.momentum = momentum
 
    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        #self.retained_gradient = grad_loss_w
        return w - self.learning_rate * self.retained_gradient

class RMSPropOptimizer:

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, grad_loss_w):
        if self.cache is None:
            self.cache = np.zeros(np.shape(grad_loss_w))
        self.cache = self.beta * self.cache + (1 - self.beta) * (grad_loss_w ** 2)
        update_value = self.learning_rate * grad_loss_w / (np.sqrt(self.cache) + self.epsilon)
        return w - update_value

class AdamOptimizer:

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):
        if self.m is None:
            self.m = np.zeros(np.shape(grad_loss_w))
            self.v = np.zeros(np.shape(grad_loss_w))
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_loss_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_loss_w ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update_value = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w - update_value