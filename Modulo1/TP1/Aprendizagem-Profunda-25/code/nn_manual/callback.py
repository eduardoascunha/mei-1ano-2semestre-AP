#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####
#### Código baseado no material da UC Aprendizagem Profunda 24-25
####

import numpy as np
import copy

class EarlyStopping:
    """
    Early stopping to stop the training when a monitored metric stops improving.
    
    Arguments:
        monitor: Metric to monitor (e.g., 'loss', 'metric')
        min_delta: Minimum change to qualify as improvement
        patience: Number of epochs with no improvement after which training will stop
        verbose: Whether to print messages
        mode: 'min' (monitor should decrease) or 'max' (monitor should increase)
        restore_best_weights: Whether to restore model weights from the epoch with the best value
    """
    
    def __init__(self, 
                 monitor='loss',
                 min_delta=0.0, 
                 patience=5, 
                 verbose=True, 
                 mode='min', 
                 restore_best_weights=True):
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Initialize variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        if mode == 'min':
            self.best = np.inf
            self.monitor_op = lambda x, best: x < best - min_delta
        elif mode == 'max':
            self.best = -np.inf
            self.monitor_op = lambda x, best: x > best + min_delta
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'.")
    
    def on_epoch_end(self, epoch, history, model):
        """
        Check if training should be stopped at the end of an epoch.
        
        Returns:
            True if training should stop, False otherwise
        """
        current = history[epoch].get(self.monitor)
        
        if current is None:
            raise ValueError(f"Monitor {self.monitor} not found in history. "
                           f"Available metrics: {list(history[epoch].keys())}")
        
        # Check if current value is better than the best
        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save model weights if needed
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best {self.monitor}: {self.best} at epoch {self.best_epoch}")
                
                # Restore to best weights if needed
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print(f"Restoring model weights from epoch {self.best_epoch}")
                    self._restore_model_weights(model, self.best_weights)
                return True
        
        return False
    
    def _get_model_weights(self, model):
        """Extract weights from the model layers."""
        weights = []
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                layer_weights = {}
                if hasattr(layer, 'weights') and layer.weights is not None:
                    layer_weights['weights'] = copy.deepcopy(layer.weights)
                if hasattr(layer, 'biases') and layer.biases is not None:
                    layer_weights['biases'] = copy.deepcopy(layer.biases)
                weights.append(layer_weights)
            else:
                weights.append(None)  # For layers without weights
        return weights
    
    def _restore_model_weights(self, model, weights):
        """Restore weights to the model layers."""
        for i, layer in enumerate(model.layers):
            if weights[i] is not None and hasattr(layer, 'weights'):
                if 'weights' in weights[i]:
                    layer.weights = weights[i]['weights']
                if 'biases' in weights[i]:
                    layer.biases = weights[i]['biases']