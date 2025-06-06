import numpy as np
from layers import DenseLayer, DropoutLayer
from losses import BinaryCrossEntropy
from activation import SigmoidActivation, ReLUActivation
from metrics import accuracy
from data import read_csv
from optimizer import AdamOptimizer, RMSPropOptimizer
from neuralnet import NeuralNetwork
from callback import EarlyStopping
import os
import random

def set_seed(seed: int):
    random.seed(seed)  
    np.random.seed(seed)  
    os.environ["PYTHONHASHSEED"] = str(seed)  

def test_configurations(configurations, dataset_train, dataset_test, dataset_val):
    results = []
    
    for config in configurations:
        set_seed(25)  
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='metric',  # Monitor validation metric or loss
            min_delta=0.01,  # Minimum change to qualify as improvement
            patience=2,  # Stop after 20 epochs without improvement
            verbose=True,  # Print messages
            mode='max',  # We want metric to increase (for accuracy)
            restore_best_weights=True  # Restore to best weights when stopped
        )
        
        # Create the neural network with the current configuration
        net = NeuralNetwork(
            epochs=50,  # Aumentar o número de épocas
            batch_size=config['batch_size'],
            verbose=True,
            loss=BinaryCrossEntropy,
            metric=accuracy,
            optimizer=config['optimizer'],
            callbacks=[early_stopping]
        )
        
        n_features = dataset_train.X.shape[1]
        
        # Add layers to the network
        net.add(DenseLayer(config['hidden_units'][0], (n_features,), init_weights=config['weight_init'][0], l2_lambda=0.01))  
        net.add(ReLUActivation())
        net.add(DropoutLayer(dropout_rate=config['dropout_rates'][0]))
        
        net.add(DenseLayer(config['hidden_units'][1], init_weights=config['weight_init'][1], l2_lambda=0.01))  
        net.add(ReLUActivation())
        net.add(DropoutLayer(dropout_rate=config['dropout_rates'][1]))
        
        net.add(DenseLayer(config['hidden_units'][2], init_weights=config['weight_init'][2], l2_lambda=0.01))  
        net.add(ReLUActivation())
        
        net.add(DenseLayer(1, init_weights="he", l2_lambda=0.01))
        net.add(SigmoidActivation())
        
        # Train the network
        net.fit(dataset_train)
        
        # Test the network
        out = net.predict(dataset_test, binary=True)
        test_score = net.score(dataset_test, out)
        
        # Validate the network
        val = net.predict(dataset_val, binary=True)
        val_score = net.score(dataset_val, val)
        
        # Save the results
        results.append({
            'config': config,
            'test_score': test_score,
            'val_score': val_score
        })
        
        print(f"Configuration: {config}")
        print(f"Test Score: {test_score}")
        print(f"Validation Score: {val_score}")
        print("-" * 40)
    
    return results

if __name__ == '__main__':
    set_seed(25)
    
    # Load datasets
    dataset_train = read_csv('train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('test.csv', sep=',', features=True, label=True)
    dataset_val = read_csv('validation.csv', sep=',', features=True, label=True)
    
    print("Done reading!")
    

    configurations = [
        # Configuração 21: Adam com learning rate baixo, dropout alto e inicialização He/Xavier
        {
            'batch_size': 64,
            'optimizer': AdamOptimizer(learning_rate=0.0005),  # Ajustar learning rate
            'hidden_units': [256, 128, 64],  # Aumentar o número de neurônios
            'dropout_rates': [0.5, 0.4],  # Aumentar dropout
            'weight_init': ['he', 'xavier', 'he']  # He na primeira e última, Xavier na intermediária
        },

        # Configuração 22: RMSProp com learning rate médio, dropout alto e inicialização Xavier
        {
            'batch_size': 32,
            'optimizer': RMSPropOptimizer(learning_rate=0.005, beta=0.99),  # Ajustar learning rate
            'hidden_units': [200, 100, 50],  # Aumentar o número de neurônios
            'dropout_rates': [0.6, 0.5],  # Aumentar dropout
            'weight_init': ['xavier', 'xavier', 'xavier']  # Xavier em todas as camadas
        },

        # Configuração 23: Adam com learning rate médio, dropout baixo e inicialização He
        {
            'batch_size': 128,
            'optimizer': AdamOptimizer(learning_rate=0.001),  # Ajustar learning rate
            'hidden_units': [300, 150, 75],  # Aumentar o número de neurônios
            'dropout_rates': [0.3, 0.2],  # Manter dropout moderado
            'weight_init': ['he', 'he', 'he']  # He em todas as camadas
        },

        # Configuração 24: RMSProp com learning rate baixo, dropout moderado e inicialização He/Xavier
        {
            'batch_size': 16,
            'optimizer': RMSPropOptimizer(learning_rate=0.0005, beta=0.9),  # Ajustar learning rate
            'hidden_units': [150, 75, 30],  # Aumentar o número de neurônios
            'dropout_rates': [0.4, 0.3],  # Manter dropout moderado
            'weight_init': ['he', 'xavier', 'he']  # He na primeira e última, Xavier na intermediária
        },

        # Configuração 25: Adam com learning rate alto, dropout moderado e inicialização Xavier
        {
            'batch_size': 64,
            'optimizer': AdamOptimizer(learning_rate=0.01),  # Ajustar learning rate
            'hidden_units': [200, 100, 50],  # Aumentar o número de neurônios
            'dropout_rates': [0.4, 0.3],  # Aumentar dropout
            'weight_init': ['xavier', 'xavier', 'xavier']  # Xavier em todas as camadas
        }
    ]
    
    # Test configurations
    results = test_configurations(configurations, dataset_train, dataset_test, dataset_val)
    
    # Print all results
    for result in results:
        print(f"Configuration: {result['config']}")
        print(f"Test Score: {result['test_score']}")
        print(f"Validation Score: {result['val_score']}")
        print("-" * 40)