import numpy as np

from layers import DenseLayer, DropoutLayer
from losses import LossFunction, MeanSquaredError
from optimizer import Optimizer
from losses import BinaryCrossEntropy
from activation import SigmoidActivation, ReLUActivation
from metrics import mse, accuracy, precision_recall_f1, recall
from data import read_csv
from optimizer import Optimizer,AdamOptimizer, RMSPropOptimizer
from neuralnet import NeuralNetwork
from callback import EarlyStopping
import numpy as np
import os
import random
import pickle


def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo

if __name__ == '__main__':

    set_seed(25)
    # training data
    dataset_train = read_csv('datasets/train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('datasets/test.csv', sep=',', features=True, label=True)

    print("Done reading!")
    # network
    
    early_stopping = EarlyStopping(
        monitor='metric',  # Monitor validation metric or loss
        min_delta=0.0001,       # Minimum change to qualify as improvement
        patience=10,           # Stop after 10 epochs without improvement
        verbose=True,          # Print messages
        mode='max',            # We want metric to increase (for accuracy)
        restore_best_weights=True  # Restore to best weights when stopped
    )

    net = NeuralNetwork(epochs=200, batch_size=64, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, optimizer=AdamOptimizer(learning_rate=0.1),callbacks=[early_stopping])

    n_features = dataset_train.X.shape[1]
    net.add(DenseLayer(150, (n_features,),init_weights="xavier"))
    net.add(ReLUActivation())

    net.add(DropoutLayer(dropout_rate=0.6))

    net.add(DenseLayer(75,init_weights="he"))
    net.add(ReLUActivation())
    
    net.add(DropoutLayer(dropout_rate=0.5))

    net.add(DenseLayer(1,init_weights="he"))
    net.add(SigmoidActivation())
    #net.add(ReLUActivation())
    #net.add(DropoutLayer(droupout_rate=0.1))

    # train
    net.fit(dataset_train)

    # test
    out = net.predict(dataset_test,binary=True)
    print(f"Test: {net.score(dataset_test, out)}")
    # write predictions on file
    np.savetxt('predictions.csv', out, delimiter=',')

    # Load validation dataset
    dataset_val = read_csv('datasets/validation.csv', sep=',', features=True, label=True)

    # Get predictions
    val = net.predict(dataset_val, binary=True)

    # Get real labels
    real = dataset_val.get_y()

    # Save results with header
    #output_data = np.column_stack((real, val))
    #np.savetxt('validations_predictions_manual_nn.csv', output_data, delimiter=',', header="real,predicted", comments='')

    # Print validation accuracy
    print(f"Validation accuracy: {net.score(dataset_val, val)}")


    # Temporarily remove callbacks before saving
    callbacks_backup = net.callbacks  # Store callbacks
    net.callbacks = None  # Remove them for pickling

    # Save the trained model
    with open('trained_model1.pkl', 'wb') as f:
        pickle.dump(net, f)



    """
    
  
    # Predict com dados do stor!
    dataset_stor = read_csv('input_prof.csv', sep=',', features=True, label=False)

    binary_conv = {0: "Human", 1: "AI"}

    # Get predictions
    out = net.predict(dataset_stor, binary=True)

    # Convert numerical predictions to labels
    out_labels = np.vectorize(binary_conv.get)(out)

    # Create row IDs (D1-1, D1-2, ..., D1-N)
    num_samples = len(out_labels)
    ids = [f"D1-{i+1}" for i in range(num_samples)]

    # Stack IDs and labels into a single 2D array
    output_array = np.column_stack((ids, out_labels))

    # Save to file with header
    np.savetxt('dataset1_outputs1_grupo.csv', output_array, delimiter='\t', fmt='%s', header="ID\tLabel", comments='')
    compare_csv_files("dataset1_outputs.csv", "dataset1_outputs1_grupo.csv")
  """
