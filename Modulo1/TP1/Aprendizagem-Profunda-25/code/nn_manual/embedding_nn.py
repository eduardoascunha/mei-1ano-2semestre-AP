import os
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo


if __name__ == '__main__':
    from neuralnet import NeuralNetwork
    from layers import EmbeddingLayer, DenseLayer, DropoutLayer
    from losses import BinaryCrossEntropy
    from activation import SigmoidActivation, ReLUActivation
    from metrics import mse, accuracy
    from data import read_csv,Data
    from optimizer import Optimizer,AdamOptimizer
    from embedding import load_glove_embeddings, create_embedding_matrix, features_to_word_index, convert_onehot_to_indices

    set_seed(25)
    # training data
    dataset_train = read_csv('../train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('../test.csv', sep=',', features=True, label=True)
    dataset_val = read_csv('../validation.csv', sep=',', features=True, label=True)

    print("Done reading!")

    # Create word index directly from features
    word_index = features_to_word_index(dataset_train.features)
    vocab_size = len(word_index) + 1  # +1 for padding or unknown words

    # Load GloVe embeddings
    glove_path = 'glove.6B.100d.txt'
    embeddings_index = load_glove_embeddings(glove_path, embedding_dim=100)

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, oov_strategy="mean", normalize=True, embedding_dim=100)
    print(f"Created embedding matrix with shape: {embedding_matrix.shape}")

    # Create neural network
    net = NeuralNetwork(epochs=10, batch_size=16, verbose=True,
                        loss=BinaryCrossEntropy, metric=accuracy, learning_rate=0.1)

    # Add the modified embedding layer that handles one-hot vectors directly
    net.add(EmbeddingLayer(vocab_size, 100, embedding_matrix=embedding_matrix, trainable=False))

    # Add remaining layers
    net.add(DenseLayer(20, (100,), init_weights="xavier"))
    net.add(ReLUActivation())

    net.add(DenseLayer(1, init_weights="he"))
    net.add(SigmoidActivation())

    # Train the network with original one-hot encoded data
    net.fit(dataset_train)

    out = net.predict(dataset_test,binary=True)
    print(f"Test: {net.score(dataset_test, out)}")
    # write predictions on file
    np.savetxt('out_test_predictions.csv', out, delimiter=',')

    # Validate the network
    val = net.predict(dataset_val,binary=True)
    print(f"Validation accuracy: {net.score(dataset_val, val)}")
    np.savetxt('out_validations_predictions.csv', val, delimiter=',')
