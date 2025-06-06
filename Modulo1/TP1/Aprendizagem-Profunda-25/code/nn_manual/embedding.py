import numpy as np

def load_glove_embeddings(file_path, embedding_dim=100):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')

            if len(vector) != embedding_dim:
                raise ValueError("Len do vetor != embedding_dim.")

            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index=None, oov_strategy="random", normalize=False, embedding_dim=100, file_path=None):

    if embeddings_index == None and file_path == None:
        raise ValueError("ERRO NOS ARGUMENTOS")

    if embeddings_index == None and file_path != None:
        embeddings_index = load_glove_embeddings(file_path, embedding_dim)

    vocab_size = len(word_index) + 1  # +1 for padding if needed
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # calcular a media dos embeddings
    if oov_strategy == 'mean':
        all_embeddings = np.array(list(embeddings_index.values()))
        mean_embedding = np.mean(all_embeddings, axis=0)

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            if normalize == True:
                embedding_vector = embedding_vector / np.linalg.norm(embedding_vector) # normalização L2 

            embedding_matrix[i] = embedding_vector  # Use pre-trained vector

        else:
            if oov_strategy == "random":
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))  # Random init
            
            elif oov_strategy == "mean":
                embedding_matrix[i] = mean_embedding

            elif oov_strategy == "zero":
                embedding_matrix[i] = np.zeros((embedding_dim,))
            
            else:
                raise ValueError("Estrategia de OOV escolhida invalida.")

    return embedding_matrix

def features_to_word_index(vocab):
    return {word: i for i, word in enumerate(vocab)}

def convert_onehot_to_indices(onehot_matrix):
    return np.argmax(onehot_matrix, axis=1)
