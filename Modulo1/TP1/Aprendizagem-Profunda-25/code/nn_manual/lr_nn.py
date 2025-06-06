import numpy as np
import matplotlib.pyplot as plt
import os
import random

def set_seed(seed: int):
    random.seed(seed)  # Python
    np.random.seed(seed)  # Numpy
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo

class NeuralNetworkLogisticRegression:
    
    def __init__(self, dataset, standardize=False, regularization=False, lamda=1, learning_rate=0.01, epochs=1000):
        self.X = dataset.X
        self.y = dataset.y
        self.standardized = standardize
        self.regularization = regularization
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Padronização dos dados, se necessário
        if standardize:
            self.mu = np.mean(self.X, axis=0)
            self.sigma = np.std(self.X, axis=0)
            self.X = (self.X - self.mu) / self.sigma

        # Adicionar coluna de bias (intercepto)
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))
        self.theta = np.zeros(self.X.shape[1])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def costFunction(self, theta=None):
        if theta is None:
            theta = self.theta
        m = self.X.shape[0]
        p = self.sigmoid(np.dot(self.X, theta))
        cost = (-self.y * np.log(p) - (1 - self.y) * np.log(1 - p))
        res = np.sum(cost) / m
        return res

    def costFunctionReg(self, theta=None, lamda=1):
        if theta is None:
            theta = self.theta
        m = self.X.shape[0]
        p = self.sigmoid(np.dot(self.X, theta))
        cost = (-self.y * np.log(p) - (1 - self.y) * np.log(1 - p))
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2 * m)
        return (np.sum(cost) / m) + reg

    def gradientDescent(self):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        for _ in range(self.epochs):
            p = self.sigmoid(np.dot(self.X, self.theta))
            delta = self.X.T.dot(p - self.y)
            if self.regularization:
                reg_term = (self.lamda / m) * self.theta
                reg_term[0] = 0  # Não regularizar o termo de bias
                delta += reg_term
            self.theta -= (self.learning_rate / m) * delta

    def predict(self, instance):
        p = self.probability(instance)
        return 1 if p >= 0.5 else 0

    def probability(self, instance):
        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1] - 1])
        if self.standardized:
            x[1:] = (x[1:] - self.mu) / self.sigma
        return self.sigmoid(np.dot(self.theta, x))

    def predictMany(self, Xt):
        if self.standardized:
            Xt = (Xt - self.mu) / self.sigma
        Xt = np.hstack((np.ones([Xt.shape[0], 1]), Xt))
        p = self.sigmoid(np.dot(Xt, self.theta))
        return np.where(p >= 0.5, 1, 0)

    def accuracy(self, Xt, yt):
        preds = self.predictMany(Xt)
        errors = np.abs(preds - yt)
        return 1.0 - np.sum(errors) / yt.shape[0]

    def f1_score(self, Xt, yt):
        preds = self.predictMany(Xt)
        
        tp = np.sum((preds == 1) & (yt == 1))
        fp = np.sum((preds == 1) & (yt == 0))
        fn = np.sum((preds == 0) & (yt == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall

    def score(self, dataset):
        Xt = dataset.X
        yt = dataset.y
        return self.accuracy(Xt, yt)

    def score_f1(self, dataset):
        Xt = dataset.X
        yt = dataset.y
        return self.f1_score(Xt, yt)

if __name__ == '__main__':
    from data import read_csv

    set_seed(25)

    # Carregar os dados
    dataset_train = read_csv('train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('test.csv', sep=',', features=True, label=True)
    dataset_val = read_csv('validation.csv', sep=',', features=True, label=True)
    dataset_stor = read_csv('../reg_logistica/input_prof.csv', sep=',', features=True, label=False)

    print("Done reading!")

    # Verificar o balanceamento das classes
    train_pos = np.sum(dataset_train.y)
    train_neg = len(dataset_train.y) - train_pos
    print(f"Distribuição no treino: Positivos={train_pos}, Negativos={train_neg}, Ratio={train_pos/len(dataset_train.y):.2f}")

    # Criar e treinar o modelo
    nn_log_model = NeuralNetworkLogisticRegression(dataset_train, standardize=True, regularization=True, lamda=0.1, learning_rate=0.01, epochs=1000)
    nn_log_model.gradientDescent()

    # Testar o modelo - Accuracy
    test_accuracy = nn_log_model.score(dataset_test)
    val_accuracy = nn_log_model.score(dataset_val)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Testar o modelo - F1 Score
    test_f1, test_precision, test_recall = nn_log_model.score_f1(dataset_test)
    print(f"Test F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    val_f1, val_precision, val_recall = nn_log_model.score_f1(dataset_val)
    print(f"Validation F1 Score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    # Previsões no dataset do professor
    binary_conv = {0: "Human", 1: "AI"}
    out = nn_log_model.predictMany(dataset_stor.X)
    out_labels = np.vectorize(binary_conv.get)(out)

    # Criar IDs para as previsões
    num_samples = len(out_labels)
    ids = [f"D1-{i+1}" for i in range(num_samples)]

    # Empilhar IDs e labels
    output_array = np.column_stack((ids, out_labels))

    # Salvar no formato desejado
    np.savetxt('dataset1_outputs1_nn.csv', output_array, delimiter='\t', fmt='%s', header="ID\tLabel", comments='')