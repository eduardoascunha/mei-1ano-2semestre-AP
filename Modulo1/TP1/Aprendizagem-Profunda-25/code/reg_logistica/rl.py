import numpy as np
import matplotlib.pyplot as plt

import os
import random
def set_seed(seed: int):
    random.seed(seed) # Python
    np.random.seed(seed)  # Numpy, é o gerador utilizado pelo sklearn
    os.environ["PYTHONHASHSEED"] = str(seed)  # sistema operativo


class LogisticRegression:
    
    def __init__(self, dataset, standardize = False, regularization = False, lamda = 1,optimization_method='scipy'):
        if standardize:
            dataset.standardize()
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.Xst ))
            self.standardized = True
        else:
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.X ))
            self.standardized = False
        self.y = dataset.y
        self.theta = self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        self.data = dataset
        self.optimization_method = optimization_method

    def buildModel(self, alpha=0.01, iters=1000, tol=1e-5):

        if self.optimization_method == 'gradient_descent':
            # Usa Gradient Descent
            self.gradientDescent(alpha=alpha, iters=iters, tol=tol)
        else:
            # Método padrão (SciPy)
            if self.regularization:
                self.optim_model_reg(self.lamda)    
            else:
                self.optim_model()

    def gradientDescent(self, alpha=0.01, iters=100, tol=1e-5):
        m = self.X.shape[0]  
        n = self.X.shape[1]  
        self.theta = np.zeros(n)  
        prev_cost = float('inf')  

        for its in range(iters):
            J = self.costFunction()  
            if its % 1000 == 0: 
                print(f"Iteration {its}: Cost = {J}")  

            # Verifica a convergência
            if abs(prev_cost - J) < tol:
                print(f"Converged at iteration {its} with cost {J}")
                break
            prev_cost = J  # Atualiza o custo anterior

            # Calcula o gradiente
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)
            # Atualiza os parâmetros
            self.theta -= (alpha / m) * delta
            
    
    def optim_model(self):
        from scipy import optimize

        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, _, _, _ = optimize.fmin(lambda theta: self.costFunction(theta), initial_theta, **options)
    
    
    def optim_model_reg(self, lamda):
        from scipy import optimize

        n = self.X.shape[1]
        initial_theta = np.ones(n)        
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda), initial_theta, method='BFGS', 
                                    options={"maxiter":500, "disp":False} )
        self.theta = result.x    
  
    
    def predict(self, instance):
        p = self.probability(instance)
        if p >= 0.5: res = 1
        else: res = 0
        return res
    
    def probability(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        if self.standardized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.data.mu) / self.data.sigma
            else: x[1:] = (x[1:] - self.mu) 
        
        return sigmoid ( np.dot(self.theta, x) )


    def costFunction(self, theta = None):
        if theta is None: theta= self.theta        
        m = self.X.shape[0]
        p = sigmoid ( np.dot(self.X, theta) )
        eps = 1e-15  # Clipping to avoid log(0)
        p = np.clip(p, eps, 1 - eps)
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        res = np.sum(cost) / m
        return res
        
        
    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta= self.theta        
        m = self.X.shape[0]
        p = sigmoid ( np.dot(self.X, theta) )
        eps = 1e-15  # Clipping to avoid log(0)
        p = np.clip(p, eps, 1 - eps)
        cost  = (-self.y * np.log(p) - (1-self.y) * np.log(1-p) )
        reg = np.dot(theta[1:], theta[1:]) * lamda / (2*m)
        return (np.sum(cost) / m) + reg
        
    def predictMany(self, Xt):
        p = sigmoid ( np.dot(Xt, self.theta) )
        return np.where(p >= 0.5, 1, 0)
    
    def accuracy(self, Xt, yt):
        preds = self.predictMany(Xt)
        errors = np.abs(preds-yt)
        return 1.0 - np.sum(errors)/yt.shape[0]
    
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
        Xt = np.hstack((np.ones([dataset.nrows(), 1]), dataset.X))
        yt = dataset.y
        return self.accuracy(Xt, yt)
    
    def score_f1(self, dataset):
        Xt = np.hstack((np.ones([dataset.nrows(), 1]), dataset.X))
        yt = dataset.y
        return self.f1_score(Xt, yt)
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
  
if __name__ == '__main__':
    from data import read_csv 

    set_seed(25)

    # Carregar os dados
    dataset_train = read_csv('../../datasets/train.csv', sep=',', features=True, label=True)
    dataset_test = read_csv('../../datasets/test.csv', sep=',', features=True, label=True)
    dataset_val = read_csv('../../datasets/validation.csv', sep=',', features=True, label=True)
    dataset_stor = read_csv('../../datasets/input_prof.csv', sep=',', features=True, label=False)


    print("Done reading!")

    # Verificar o balanceamento das classes
    train_pos = np.sum(dataset_train.y)
    train_neg = len(dataset_train.y) - train_pos
    print(f"Distribuição no treino: Positivos={train_pos}, Negativos={train_neg}, Ratio={train_pos/len(dataset_train.y):.2f}")

    # Criar e treinar o modelo
    #log_model = LogisticRegression(dataset_train, standardize=True, regularization=True, lamda=0.1, optimization_method= 'gradient_descendent')
    log_model = LogisticRegression(dataset_train, standardize=True, regularization=True, lamda=0.1)
    log_model.buildModel()  


    #-----------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------

    # Testar o modelo - Accuracy
    test_accuracy = log_model.score(dataset_test)
    val_accuracy = log_model.score(dataset_val)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Testar o modelo - F1 Score
    test_f1, test_precision, test_recall = log_model.score_f1(dataset_test)
    print(f"Test F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    val_f1, val_precision, val_recall = log_model.score_f1(dataset_val)
    print(f"Validation F1 Score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    # Previsões no dataset do professor
    binary_conv = {0: "Human", 1: "AI"}
    out = log_model.predictMany(np.hstack((np.ones([dataset_stor.nrows(), 1]), dataset_stor.X)))
    out_labels = np.vectorize(binary_conv.get)(out)

    # Criar IDs para as previsões
    num_samples = len(out_labels)
    ids = [f"D1-{i+1}" for i in range(num_samples)]

    # Empilhar IDs e labels
    output_array = np.column_stack((ids, out_labels))

    # Salvar no formato desejado
    np.savetxt('dataset1_outputs1_rl.csv', output_array, delimiter='\t', fmt='%s', header="ID\tLabel", comments='')