import numpy as np 
import pandas as pd  


class Data:
    def __init__(self, X, y = None, features = None, label= None):

        if X is None:
            raise ValueError("X cannot be None")
       
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        
        if y is not None and label is None:
            label = "y"
        
        self.X = X
        self.y = y
        self.features = features
        self.label = label


    def shape(self):
        return self.X.shape

    def has_label(self):
        return self.y is not None

    def get_classes(self):
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self):
        return np.nanmean(self.X, axis=0)

    def get_variance(self):
        return np.nanvar(self.X, axis=0)

    def get_median(self):
        return np.nanmedian(self.X, axis=0)

    def get_min(self):
        return np.nanmin(self.X, axis=0)

    def get_max(self):
        return np.nanmax(self.X, axis=0)

    def summary(self):
        data = {
            "mean": self.get_mean(),        # Média
            "median": self.get_median(),    # Mediana
            "min": self.get_min(),          # Mínimo
            "max": self.get_max(),          # Máximo
            "var": self.get_variance()      # Variância
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    def head(self, n=5):
        """
        Returns a new Data object containing the first n rows.
        """
        X_head = self.X[:n]
        y_head = self.y[:n] if self.has_label() else None
        return Data(X=X_head, y=y_head, features=self.features, label=self.label)


    def standardize(self):
        self.mu = np.mean(self.X, axis = 0)
        self.Xst = self.X - self.mu
        self.sigma = np.std(self.X, axis = 0)
        self.Xst = self.Xst / self.sigma

    def nrows(self):
        return self.X.shape[0]
    
    def ncols(self):
        return self.X.shape[1]
    
    
def read_csv(filename,
             sep = ',',         # Separador usado no CSV
             features = False,
             label = False):
 
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]  
        label = data.columns[-1]  
        X = data.iloc[:, :-1].to_numpy()  
        y = data.iloc[:, -1].to_numpy() 

    elif features and not label:
        features = data.columns 
        X = data.to_numpy()
        y = None 

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()  
        y = data.iloc[:, -1].to_numpy()  
        features = None  
        label = data.columns[-1]  

    else:
        X = data.to_numpy()  
        y = None  
        features = None  
        label = None 

    return Data(X=X, y=y, features=features, label=label)
