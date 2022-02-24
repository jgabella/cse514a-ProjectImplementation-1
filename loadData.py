import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


'''
    INPUT:
        path: String to the xls datasheet
        preprocess: boolean to determine if you'd like to preprocess the data with zero-means normalization
        univariate: boolean to use univariate vs multivariate model.
        column_select: Int column to use for the univariate model if working with a multivariate dataset
    
    Output:
        X - Feature matrix in the shape of (n,d) where n is the number of data points and d is the number of features
        Y - Output matrix in the shape of (n,)

'''
def loadData(path, preprocess=False, univariate=False, column_select=0):
    df = pd.read_excel(path)
    if univariate:
        X = df.iloc[:, column_select]  # Last column in data frame are labeled output values
        Y = df.iloc[:, -1]
        X = X.to_numpy().reshape((len(X), 1))  # shape nxd
        Y = Y.to_numpy()
    else:
        X = df.iloc[:, :-1] #Last column in data frame are labeled output values
        Y = df.iloc[:, -1]
        X = X.to_numpy()  # shape nxd
        Y = Y.to_numpy()

    if preprocess:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    bias_col = np.ones(shape=(1, len(X)))
    X = np.insert(X, 0, bias_col, axis=1)
    Xtr = X[:900, :]
    Xtest = X[901:,:]
    Ytr = Y[:900]
    Ytest = Y[901:]
    # Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size=0.2)
    return Xtr, Xtest, Ytr, Ytest

# X, Y = loadData('Concrete_Data.xls', preprocess=True)