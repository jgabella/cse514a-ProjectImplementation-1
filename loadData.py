import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

# for col in range(8):
#     Xtr1, Xt1, Ytr, Yt = loadData('Concrete_Data.xls', preprocess=True, univariate=True, column_select=col)
#     Xtr1 = np.delete(Xtr1, 0, axis=1).T[0]
#     Xt1 = np.delete(Xt1, 0, axis=1).T[0]
#     X1 = np.concatenate([Xtr1, Xt1])
#     Xtr, Xt, Ytr, Yt = loadData('Concrete_Data.xls', preprocess=False, univariate=True, column_select=col)
#     Xtr = np.delete(Xtr, 0, axis=1).T[0]
#     Xt = np.delete(Xt, 0, axis=1).T[0]
#     X = np.concatenate([Xtr, Xt])
#     plt.hist(X1, color='r')
#     plt.hist(X, color='b')
#     plt.show()
