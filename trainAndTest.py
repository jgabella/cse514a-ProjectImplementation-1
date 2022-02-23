import numpy as np
from MSE import mse

def trainSGD(X,Y, max_its, eta, tolerance=1e-03):
    """
    :param X: Complete feature matrix of dataset (n, d)
    :param Y: Complete output matrix of dataset (n, )
    :param max_its: Maximum iterations terminating condition
    :param eta: Initial stepsize
    :param thresh: Norm threshold terminating condition
    :return: w - weight vector
    """
    (n, d) = X.shape
    w = np.zeros(shape=(1, d))
    prevw = w
    prevloss, _ = mse(w, X, Y)
    min_step = 1e-14
    for t in range(max_its):
        loss, gradient = mse(w, X, Y)
        if loss > prevloss: #If my loss increased, scale back to previous weight vector, and reduce stepsize by half
            eta *= 0.5
            w = prevw
        else:
            eta *= 1.01
        norm = np.linalg.norm(gradient)
        if norm < tolerance:
            return w
        if eta < min_step:
            eta = min_step
        prevw = w
        w = w - eta*gradient
    return w

def test(Xtest, Ytest, w):
    loss, _ = mse(w, Xtest, Ytest)