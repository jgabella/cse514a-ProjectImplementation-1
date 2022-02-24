import numpy as np
from MSE import mse, mse_loss

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
    numits = 0
    for t in range(max_its):
        numits = t
        # num_rows = X.shape[0]
        # random_indices = np.random.choice(num_rows, size=50, replace=False)
        # batch = X[random_indices, :]
        loss, gradient = mse(w, X, Y)
        if loss > prevloss: #If my loss increased, roll back to previous weight vector, and reduce stepsize by half
            eta *= 0.5
            w = prevw
        else:
            eta *= 1.01
        norm = np.linalg.norm(gradient)
        if norm < tolerance:
            return w, numits
        if eta < min_step:
            eta = min_step
        prevw = w
        w = w - eta*gradient
        prevloss = loss
    return w, numits

def test(X, Y, w):
    loss = mse_loss(w, X, Y)
    var_exp = 1 - (loss/np.var(Y))
    return loss, var_exp