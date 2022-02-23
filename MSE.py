import numpy as np

def mse(w, X, y):

    """
    :param w: Weight vector (1, d)
    :param X: Feature matrix (n, d)
    :param y: Output matrix (n, )
    :return: loss: Calculated error using mean-squared error using the w input vector
    :return: gradient: The gradient vector to update w with (1, d)
    """

    (n, d) = X.shape
    gradient = np.zeros(shape=w.shape)
    loss = 0
    for i in range(n):
        loss += (y[i] - np.dot(w, X[i]))**2
        grad_add = 2*(y[i]-np.dot(w, X[i]))*X[i]
        gradient = np.sum(gradient, grad_add)
    loss = loss/n
    gradient *= (-2/n)
    return loss, gradient

def mse_loss(w, X, y):
    (n, d) = X.shape
    loss = 0
    for i in range(n):
        loss += (y[i] - np.dot(w, X[i])) ** 2
    loss = loss / n
    return loss