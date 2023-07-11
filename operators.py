import numpy as np

def get_operators(shape, gamma1, gamma2):
    def grad_f(x):
        return x
    def prox_g(x):
        return np.sign(x) * np.fmax(0, np.abs(x) - gamma1)
    def prox_h(x):
        return x
    def phi(x):
        return x
    size = np.prod(shape)
    L = np.eye(size, size)
    return phi, grad_f, prox_g, prox_h, L