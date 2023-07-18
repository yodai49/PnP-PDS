import numpy as np
def test_pds(x_0, grad_f, prox_g, prox_h_dual, L, gamma1, gamma2, max_iter):
    # L is omitted to improve performance
    x_n = x_0
    y_n = x_0
    for i in range(max_iter):
        # Forward-Backward algorithm
        #x_n = prox_g(x_n - gamma1 * grad_f(x_n))

        # Primal-dual spilitting algorithm
        x_prev = x_n
        x_n = prox_g(x_n - gamma1 * (grad_f(x_n) + y_n))
        y_n = prox_h_dual(y_n + gamma2 * (2 * x_n - x_prev))
        print('iter: ', i)
    return x_n