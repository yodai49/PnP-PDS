import numpy as np
def test_pds(x_0, grad_f, prox_g, prox_h, L, gamma1, gamma2, max_iter):
    # 実際にPDSを解く部分
    # L is omitted to improve performance
    x_n = x_0
    y_n = x_0
    for i in range(max_iter):
#        x_n1 = prox_g(x_n - gamma1 * (grad_f(x_n) + y_n))
#        y_n1 = prox_h(y_n - gamma2 * (2 * x_n1 - x_n))
        #x_n1 = prox_g(x_n - gamma1 * grad_f(x_n))
        print(grad_f(x_n))
        x_n = x_n - gamma1 * grad_f(x_n)
#        y_n = y_n1
        print('iter: ', i)
    return x_n