def get_operators():
    def grad_f(x):
        return x+1
    def prox_g(x):
        return 2*x
    def prox_h(x):
        return x+3
    return grad_f, prox_g, prox_h