def test_pds(x_0, grad_f, prox_g, prox_h, L, max_iter):
    # 実際にPDSを解く部分
    print(grad_f(prox_g(prox_h(0))))