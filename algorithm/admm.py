import numpy as np
import operators as op

def step1ofADMMforPoisson (u, v, y, phi, adj_phi, poisson_alpha, myLambda, m, gammaInADMMStep1):
    # ADMMのステップ1を計算する関数 ポアソンノイズ用
    # lambdaはmyLambdaで指定
    MAX_ITER = m
    gamma = gammaInADMMStep1
    lambydaInStep1 = myLambda
    x_n = np.ones(u.shape)
    for i in range(0, MAX_ITER):
        grad = -adj_phi(y / (poisson_alpha*phi(x_n)))/poisson_alpha + adj_phi(np.ones(x_n.shape))/poisson_alpha + lambydaInStep1 * (x_n - v + u)
        x_n = x_n - gamma * grad
#        val = -y.flatten() @ np.log(phi(x_n).flatten()*poisson_alpha) + np.ones(x_n.size) @ phi(x_n).flatten()*poisson_alpha + lambydaInStep1 / 2 * np.linalg.norm(x_n - v + u)**2
#        print(val)
    return x_n

def step2ofADMM_REDforPoisson (x_n, u_prev, v_prev, beta, myLambda, path_prox, ch, m):
    # ADMMのステップ2(RED)を計算する関数 ポアソンノイズ用
    # betaはstep1ofADMMforPoisson のlambdaInStep1と同じ
    MAX_ITER = m
    lambydaInStep2 = myLambda
    z_str = x_n + u_prev
    z_n = v_prev
    for i in range(0, MAX_ITER):
        z_n = op.denoise(z_n, path_prox, ch)
        z_n = 1 / (beta + lambydaInStep2) * (lambydaInStep2 * z_n + beta * z_str)
    return z_n

def step1ofADMMforSparseX(s_n, z_n, y_n, phi, adj_phi, path_prox, ch, gamma1, m):
    MAX_ITER = m
    x_n = np.ones(s_n.shape)
    for i in range(0, MAX_ITER):
        x_n = x_n - 1 / gamma1 * adj_phi(phi(x_n) + s_n - z_n + y_n)
        x_n = op.denoise(x_n, path_prox, ch)
    return x_n

def step1ofADMMforSparseS(x_n, z_n, y_n, phi, alpha_s, sp_nl, gamma1, m):
    MAX_ITER = m
    s_n = np.ones(x_n.shape)
    for i in range(0, MAX_ITER):
        s_n = s_n - 1 / gamma1 * (phi(x_n) + s_n - z_n + y_n)
        s_n = op.proj_l1_ball(s_n, alpha_s, sp_nl)
    return s_n