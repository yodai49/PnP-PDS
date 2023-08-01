import numpy as np
def psnr(img_1, img_2, data_range=1):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def test_iter(x_0, x_true, grad_f, prox_g, prox_h_dual, grad_f_s, prox_g_s, phi, adj_phi, gamma1, gamma2, max_iter, method="PDS_S"):
    # test FBS or PDS algorithm
    x_n = x_0
    y_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    for i in range(max_iter):
        x_prev = x_n
        s_prev = s_n

        if(method == "FBS"):
            # Forward-Backward spilitting algorithm
            x_n = prox_g(x_n - gamma1 * grad_f(x_n))
        elif(method == "PDS"):
            # Primal-dual spilitting algorithm
            x_n = prox_g(x_n - gamma1 * (grad_f(x_n, np.zeros(x_n.shape)) + adj_phi(y_n)))
            y_n = prox_h_dual(y_n + gamma2 * phi(2 * x_n - x_prev))
        elif(method == "PDS_S"):
            # Primal-dual spilitting algorithm with sparse noise
            x_n = prox_g(x_n - gamma1 * (grad_f(x_n, s_n) + adj_phi(y_n)))
            s_n = prox_g_s(s_n - gamma1 * (grad_f_s(x_n, s_n) + y_n))
            y_n = prox_h_dual(y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev))
        else:
            print("Unknown method:", method)
            return x_0, c

        c[i] = np.linalg.norm((x_n - x_prev).flatten()) / np.linalg.norm(x_0.flatten())
        psnr_data[i] = psnr(x_n, x_true)
        print('Method:' , method, '  iter: ', i, ' / ', max_iter, ' PSNR: ', psnr_data[i])
    
    return x_n, s_n+0.5, c, psnr_data