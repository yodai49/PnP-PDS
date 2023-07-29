import numpy as np
def psnr(img_1, img_2, data_range=1):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def test_iter(x_0, x_true, grad_f, prox_g, prox_h_dual, L,L_T, gamma1, gamma2, max_iter, method="PDS"):
    # L can be omitted to improve performance
    x_n = x_0
    y_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    for i in range(max_iter):
        x_prev = x_n

        if(method == "FB"):
            # Forward-Backward algorithm
            x_n = prox_g(x_n - gamma1 * grad_f(x_n))
        elif(method == "PDS"):
            # Primal-dual spilitting algorithm
            x_n = prox_g(x_n - gamma1 * (grad_f(x_n) + L_T(y_n)))
            y_n = prox_h_dual(y_n + gamma2 * L(2 * x_n - x_prev))
        else:
            print("Unknown method:", method)
            return x_0, c

        c[i] = np.linalg.norm((x_n - x_prev).flatten()) / np.linalg.norm(x_0.flatten())
        psnr_data[i] = psnr(x_n, x_true)
        if(i % 50 == 0):
            print('Method:' , method, '  iter: ', i, ' / ', max_iter)
    
    return x_n, c, psnr_data