import numpy as np
import operators as op

def psnr(img_1, img_2, data_range=1):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def test_iter(x_0, x_true, phi, adj_phi, gamma1, gamma2, alpha_eta, alpha_epsilon, gaussian_nl, sp_nl, path_prox, max_iter, method="PDS_S"):
    # test algorithm (FBS / PDS)
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
            x_n = op.denoise(x_n - gamma1 * op.grad_x_l2(x_n, np.zeros(x_0.shape), phi, adj_phi, x_0), path_prox)
        elif(method == "PDS"):
            # Primal-dual spilitting algorithm
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_0) + adj_phi(y_n)), path_prox)
            y_n = op.proj_l2_ball_dual(y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev), gamma2, alpha_epsilon, gaussian_nl, x_0)
        elif(method == 'PDS_with_sparse_noise'):
            # Primal-dual spilitting algorithm with HTV
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_0) + adj_phi(y_n)), path_prox)
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_0) + y_n), alpha_eta, sp_nl)
            y_n = op.proj_l2_ball_dual(y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev), gamma2, alpha_epsilon, gaussian_nl, x_0)
        elif(method == 'PnP-PDS_with_sparse_noise'):
            # Primal-dual spilitting algorithm with denoiser
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_0) + adj_phi(y_n)), path_prox)
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_0) + y_n), alpha_eta, sp_nl)
            y_n = op.proj_l2_ball_dual(y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev), gamma2, alpha_epsilon, gaussian_nl, x_0)
        else:
            print("Unknown method:", method)
            return x_0, c

        c[i] = np.linalg.norm((x_n - x_prev).flatten()) / np.linalg.norm(x_0.flatten())
        psnr_data[i] = psnr(x_n, x_true)
        print('Method:' , method, '  iter: ', i, ' / ', max_iter, ' PSNR: ', psnr_data[i])
    
    return x_n, s_n+0.5, c, psnr_data