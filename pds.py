import numpy as np
import operators as op

def psnr(img_1, img_2, data_range=1):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def test_iter(x_0, x_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, gaussian_nl, sp_nl, path_prox, max_iter, method="ours-A"):
    # test algorithm
    x_n = x_0
    y_n = np.zeros(x_0.shape)
    y1_n = np.concatenate([np.zeros(x_0.shape), np.zeros(x_0.shape)], 0)
    y2_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    for i in range(max_iter):
        x_prev = x_n
        s_prev = s_n

        if(method == 'ours-A' or method == 'ours-C'):
            # Primal-dual spilitting algorithm with denoiser
            x_n = op.denoise(x_n - gamma1 * adj_phi(y_n), path_prox)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonA-1' or method == 'comparisonC-1'):
            # Forward-backward spilitting algorithm with denoiser
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_0)), path_prox)
        elif(method == 'comparisonA-2' or method == 'comparisonC-2'):
            # Primal-dual spilitting algorithm with HTV
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev))
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)        
        elif(method == 'ours-B'):
            # Primal-dual spilitting algorithm with denoiser
            x_n = op.denoise(x_n - gamma1 * adj_phi(y_n), path_prox)
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonB-1'):
            # Forward-backward spilitting algorithm with denoiser
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_0)), path_prox)
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_0)), alpha_s, sp_nl)
        elif(method == 'comparisonB-2'):
            # Primal-dual spilitting algorithm with HTV
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y2_n, alpha_s, sp_nl)
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        else:
            print("Unknown method:", method)
            return x_0, c

        c[i] = np.linalg.norm((x_n - x_prev).flatten()) / np.linalg.norm(x_0.flatten())
        psnr_data[i] = psnr(x_n, x_true)
        print('Method:' , method, '  iter: ', i, ' / ', max_iter, ' PSNR: ', psnr_data[i])

    return x_n, s_n+0.5, c, psnr_data