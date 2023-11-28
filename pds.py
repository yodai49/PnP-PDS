import numpy as np
import operators as op
import bm3d

def psnr(img_1, img_2, data_range=1):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def test_iter(x_0, x_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, gaussian_nl, sp_nl, poisson_alpha, path_prox, max_iter, method="ours-A", ch = 3):
    # x_0　初期値
    # x_true 真の画像
    # phi, adj_phi 観測作用素とその随伴作用素
    # gamma1, gamma2 PDSのステップサイズ
    # alpha_s スパースノイズのalpha
    # alpha_n ガウシアンノイズのalpha
    # myLambda PnP-FBSのステップサイズ
    # gaussian_nl, sp_nl　ガウシアンノイズの分散とスパースノイズの重畳率
    # path_prox ガウシアンデノイザーのパス
    # max_iter アルゴリズムのイタレーション数
    # method 手法　ours:提案手法　comparison 比較手法（1はPnP-FBS, 2は制約条件版のTV, 3は和版のTV）
    #              A: 観測＋ガウシアンノイズ　B: 観測＋ガウシアンノイズ＋スパースノイズ　C:観測＋ポアソンノイズ
    #               ours-A  comparisonA-1  などのように指定する

    x_n = x_0
    y_n = np.zeros(x_0.shape) # 次元が画像と同じ双対変数
    y1_n = np.concatenate([np.zeros(x_0.shape), np.zeros(x_0.shape)], 0)
    y2_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    z_n = np.zeros(x_0.shape)
    d_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    for i in range(max_iter):
        x_prev = x_n
        s_prev = s_n

        if(method == 'ours-A'):
            # Primal-dual spilitting algorithm with denoiser (Gaussian noise)
            x_n = op.denoise(x_n - gamma1 * adj_phi(y_n), path_prox, ch)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'ours-B'):
            # Primal-dual spilitting algorithm with denoiser  (Gaussian noise + sparse noise)
            x_n = op.denoise(x_n - gamma1 * adj_phi(y_n), path_prox, ch)
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'ours-C'):
            # Primal-dual spilitting algorithm with denoiser (Poisson noise)
            x_n = op.denoise(x_n - gamma1 * adj_phi(y_n), path_prox, ch)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_0)
        elif(method == 'comparisonA-1'):
            # Forward-backward spilitting algorithm with DnCNN
            x_n = op.denoise(x_n - gamma1 * myLambda * 0.5 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_0)), path_prox, ch)
        elif(method == 'comparisonA-2'):
            # Primal-dual spilitting algorithm with HTV
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev))
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonA-3'):
            # Primal-dual splitting with HTV (additive formulation):
            x_n = x_n - gamma1 * (adj_phi(phi(x_n)-x_0) + op.D_T(y1_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, gamma2)   
        elif(method == 'comparisonA-4'):
            # BM3D-PnP-FBS (additive formulation)
            x_n = x_n - gamma1 * op.grad_x_l2(x_n, np.zeros(x_n.shape) , phi, adj_phi, x_0)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=0.01)
            x_n = np.moveaxis(x_n, -1, 0)
        elif(method == 'comparisonA-5'):
            # BM3D-PnP-PDS (constrained formulation)
            x_n = x_n - gamma1 *adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=0.01)
            x_n = np.moveaxis(x_n, -1, 0)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonB-1'):
            # DnCNN-PnP-FBS (additive formulation)
            x_n = op.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_0)), path_prox, ch)
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_0)), alpha_s, sp_nl)
        elif(method == 'comparisonB-2'):
            # HTV (constrained formulation)
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y2_n, alpha_s, sp_nl)
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonB-3'):
            # BM3D-PnP-PDS (Gaussian noise + sparse noise)
            x_n = x_n - gamma1 * adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=0.01)
            x_n = np.moveaxis(x_n, -1, 0)
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_0)
        elif(method == 'comparisonC-1'):
            # BM3D-PnP-PDS (Poisson noise) 未チェック
            x_n = bm3d.bm3d(x_n - gamma1 * adj_phi(y_n), path_prox, ch)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_0)
        elif(method == 'comparisonC-2'):
            # BM3D-PnP-ADMM (Poisson noise)
            x_n = step1ofADMMforPoisson (d_n, z_n, x_0, phi, adj_phi, poisson_alpha)
            z_n = op.denoise(x_n + d_n, path_prox, ch)
            d_n = d_n + x_n - z_n
        else:
            print("Unknown method:", method)
            return x_0, c

        c[i] = np.linalg.norm((x_n - x_prev).flatten()) / np.linalg.norm(x_prev.flatten())
#        c[i] = np.linalg.norm(phi(x_n) - x_0)
        psnr_data[i] = psnr(x_n, x_true)
        if(i % 10 == 0):
            print('Method:' , method, '  iter: ', i, ' / ', max_iter, ' PSNR: ', psnr_data[i])

    return x_n, s_n+0.5, c, psnr_data

def step1ofADMMforPoisson (u, v, y, phi, adj_phi, poisson_alpha):
    # ADMMのステップ1を計算する関数 ポアソンノイズ用
    MAX_ITER = 100
    gamma = 10
    lambydaInStep1 = 0.025
    x_n = np.ones(u.shape)
    for i in range(0, MAX_ITER):
        grad = -adj_phi(y / (poisson_alpha*phi(x_n)))/poisson_alpha + adj_phi(np.ones(x_n.shape))/poisson_alpha + lambydaInStep1 * (x_n - v + u)
        x_n = x_n - gamma * grad
        #val = -y.flatten() @ np.log(phi(x_n).flatten()*poisson_alpha) + np.ones(x_n.size) @ phi(x_n).flatten()*poisson_alpha + lambydaInStep1 / 2 * np.linalg.norm(x_n - v + u)**2
        #print(val)
    #print("-------")
    return x_n