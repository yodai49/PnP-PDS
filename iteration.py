import numpy as np
import operators as op
import bm3d, time, torch

from models.denoiser import Denoiser as Denoiser_J
from models.network_dncnn import DnCNN as Denoiser_KAIR
from utils.utils_eval import eval_psnr, eval_ssim
from algorithm.admm import *

def test_iter(x_0, x_obsrv, x_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, gaussian_nl, sp_nl, poisson_alpha, path_prox, max_iter, method="A-Proposed", ch = 3, r=1):
    # x_0　     初期値
    # x_obsrv   観測画像
    # x_true    真の画像
    # phi, adj_phi 観測作用素とその随伴作用素
    # gamma1, gamma2 PDSのステップサイズ
    # alpha_s   スパースノイズのalpha
    # alpha_n   ガウシアンノイズのalpha
    # myLambda  PnP-FBSのステップサイズ
    # gaussian_nl, sp_nl　ガウシアンノイズの分散とスパースノイズの重畳率
    # path_prox ガウシアンデノイザーのパス
    # max_iter アルゴリズムのイタレーション数
    # method 手法
    x_n = x_0
    y_n = np.zeros(x_0.shape) # 次元が画像と同じ双対変数
    y1_n = np.concatenate([np.zeros(x_0.shape), np.zeros(x_0.shape)], 0)
    y2_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    z_n = np.zeros(x_0.shape)
    d_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    ssim_data = np.zeros(max_iter)

    if (method.find('unstable') != -1):
        if (ch == 3):
            nb = 20
        elif(ch == 1):
            nb = 17
        denoiser_KAIR = Denoiser_KAIR(in_nc=ch, out_nc=ch, nc=64, nb=nb, act_mode='R', model_path = path_prox)
    elif(method.find('Proposed') != -1 or method.find('DnCNN') != -1):
        denoiser_J = Denoiser_J(file_name=path_prox, ch = ch)

    start_time = time.process_time()
    for i in range(max_iter):
        x_prev = x_n
        s_prev = s_n

        if(method == 'A-Proposed'):
            # Primal-dual spilitting algorithm with denoiser (Gaussian noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'B-Proposed'):
            # Primal-dual spilitting algorithm with denoiser  (Gaussian noise + sparse noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl, r)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        elif(method == 'C-Proposed'):
            # Primal-dual spilitting algorithm with denoiser (Poisson noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * adj_phi(y_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)



        ########################################################
        #   For Experiment A (Gaussian noise )                 #
        ########################################################

        elif(method == 'A-PnPFBS-DnCNN'):
            # Forward-backward spilitting algorithm with DnCNN
            x_n = denoiser_J.denoise(x_n - gamma1 * myLambda * 0.5 * (op.grad_x_l2(x_n, np.zeros(x_n.shape), phi, adj_phi, x_obsrv)))
        elif(method == 'A-PnPPDS-BM3D'):
            # BM3D-PnP-PDS (constrained formulation)
            x_n = x_n - gamma1 *adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=np.sqrt(gamma1))
            x_n = np.moveaxis(x_n, -1, 0)
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'A-PnPFBS-BM3D'):
            # BM3D-PnP-FBS (additive formulation)
            x_n = x_n - gamma1 * op.grad_x_l2(x_n, np.zeros(x_n.shape) , phi, adj_phi, x_obsrv)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=1)
            x_n = np.moveaxis(x_n, -1, 0)       
        elif(method == 'A-PDS-TV'):
            # Primal-dual spilitting algorithm with TV
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev))
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'A-FBS-TV'):
            # Primal-dual splitting with TV (additive formulation):
            x_n = x_n - gamma1 * (adj_phi(phi(x_n)-x_obsrv) + op.D_T(y1_n))
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)   
        elif(method == 'A-RED-DnCNN'):
            # DnCNN RED 
            # https://arxiv.org/pdf/1611.02862.pdf のsigmaをgamma1にlambdaをmyLambdaに置き換えた
            x_n = denoiser_J.denoise(x_n)
            mu = 2 / (1/gamma1**2 + myLambda)
            x_n = x_prev - mu * ((1 / gamma1**2) * adj_phi(phi(x_prev) - x_obsrv) + myLambda * (x_prev - x_n))
        elif(method == 'A-PnPPDS-unstable-DnCNN'):
            x_n = x_n - gamma1 * adj_phi(y_n)
            x_n_tensor = torch.from_numpy(np.ascontiguousarray(x_n)).float().unsqueeze(0)
            x_n_tensor = denoiser_KAIR(x_n_tensor)
            x_n = x_n_tensor.data.squeeze().detach().numpy().copy()
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)


        ########################################################
        #   For Experiment B (Sparse noise + Gaussian noise )  #
        ########################################################
        elif(method == 'comparisonB-1'):
            # BM3D-PnP-PDS (Gaussian noise + sparse noise)
            x_n = x_n - gamma1 * adj_phi(y_n)
            x_n = np.moveaxis(x_n, 0, 2)
            x_n = bm3d.bm3d_rgb(x_n, sigma_psd=0.01)
            x_n = np.moveaxis(x_n, -1, 0)
            s_n = op.proj_l1_ball(s_n - gamma1 * y_n, alpha_s, sp_nl)
            y_n = y_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'comparisonB-2'):
            # ADMM algorithm with denoiser  (Gaussian noise + sparse noise)
            x_n = step1ofADMMforSparseX(s_n, z_n, y_n, phi, adj_phi, path_prox, ch, gamma1, m1)
            s_n = step1ofADMMforSparseS(x_n, z_n, y_n, phi, alpha_s, sp_nl, gamma1, m2)
            z_n = op.proj_l2_ball(phi(x_n) + s_n + y_n, alpha_n, gaussian_nl, sp_nl, x_obsrv)
            y_n = y_n + phi(x_n) + s_n - z_n
        elif(method == 'comparisonB-3'):
            # HTV (constrained formulation)
            x_n = x_n - gamma1 * (op.D_T(y1_n) + adj_phi(y2_n))
            s_n = op.proj_l1_ball(s_n - gamma1 * y2_n, alpha_s, sp_nl)
            y1_n = y1_n + gamma2 * op.D(2 * x_n - x_prev)
            y1_n = y1_n - gamma2 * op.prox_l12(y1_n / gamma2, 1 / gamma2)
            y2_n = y2_n + gamma2 * (phi(2 * x_n - x_prev) + 2 * s_n - s_prev)
            y2_n = y2_n - gamma2 * op.proj_l2_ball(y2_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv)
        elif(method == 'comparisonB-4'):
            # DnCNN RED 
            # https://arxiv.org/pdf/1611.02862.pdf のsigmaをgamma1にlambdaをmyLambdaに置き換えた
            x_n = x_prev - gamma1 * (myLambda * adj_phi(phi(x_prev) + s_n - x_obsrv) + (x_prev - denoiser_J.denoise(x_n)))
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_obsrv)), alpha_s, sp_nl)
        elif(method == 'comparisonB-5'):
            # DnCNN-PnP-FBS (additive formulation)
            x_n = denoiser_J.denoise(x_n - gamma1 * (op.grad_x_l2(x_n, s_n, phi, adj_phi, x_obsrv)))
            s_n = op.proj_l1_ball(s_n - gamma1 * (op.grad_s_l2(x_n, s_n, phi, x_obsrv)), alpha_s, sp_nl)


        ########################################################
        #   For Experiment C (Poisson noise )                  #
        ########################################################

        elif(method == 'C-PnPPDS-BM3D'):
            # BM3D-PnP-PDS (Poisson noise) 
            x_n = bm3d.bm3d(x_n - gamma1 * adj_phi(y_n), sigma_psd = np.sqrt(gamma1))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)
        elif(method == 'C-PnPADMM-DnCNN'):
            # DnCNN-PnP-ADMM (Poisson noise)
            x_n = step1ofADMMforPoisson (d_n, z_n, x_obsrv, phi, adj_phi, poisson_alpha, myLambda, m1, gammaInADMMStep1)
            z_n = denoiser_J.denoise(x_n + d_n)
            d_n = d_n + x_n - z_n
        elif(method == 'C-RED-DnCNN'):
            # DnCNN RED (Poisson noise)
            # https://arxiv.org/pdf/1611.02862.pdf のu_nをd_nに、v_nをz_nに置き換えた
            # Step2で使うlambdaはgamma1を用いる
            x_n = step1ofADMMforPoisson (d_n, z_n, x_obsrv, phi, adj_phi, poisson_alpha, myLambda, m1, gammaInADMMStep1)
            z_n = step2ofADMM_REDforPoisson (x_n, d_n, z_n, myLambda,  gamma1, path_prox, ch, m2)
            d_n = d_n + x_n - z_n
        elif(method == 'C-PnP-unstable-DnCNN'):
            # Not firmly-nonexpansive DnCNN (Poisson noise)
            x_n = x_n - gamma1 * adj_phi(y_n)
            x_n_tensor = torch.from_numpy(np.ascontiguousarray(x_n)).float().unsqueeze(0).unsqueeze(0)
            x_n_tensor = denoiser_KAIR(x_n_tensor)
            x_n = x_n_tensor.data.squeeze().squeeze().detach().numpy().copy()
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_alpha, x_obsrv)


        else:
            print("Unknown method:", method)
            return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time

        c[i] = np.linalg.norm((x_n - x_prev).flatten(), 2) / np.linalg.norm(x_prev.flatten(), 2)
        psnr_data[i] = eval_psnr(x_true, x_n)
        ssim_data[i] = eval_ssim(x_true, x_n)


    torch.cuda.synchronize(); 
    end_time = time.process_time()
    average_time = (end_time - start_time)/max_iter

    return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time

