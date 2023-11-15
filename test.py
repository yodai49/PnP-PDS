import argparse
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from operators import get_observation_operators
from pds import test_iter
import operators as op
from utils.helpers import save_imgs, add_salt_and_pepper_noise, add_gaussian_noise, apply_poisson_noise

parser = argparse.ArgumentParser(description="testing PnP-PDS")
parser.add_argument("--method", type=str, default='ours-A', help='method name')
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="architecture of network")
parser.add_argument("--gamma1", type=float, default=1.99, help='step size for the primal problem')
parser.add_argument("--gamma2", type=float, default=1.99, help='step size for the dual problem')
parser.add_argument("--alpha_n", type=float, default=0.9, help='parameter of the l2 regularization')
parser.add_argument("--alpha_s", type=float, default=0.9, help='parameter of the l1 regularization (for sparse noise)')
parser.add_argument("--max_iter", type=int, default=1000, help='max iteration of the pds algorithm')
parser.add_argument("--n_ch", type=int, default=3, help="channels")
parser.add_argument("--gaussian_nl", type=float, default=0.01, help='gaussian noise level')
parser.add_argument("--sp_nl", type=float, default=0.01, help='salt and pepper noise level')
parser.add_argument("--deg_op", type=str, default='blur', help='degradation operator')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--r", type=float, default=0.2, help='level of random sampling')
parser.add_argument("--pth_config_file", type=str, default='config/setup.json')
opt = parser.parse_args()
with open(opt.pth_config_file, 'r') as f:
    config = json.load(f)

def grid_search(grid_num = 6):
    param_psnr = np.zeros((grid_num))
    param_psnr_best = np.zeros((grid_num))
    param_epsilon_dash = np.zeros((grid_num))
    for i in range(0, grid_num):
        gridEpsilon = 0.6 + (i / grid_num)*0.6
        param_psnr[i] = 0
        param_epsilon_dash[i] = gridEpsilon
        print('epsilon_dash: ', gridEpsilon)
        param_psnr[i] = eval_restoration(gaussian_nl=0.01, sp_nl=0, poisson_noise=False, max_iter = 400, gamma1 = 0.99, gamma2 = 0.1, r=1, alpha_n = gridEpsilon, alpha_s = 0, myLambda=1, result_output=False, architecture='preDnCNN_nobn_nch_3_nlev_0.01', deg_op = 'blur', method = 'ours-A')
    x = param_epsilon_dash.flatten()
    y = param_psnr.flatten()
    #z = param_psnr_best.flatten()
    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111, projection='3d')
    
    #ax.scatter3D(x, y, z, label='PSNR')
    plt.title('Convergence')
    #plt.gca().set_yscale('log')
    plt.scatter(x, y)
    #plt.scatter(x, z)
    plt.xlabel('epsilon')
    plt.ylabel('PSNR')

    plt.show()


def eval_restoration(max_iter = 1000, gaussian_nl = 0.01, sp_nl = 0.01, poisson_noise = True, poisson_alpha = 0.1, gamma1 = 1.99, gamma2 = 1.99, alpha_n = 0.9, alpha_s = 0.9, myLambda=1, result_output = True, method = 'ours-A', architecture = '', r=0.5, deg_op = 'blur'):
    # 戻り値：psnr（すべての画像の平均値）

    path_test = config['path_test']
    pattern_red = config['pattern_red']
    path_result = config['path_result']

    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))
    path_kernel = config['root_folder'] + 'blur_models/' + opt.kernel + '.mat'
    path_prox = config['root_folder'] + 'nn/' + architecture + '.pth'

    psnr = np.zeros((len(path_images))) # 各画像のイタレーション終了時のPSNRを格納した配列
    cnt = 0

    for path_img in path_images:
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        img_true = np.moveaxis(img_true, -1, 0)

        phi, adj_phi = get_observation_operators(operator = deg_op, path_kernel = path_kernel, r = r)
        img_blur = phi(img_true)
        img_blur = add_gaussian_noise(img_blur, gaussian_nl)
        if(poisson_noise):
            img_blur = apply_poisson_noise(img_blur, poisson_alpha)
        img_blur = add_salt_and_pepper_noise(img_blur, sp_nl)
        
        x_0 = np.copy(img_blur)
        
        #epsilon = np.linalg.norm(gaussian_noise) / np.sqrt(img_true.size) # oracle
        #print(epsilon)
        
        img_sol, s_sol, temp_c, temp_psnr = test_iter(x_0, img_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, gaussian_nl, sp_nl, path_prox, max_iter, method)

        print(path_img)
        psnr[cnt] = temp_psnr[-1]
        print('PSNR: ', psnr[cnt])

        pictures = [img_true, img_blur, img_sol]
        timestamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        print(timestamp)
        path_pictures = [path_result + path_img[path_img.rfind('\\'):] + '_true_' + method + '(' + deg_op + ')_gaussiannl' + str(gaussian_nl) + '(' + timestamp + ')',  path_result +  path_img[path_img.rfind('\\'):] + '_blur_' + method + '(' + deg_op + ')_gaussian-nl' + str(gaussian_nl) + '(' + timestamp + ')', path_result + path_img[path_img.rfind('\\'):]+ '_sol_'  + method + '(' + deg_op + ')_gaussian-nl' + str(gaussian_nl) + '(' + timestamp + ')']
        save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')

        np.save(path_result + 'PSNR_' + method + '(' + deg_op + ')_nl' + str(gaussian_nl) + '_' + path_img[-8:], temp_psnr)
        np.save(path_result + 'c_' + method + '(' + deg_op + ')_nl' + str(gaussian_nl) + '_' + path_img[-8:], temp_c)

        cnt=cnt+1

        if(result_output):
            x = np.arange(0, max_iter, 1)
            plt.title('PSNR')
           #plt.gca().set_yscale('log')
            plt.plot(x, temp_psnr)
            plt.xlabel('iteration')
            plt.ylabel('PSNR')
            plt.show()

    params = {'mean_PSNR':np.mean(psnr), 'PSNR':psnr, 'gamma1': gamma1, 'gamma2': gamma2, 'alpha_n': alpha_n, 'gaussian_nl':gaussian_nl, 'sp_nl':sp_nl, 'alpha_n':alpha_n, 'alpha_s':alpha_s, 'max_iter':max_iter, 'myLambda': myLambda, 'r':r, deg_op:'deg_op'}
    np.save(path_result + 'RESULT_AND_PARAMS_' + method + '(' + deg_op + ')_nl' + str(gaussian_nl) + '_' + path_img[-8:], params)

    return np.mean(psnr)

if (__name__ == '__main__'):
    #grid_search(24)

    psnr = eval_restoration(gaussian_nl=0.0, sp_nl=0.0, poisson_noise=True, poisson_alpha=0.01, max_iter = 40, gamma1 = 0.99, gamma2 = 0.1, r=1, alpha_n = 0.9, alpha_s = 1, myLambda=1, result_output=True, architecture='preDnCNN_nobn_nch_3_nlev_0.01', deg_op = 'blur', method = 'ours-B')

    #python test.py --max_iter=2000 --gamma1=0.49 --gamma2=0.99 --gaussian_nl=0.01 --sp_nl=0.0 --architecture=preDnCNN_nobn_nch_3_nlev_0.01 --alpha_n=0.95 --method=comparisonC-1 --r=0.7 --deg_op=random_sampling