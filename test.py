import argparse
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from operators import get_operators, get_blur_operators
from pds import test_iter
from utils.helpers import save_imgs

parser = argparse.ArgumentParser(description="testing PnP-PDS")
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="architecture of network")
parser.add_argument("--gamma1", type=float, default=1.99, help='step size for the primal problem')
parser.add_argument("--gamma2", type=float, default=1.99, help='step size for the dual problem')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--lambda1", type=float, default=0.1, help='parameter of the function g')
parser.add_argument("--lambda2", type=float, default=0.1, help='parameter of the function h')
parser.add_argument("--max_iter", type=int, default=1000, help='max iteration of the pds algorithm')
parser.add_argument("--n_ch", type=int, default=3, help="channels")
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--pth_config_file", type=str, default='config/setup.json')
opt = parser.parse_args()
with open(opt.pth_config_file, 'r') as f:
    config = json.load(f)

def grid_search(grid_num = 6):
    param_psnr = np.zeros((grid_num))
    param_psnr_best = np.zeros((grid_num))
    param_epsilon_dash = np.zeros((grid_num))
    for i in range(0, grid_num):
        epsilon_dash = 0.45 + (i / grid_num)*0.2
        param_psnr[i] = 0
        param_epsilon_dash[i] = epsilon_dash
        print('epsilon_dash: ', epsilon_dash)
        param_psnr[i], param_psnr_best[i] = eval_pds(noise_level=opt.noise_level, max_iter = opt.max_iter, gamma1 = opt.gamma1, gamma2 = opt.gamma2, lambda1 = opt.lambda1, lambda2 = opt.lambda2, epsilon_dash = epsilon_dash, result_output=False)
    x = param_epsilon_dash.flatten()
    y = param_psnr.flatten()
    z = param_psnr_best.flatten()
    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111, projection='3d')
    
    #ax.scatter3D(x, y, z, label='PSNR')
    plt.title('Convergence')
    #plt.gca().set_yscale('log')
    plt.scatter(x, y)
    plt.scatter(x, z)
    plt.xlabel('epsilon')
    plt.ylabel('PSNR')

    plt.show()


def eval_pds(max_iter = 1000, noise_level = 0.01, gamma1 = 1.99, gamma2 = 1.99, lambda1 = 0.1, epsilon_dash = 0.55, lambda2 = 0.1, result_output = True):
    path_test = config['path_test']
    pattern_red = config['pattern_red']
    path_result = config['path_result']

    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))
    path_kernel = config['root_folder'] + 'blur_models/' + opt.kernel + '.mat'
    path_prox = config['root_folder'] + 'nn/' + opt.architecture + '.pth'

    for path_img in path_images:
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        img_true = np.moveaxis(img_true, -1, 0)

        phi, adj_phi = get_blur_operators(shape = img_true.shape, path_kernel = path_kernel)
        noise = np.random.randn(*img_true.shape)
        img_blur = phi(np.copy(img_true)) + noise_level * noise
        x_0 = np.copy(img_blur)
        
        grad_f, prox_g, prox_h_dual, L = get_operators(shape = img_true.shape, gamma1 = gamma1, gamma2 = gamma2, lambda1 = lambda1, lambda2 = lambda2, epsilon_dash=epsilon_dash, phi = phi, adj_phi = adj_phi, path_prox = path_prox, x_0 = x_0)
        
        img_sol, c, psnr = test_iter(x_0 = x_0, x_true = img_true, grad_f = grad_f, prox_g = prox_g, prox_h_dual = prox_h_dual, L = phi, L_T = adj_phi, gamma1 = gamma1, gamma2 = gamma2, max_iter = max_iter, method = "PDS")
        #img_sol, c_pds, psnr_pds = test_iter(x_0 = x_0, x_true = img_true, grad_f = grad_f, prox_g = prox_g, prox_h_dual = prox_h_dual, L = L, gamma1 = gamma1, gamma2 = gamma2, max_iter = max_iter)
            
        print(path_img)
        print('PSNR: ', psnr[-1])
        if(result_output):
            pictures = [img_true, img_blur, img_sol]
            path_pictures = [path_result + path_img[path_img.rfind('\\'):] + '_true',  path_result +  path_img[path_img.rfind('\\'):] + '_blur', path_result + path_img[path_img.rfind('\\'):]+ '_sol']
            save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')

    if(result_output):
        x = np.arange(0, max_iter, 1)
        plt.title('Convergence')
        #plt.gca().set_yscale('log')
        plt.plot(x, psnr)
        plt.xlabel('iteration')
        plt.ylabel('c_n')
        plt.show()

    return psnr[-1], np.max(psnr)

if (__name__ == '__main__'):
    #grid_search(2)
    eval_pds(noise_level=opt.noise_level, max_iter = opt.max_iter, gamma1 = opt.gamma1, gamma2 = opt.gamma2, lambda1 = opt.lambda1, lambda2 = opt.lambda2, epsilon_dash = 0.69, result_output=True)