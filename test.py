import argparse
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from operators import get_observation_operators
from pds import test_iter
import operators as op
from utils.helpers import save_imgs, add_salt_and_pepper_noise

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
        gridEpsilon = 0.7 + (i / grid_num)*0.3
        param_psnr[i] = 0
        param_epsilon_dash[i] = gridEpsilon
        print('epsilon_dash: ', gridEpsilon)
        param_psnr[i], param_psnr_best[i] = eval_restoration(gaussian_nl=opt.gaussian_nl, sp_nl=opt.sp_nl, max_iter = opt.max_iter, gamma1 = opt.gamma1, gamma2 = opt.gamma2, alpha_n = gridEpsilon, alpha_s = opt.alpha_s, result_output=False, method = opt.method)
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


def eval_restoration(max_iter = 1000, gaussian_nl = 0.01, sp_nl = 0.01, gamma1 = 1.99, gamma2 = 1.99, alpha_n = 0.9, alpha_s = 0.9, result_output = True, method = 'ours-A'):
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

        phi, adj_phi = get_observation_operators(operator = opt.deg_op, path_kernel = path_kernel, r = opt.r)
        gaussian_noise = np.random.randn(*img_true.shape)
        img_blur = phi(np.copy(img_true)) + gaussian_nl * gaussian_noise
        img_blur = add_salt_and_pepper_noise(img_blur, sp_nl)
        
        x_0 = np.copy(img_blur)
        
        #epsilon = np.linalg.norm(gaussian_noise) / np.sqrt(img_true.size) # oracle
        #print(epsilon)
        
        img_sol, s_sol, _, psnr = test_iter(x_0, img_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, gaussian_nl, sp_nl, path_prox, max_iter, method)
        
        print(path_img)
        print('PSNR: ', psnr[-1])
        if(result_output):
            pictures = [img_true, img_blur, img_sol, s_sol]
            path_pictures = [path_result + path_img[path_img.rfind('\\'):] + '_true',  path_result +  path_img[path_img.rfind('\\'):] + '_blur', path_result + path_img[path_img.rfind('\\'):]+ '_sol', path_result + path_img[path_img.rfind('\\'):]+ '_sp_noise']
            save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')

    if(result_output):
        x = np.arange(0, max_iter, 1)
        plt.title('PSNR')
        #plt.gca().set_yscale('log')
        plt.plot(x, psnr)
        plt.xlabel('iteration')
        plt.ylabel('PSNR')
        plt.show()

    return psnr[-1], np.max(psnr)

if (__name__ == '__main__'):
    #grid_search(3)
    eval_restoration(gaussian_nl=opt.gaussian_nl, sp_nl=opt.sp_nl, max_iter = opt.max_iter, gamma1 = opt.gamma1, gamma2 = opt.gamma2, alpha_n = opt.alpha_n, alpha_s = opt.alpha_s, result_output=True, method = opt.method)