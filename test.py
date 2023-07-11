import argparse
import glob
import json
import numpy as np
import os
import torch

from pds import test_pds
from PIL import Image
from operators import get_operators

parser = argparse.ArgumentParser(description="testing PnP-PDS")
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="type of network")
parser.add_argument("--gamma1", type=float, default=1.99, help='step size for the primal problem')
parser.add_argument("--gamma2", type=float, default=1.99, help='step size for the dual problem')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--max_iter", type=int, default=1000, help='max iteration of the pds algorithm')
parser.add_argument("--n_ch", type=int, default=3, help="channels")
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--pth_config_file", type=str, default='config/setup.json')
opt = parser.parse_args()
with open(opt.pth_config_file, 'r') as f:
    config = json.load(f)

def eval_pds(max_iter = 1000, noise_level = 0.003, gamma1 = 1.99, gamma2 = 1.99):
    path_test = config['path_test']
    pattern_red = config['pattern_red']

    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))

    for path_img in path_images:
        img_true = Image.open(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        img_true = np.moveaxis(img_true, -1, 0)

        phi, grad_f, prox_g, prox_h, L = get_operators(shape = img_true.shape, gamma1 = gamma1, gamma2 = gamma2)

        noise = noise_level * np.random.randn(*img_true.shape)
        img_blur = phi(img_true) + noise

        x_0 = img_blur.flatten()
        img_sol = test_pds(x_0 = x_0, grad_f = grad_f, prox_g = prox_g, prox_h = prox_h, L = L, gamma1 = gamma1, gamma2 = gamma2, max_iter = max_iter)
        img_sol = np.reshape(img_sol, img_true.shape)
        print(img_blur)
        print(img_sol)

        # 画像を保存する部分をここに追加


if (__name__ == '__main__'):
    noises = [0.005]
    for noise_level in noises:
        eval_pds(noise_level=noise_level, max_iter = opt.max_iter, gamma1 = opt.gamma1, gamma2 = opt.gamma2)