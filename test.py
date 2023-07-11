import argparse
import glob
import json
import os
import torch

from pds import test_pds 
from operators import get_operators

parser = argparse.ArgumentParser(description="testing PnP-PDS")
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="type of network")
parser.add_argument("--gamma", type=float, default=1.99, help='step size for the dual problem')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--max_iter", type=int, default=1000, help='max iteration of the pds algorithm')
parser.add_argument("--n_ch", type=int, default=3, help="channels")
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--pth_config_file", type=str, default='config/setup.json')
opt = parser.parse_args()
with open(opt.pth_config_file, 'r') as f:
    config = json.load(f)

def eval_pds(max_iter = 1000, noise_level = 0.003,):
    path_test = config['path_test']
    pattern_red = config['pattern_red']

    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))

    for path_img in path_images:
        img_blur = path_img
        grad_f, prox_g, prox_h = get_operators() 
        L = 1
        #img_sol, img_best = 
        test_pds(x_0 = img_blur, grad_f = grad_f, prox_g = prox_g, prox_h = prox_h, L = L, max_iter = max_iter,)

if (__name__ == '__main__'):
    noises = [0.005]
    for noise_level in noises:
        eval_pds(noise_level=noise_level, max_iter = opt.max_iter)