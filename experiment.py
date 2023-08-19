from test import eval_restoration
import numpy as np

if (__name__ == '__main__'):
    psnr = eval_restoration(gaussian_nl=0.01, sp_nl=0, max_iter = 2000, gamma1 = 0.99, gamma2 = 0.1, r=0.8, alpha_n = 0.9, alpha_s = 0, result_output=False, architecture='preDnCNN_nobn_nch_3_nlev_0.01', deg_op = 'random_sampling', method = 'comparisonC-1')
    print(psnr, np.mean(psnr))
