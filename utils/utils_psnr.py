import numpy as np

def eval_psnr(img_1, img_2):
    data_range=1
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)
