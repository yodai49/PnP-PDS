import matplotlib.pyplot as plt
import operators as op
import cv2
import numpy as np

def check_homogeneity(ch=3, epsilon = 0.05, path_prox = '', path_images = ''):
    # デノイザーがhomogeneityを満たすかどうかを確認する

    for path_img in path_images:
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        img_true = np.moveaxis(img_true, -1, 0)
        img_x = (1 + epsilon) * op.denoise(img_true, path_prox, ch)
        img_y = op.denoise((1 + epsilon) * img_true, path_prox, ch)
        x = img_x.flatten()
        y = img_y.flatten()
        # = plt.figure()
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x, y, s=1, c=None, marker='.')
        plt.xlabel('$(1+\\varepsilon)f(x)$')
        plt.ylabel('$f((1+\\varepsilon)x)$')
        plt.grid()
        plt.show()
#        fig.savefig('./result-homogeneity/' + os.path.splitext(os.path.basename(path_img))[0] + '.png')
    return 0