import cv2, datetime, glob, json, os
import numpy as np
import random

def trim_images (path_org, path_res, pattern_red, count = 7, size = 128):
    random.seed (1234)
    path_images = sorted(glob.glob(os.path.join(path_org, pattern_red)))
    filtered_images = random.sample(path_images, count)
    for path_img in filtered_images:
        img_org = cv2.imread(path_img)
        file_name =  (path_img[path_img.rfind('\\'):])[1:]
        x_size = img_org.shape[0]
        y_size = img_org.shape[1]
        x_from = (int)(x_size / 2) - (int)(size / 2)
        x_to = x_from + size
        y_from = (int)(y_size / 2) - (int)(size / 2)
        y_to = y_from + size
        img_trimmed = img_org[x_from:x_to, y_from:y_to, :]
        cv2.imwrite(path_res + file_name + '.png', img_trimmed)

if (__name__ == '__main__'):

    trim_images('C:/Users/temp/Documents/lab/images/ILSVRC2012_img/', 'C:/Users/temp/Documents/lab/images/imagenet_trimmed/', '*.JPEG', 7, 128)