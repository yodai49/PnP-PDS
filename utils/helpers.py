import cv2
import numpy as np

def save_img(picture, path_picture):
    picture_temp = picture
    picture_temp = np.moveaxis(picture, 0, 2)
    picture_temp[picture_temp > 1.] = 1.
    picture_temp[picture_temp < 0.] = 0.

    cv2.imwrite(path_picture, np.uint8(picture_temp*255.))
    
def save_imgs(pictures, path_pictures, format):
    for i, picture in enumerate(pictures):
        save_img(picture = picture, path_picture = path_pictures[i] + format)