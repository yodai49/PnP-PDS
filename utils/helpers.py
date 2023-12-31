import cv2
import numpy as np

def save_img(picture, path_picture):
    picture_temp = picture
    if(np.ndim(picture) == 3):
        # color
        picture_temp = np.moveaxis(picture, 0, 2)
    picture_temp[picture_temp > 1.] = 1.
    picture_temp[picture_temp < 0.] = 0.

    cv2.imwrite(path_picture, np.uint8(picture_temp*255.))
    
def save_imgs(pictures, path_pictures, format):
    for i, picture in enumerate(pictures):
        save_img(picture = picture, path_picture = path_pictures[i] + format)
        
def add_salt_and_pepper_noise(img, noise_level):
    noise_cnt = (int)(img.shape[-2] * img.shape[-1] * noise_level / 2)
    sp_noise_x = np.random.randint(0, img.shape[-2], noise_cnt * 2)
    sp_noise_y = np.random.randint(0, img.shape[-1], noise_cnt * 2)
    if(np.ndim(img) == 2):
        #grayscale
        img[(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 0
        img[(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 1
    elif(np.ndim(img) == 3):
        for i in range(0, 3):
            img[i][(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 0
            img[i][(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 1
    return img

def add_gaussian_noise(img, noise_level):
    np.random.seed(1234)
    gaussian_noise = noise_level * np.random.randn(*img.shape)
    return img + gaussian_noise

def apply_poisson_noise(img, alpha):
    np.random.seed(1234)
    val = np.random.poisson(img * alpha)
    return val
