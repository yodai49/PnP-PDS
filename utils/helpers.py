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
        
def add_salt_and_pepper_noise(img, noise_level):
    noise_cnt = (int)(img.shape[1] * img.shape[2] * noise_level / 2)
    sp_noise_x = np.random.randint(0, img.shape[1], noise_cnt * 2)
    sp_noise_y = np.random.randint(0, img.shape[2], noise_cnt * 2)
    img[0][(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 1
    img[1][(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 1
    img[2][(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 1
    img[0][(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 0
    img[1][(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 0
    img[2][(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 0
    return img

def add_gaussian_noise(img, noise_level):
    np.random.seed(1234)
    gaussian_noise = noise_level * np.random.randn(*img.shape)
    return img + gaussian_noise

def apply_poisson_noise(img, alpha):
    np.random.seed(1234)
#    img = img.transpose(2, 1, 0)
#    for i, attr in enumerate(img):
#        for j, attr in enumerate(img[i]):
#            average_in_pixel = (img[i][j][0] + img[i][j][1] + img[i][j][2])/3
#            pixel_with_poisson_noise = np.random.poisson(average_in_pixel * alpha, 1)/alpha
#            for k, attr in enumerate(img[i][j]):
#                img[i][j][k] = pixel_with_poisson_noise
#    img = img.transpose(2, 1, 0)
    val = np.random.poisson(img * alpha)
    return val