import numpy as np

def add_salt_and_pepper_noise(img, noise_level, random_sampling_op):
    noise_cnt = (int)(img.shape[-2] * img.shape[-1] * noise_level / 2)
    noise_target = np.ones([img.shape[-2], img.shape[-1]])
    noise_target = random_sampling_op(noise_target)
    noise_list = np.arange(0)
    np.random.seed(1234)
    sp_noise_x = np.arange(0)
    sp_noise_y = np.arange(0)
    for i in range(0, noise_cnt * 2):
        x = np.random.randint(0, img.shape[-2])
        y = np.random.randint(0, img.shape[-2])
        if (noise_target[x][y] == 1 and (x * img.shape[-2] + y in noise_list) == False):
            sp_noise_x=np.append(sp_noise_x, x)
            sp_noise_y=np.append(sp_noise_y, y)
            noise_list=np.append(noise_list, x * img.shape[-2] + y)
        else:
            i=i-1
#    sp_noise_x = np.random.randint(0, img.shape[-2], noise_cnt * 2)
#    sp_noise_y = np.random.randint(0, img.shape[-1], noise_cnt * 2)
    myImg = np.copy(img)
    if(np.ndim(myImg) == 2):
        #grayscale
        myImg[(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 0
        myImg[(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 1
    elif(np.ndim(myImg) == 3):
        for i in range(0, 3):
            myImg[i][(sp_noise_x[:noise_cnt],sp_noise_y[:noise_cnt])] = 0
            myImg[i][(sp_noise_x[noise_cnt:],sp_noise_y[noise_cnt:])] = 1
    return myImg

def add_gaussian_noise(img, noise_level, random_sampling_op):
    np.random.seed(1234)
    gaussian_noise = random_sampling_op(noise_level * np.random.randn(*img.shape))
    return img + gaussian_noise

def apply_poisson_noise(img, alpha):
    np.random.seed(1234)
    val = np.random.poisson(img * alpha)
    return val
