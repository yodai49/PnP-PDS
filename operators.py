import scipy
import scipy.io
import numpy as np

from models.denoiser import Denoiser

def get_blur_operator(x, h):
    # return x*h where * is a convolution
    if(np.ndim(x) == 3):
        # color image
        l = h.shape[0]
        x = np.pad(x, ((0, 0), (l//2+1, l//2), (l//2+1, l//2)), 'wrap')
        y = np.zeros(x.shape)
        h = np.fft.fft2(h, [x.shape[-2], x.shape[-1]])
        for i in range(x.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(h * np.fft.fft2(x[i, ...])))

    return y[..., l:, l:]

def get_adj_blur_operator(x, h):
    l = h.shape[0]
    x = np.pad(x, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
    y = np.zeros(x.shape)
    h = np.fft.fft2(h, [x.shape[-2], x.shape[-1]])
    for i in range(x.shape[0]):
        y[i, ...] = np.real(np.fft.ifft2(np.conj(h) * np.fft.fft2(x[i, ...])))
    return y[..., :-l+1, :-l+1]

def get_random_sampling_operator(x, r):
    np.random.seed(1234)
    w = x.shape[-2]
    h = x.shape[-1]
    degraded_cnt = round(w * h * (1-r))
    Q = np.random.permutation(w * h)[:degraded_cnt]
    for i in range(0,3):
        X = x[i].flatten()
        X[Q] = 0
        x[i] = X.reshape(w, h)    
    return x    

def get_observation_operators(operator, path_kernel, r):
    def phi(x):
        if(operator == 'blur'):
            return get_blur_operator(x, h)
        elif (operator == 'random_sampling'):
            return get_random_sampling_operator(x, r)    
                
    def adj_phi(x):
        if(operator == 'blur'):
            return get_adj_blur_operator(x, h)
        elif (operator == 'random_sampling'):
            return get_random_sampling_operator(x, r)
    
    h = scipy.io.loadmat(path_kernel)
    h = np.array(h['blur'])
    return phi, adj_phi
    
def denoise(x, path_prox):
    denoiser = Denoiser(file_name=path_prox)
    return denoiser.denoise(x)

def grad_x_l2(x, s, phi, adj_phi, x_0):
    return 2 * adj_phi(phi(x) + s - x_0)

def grad_s_l2(x, s, phi, x_0):
    return 2 * (phi(x) + s - x_0)
    
def proj_l1_ball(x, alpha_s, sp_nl):
    # Projection on l1 ball
    eta = alpha_s * x.size * sp_nl * 0.5
    y = x.reshape((-1))
    y = np.fmax(np.abs(y)-np.fmax(np.max(((np.cumsum(np.sort(np.abs(y))[::-1])-eta)/(np.arange(1, len(y) + 1))).conj().T), 0), 0)*np.sign(y)
    val = y.reshape(x.shape)
    return val

def proj_l2_ball(x, alpha_n, gaussian_nl, sp_nl, x_0):
    # projection on l2 ball
    epsilon = np.sqrt(x.size * (1 - sp_nl)) * alpha_n * gaussian_nl
    val = x
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val

def prox_l12(x, gamma):
    myval = gamma/np.sqrt(np.sum(x*x, 0))
    return np.fmax(1 - myval, 0) * x

def prox_box_constraint(x):
    return np.fmax(0, np.fmin(1, x))

def D(x):
    # input: x (COLOR, W, H)
    # output: (COLOR*2, W, H)
    x_v_cnt = np.shape(x)[1]
    x_h_cnt = np.shape(x)[2]
    x_v = np.concatenate([np.diff(x, n=1, axis=1)[:,0:x_v_cnt-1,:], np.zeros((3, 1, x_h_cnt))], 1)
    x_h = np.concatenate([np.diff(x, n=1, axis=2)[:,:,0:x_h_cnt-1], np.zeros((3, x_v_cnt, 1))], 2)
    val = np.concatenate([x_v, x_h], 0)
    return val

def D_T(x):
    # input: x (COLOR*2, W, H)
    # output: (COLOR, W, H)
    x_v = x[0:3, :, :]
    x_h = x[3:6, :, :]
    x_v_cnt = np.shape(x_v)[1]
    x_h_cnt = np.shape(x_h)[2]
    x_v = np.concatenate([-x_v[:,0:1,:], -x_v[:, 1:x_v_cnt-1, :]+ x_v[:, 0:x_v_cnt-2, :], x_v[:,x_v_cnt-1:x_v_cnt,:]], 1)
    x_h = np.concatenate([-x_h[:,:,0:1], -x_h[:, :, 1:x_h_cnt-1]+ x_h[:, :, 0:x_h_cnt-2], x_h[:,:,x_h_cnt-1:x_h_cnt]], 2)
    val = x_v + x_h
    return val