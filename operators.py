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
    # return x*h where * is a convolution
    if(np.ndim(x) == 3):
        # color image
        l = h.shape[0]
        x = np.pad(x, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
        y = np.zeros(x.shape)
        h = np.fft.fft2(h, [x.shape[-2], x.shape[-1]])
        for i in range(x.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(h) * np.fft.fft2(x[i, ...])))

    return y[..., :-l+1, :-l+1]

def get_operators(shape, gamma1, gamma2, lambda1, lambda2, path_kernel, path_prox):
    def phi(x):
        return get_blur_operator(x, h)
    def adj_phi(x):
        return get_adj_blur_operator(x, h)
    def prox_g(x):
        denoiser = Denoiser(file_name=path_prox)
        return denoiser.denoise(x)
        #return np.sign(x) * np.fmax(0, np.abs(x) - lambda1 * gamma1)
    def prox_h(x):
        return np.fmax(0, np.fmin(1, x))
    def prox_h_dual(x):
        return x - gamma2 * prox_h(x / gamma2)
    
    h = scipy.io.loadmat(path_kernel)
    h = np.array(h['blur'])
    size = np.prod(shape)
    L = np.eye(size, size)
    return phi, adj_phi, prox_g, prox_h_dual, L