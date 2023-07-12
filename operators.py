import scipy
import scipy.io
import numpy as np

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

def get_operators(shape, gamma1, gamma2, path_kernel):
    def phi(x):
        return get_blur_operator(x, h)
    def adj_phi(x):
        return get_adj_blur_operator(x, h)
    def prox_g(x):
        return np.sign(x) * np.fmax(0, np.abs(x) - gamma1)
    def prox_h(x):
        return np.sign(x) * np.fmax(0, np.abs(x) - gamma2)
    
    h = scipy.io.loadmat(path_kernel)
    h = np.array(h['blur'])
    size = np.prod(shape)
    L = np.eye(size, size)
    return phi, adj_phi, prox_g, prox_h, L