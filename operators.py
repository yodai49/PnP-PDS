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
    if(np.ndim(x) == 3):
        # color image
        l = h.shape[0]
        x = np.pad(x, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
        y = np.zeros(x.shape)
        h = np.fft.fft2(h, [x.shape[-2], x.shape[-1]])
        for i in range(x.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(h) * np.fft.fft2(x[i, ...])))
    return y[..., :-l+1, :-l+1]

def get_observation_operators(path_kernel):
    def phi(x):
        return get_blur_operator(x, h)
    def adj_phi(x):
        return get_adj_blur_operator(x, h)
    
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
    
def proj_l1_ball(x, alpha_eta, sp_nl):
    # Projection on l1 ball
    eta = alpha_eta * 0.5 * x.size * sp_nl
    y = x.reshape((-1))
    mymax = np.max((np.cumsum(np.sort(np.abs(y))[::-1])-eta)/(y.size))
    y = np.fmax(np.abs(y)-np.fmax(mymax, 0), 0)*np.sign(y)
    val = y.reshape(x.shape)
    return val

def proj_l2_ball(x, alpha_epsilon, gaussian_nl, x_0):
    # projection on l2 ball 
    val  = x
    epsilon = np.sqrt(x.size) * alpha_epsilon * gaussian_nl
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val

def proj_l2_ball_dual(x, gamma2, alpha_epsilon, gaussian_nl, x_0):
    return x - gamma2 * proj_l2_ball(x / gamma2, alpha_epsilon, gaussian_nl, x_0)

#return np.fmax(0, np.fmin(1, x))  # box constraint
#return np.sign(x) * np.fmax(0, np.abs(x) - lambda1 * gamma1) # prox of l1
