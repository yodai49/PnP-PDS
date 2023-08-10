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
    #max(abs(x)-max(max((cumsum(sort(abs(x),1,'descend'),1)-alpha)./(1:size(x,1))),0),0).*sign(x); MATLAB

    eta = alpha_eta * 0.5 * x.size * sp_nl
    y = x.reshape((-1))
        
    mymax = np.max((np.cumsum(np.sort(np.abs(y))[::-1])-eta)/(np.arange(1, len(y) + 1)))
    y = np.fmax(np.abs(y)-np.max(mymax, 0), 0)*np.sign(y)
    val = y.reshape(x.shape)
    return val

def proj_l2_ball(x, alpha_epsilon, gaussian_nl, sp_nl, x_0):
    # projection on l2 ball
    val = x
    epsilon = np.sqrt(x.size * (1 - sp_nl)) * alpha_epsilon * gaussian_nl
    print(epsilon)
#    epsilon = alpha_epsilon
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val

#def proj_l2_ball_dual(x, gamma, alpha_epsilon, gaussian_nl, x_0):
#    return x - gamma * proj_l2_ball(x / gamma, alpha_epsilon, gaussian_nl, x_0)

def prox_l12(x, gamma):
    myval = gamma/np.sqrt(np.sum(x*x, 0))
    return np.fmax(1 - myval, 0) * x

def prox_l12_dual(x, gamma):
    return x - gamma * prox_l12(x / gamma, 1 / gamma)

def D(x):
    # input: x (COLOR, W, H)
    # output: (COLOR*2, W, H)
    x_v_cnt = np.shape(x)[1]
    x_h_cnt = np.shape(x)[2]
    x_v = np.concatenate([np.diff(x, n=1, axis=1)[:,0:x_v_cnt-1,:], np.zeros((3, 1, x_h_cnt))], 1)
    x_h = np.concatenate([np.diff(x, n=1, axis=2)[:,:,0:x_h_cnt-1], np.zeros((3, x_v_cnt, 1))], 2)
    
#    val = np.concatenate([x_v, np.diff(x, n=1, axis=2)], 0)
#    x_v = np.fmax(0, np.fmin(1, x_v))
#    x_h = np.fmax(0, np.fmin(1, x_h))
    val = np.concatenate([x_v, x_h], 0)
#    val = np.fmax(0, np.fmin(1, val))

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
    #x_h = np.concatenate([-x_h[:,:,0:1], -np.diff(x_h, 1, 2)[:,:,0:x_h_cnt-2], x_h[:,:,x_h_cnt-1:x_h_cnt]], 2)
    val = x_v + x_h

#    x_v = np.fmin(1, np.fmax(0, x[0:3,:,:]))
#    x_h = np.fmin(1 , np.fmax(0,x[3:6,:,:]))
#    print(x.shape)
#    print(np.diff(x_v, 1, 1).shape, np.diff(x_h, 1, 2).shape)
#    x_v2 = np.copy(np.concatenate([-x_v[:,0:1,:], -np.diff(x_v, 1, 1)[:,0:x_v_cnt-2,:], x_v[:,x_v_cnt-1:x_v_cnt,:]], 1))
#    x_h2 = np.copy(np.concatenate([-x_h[:,:,0:1], -np.diff(x_h, 1, 2)[:,:,0:x_h_cnt-2], x_h[:,:,x_h_cnt-1:x_h_cnt]], 2))
#    print(x_v2.shape, x_h2.shape)
#    val = -np.diff(x_v, n=1, axis=1, prepend=0) - np.diff(x_h, n=1, axis=2, prepend=0)
#    val = x_v2 + x_h2
#    x_v = np.concatenate([-np.copy(x_v[:,0:1,:]), np.copy(-x_v[:, 1:x_v_cnt-1, :]) + np.copy(x_v[:, 0:x_v_cnt-2, :]), np.copy(x_v[:,x_v_cnt-1:x_v_cnt,:])], 1)
   # x_v = np.concatenate([-x_v[:,0:1,:], -x_v[:, 1:x_v_cnt-1, :], x_v[:,x_v_cnt-1:x_v_cnt,:]], 1)
    #x_v = np.fmax(0, np.fmin(1, x_v))
    #x_h = np.fmax(0, np.fmin(1, x_h))
    #val = np.fmax(0, np.fmin(1, val))
    
    # MATLABコード　COLORが3次元目にある
    #n1
    #result = cat(1, -z(1, :, :, 1), -z(2:n1-1, :, :, 1) + z(1:n1-2, :, :, 1), z(n1-1, :, :, 1)) 
    #       + cat(2, -z(:, 1, :, 2), -z(:, 2:n2-1, :, 2) + z(:, 1:n2-2, :, 2), z(:, n2-1, :, 2));

    return val

#return np.fmax(0, np.fmin(1, x))  # box constraint
#return np.sign(x) * np.fmax(0, np.abs(x) - lambda1 * gamma1) # prox of l1
