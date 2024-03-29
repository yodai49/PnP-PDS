import cv2, datetime, glob, json, os, iteration
import numpy as np
import matplotlib.pyplot as plt
from operators import get_observation_operators
from utils.utils_image import save_imgs
from utils.utils_noise import add_salt_and_pepper_noise, add_gaussian_noise, apply_poisson_noise
from utils.utils_eval import eval_psnr, eval_ssim
from utils.utils_parse_args import *
from utils.utils_unparse_args import *
from utils.utils_method_master import get_algorithm_denoiser
from utils.utils_textfile import *

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def test_all_images (experimental_settings_arg = {}, method_arg = {}, configs_arg = {}):
    gaussian_nl, sp_nl, poisson_noise, poisson_alpha, deg_op, r  = parse_args_exp (experimental_settings_arg)
    method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1 = parse_args_method (method_arg)
    ch, add_timestamp, result_output = parse_args_configs (configs_arg)
    experimental_settings_all = unparse_args_exp (gaussian_nl, sp_nl, poisson_noise, poisson_alpha, deg_op, r)
    method_all = unparse_args_method (method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1)
    configs_all = unparse_args_configs (ch, add_timestamp, result_output)

    pattern_red = config['pattern_red']
    path_test = config['path_test']
    path_result = config['path_result']
    path_kernel = config['root_folder'] + 'blur_models/' + 'blur_1' + '.mat'
    path_prox = config['root_folder'] + 'nn/' + architecture + '.pth'
    path_images = sorted(glob.glob(os.path.join(path_test, pattern_red)))

    psnr = np.zeros((len(path_images)))
    ssim = np.zeros((len(path_images)))
    cpu_time = np.zeros((len(path_images)))
    results = {}

    for path_img in path_images:
        # =====================================
        # Prepare images and operators
        # =====================================
        index = path_images.index(path_img)
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        if(ch == 1):
            # Gray scale
            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        elif(ch == 3):
            # Color  (3 x H x W)
            img_true = np.moveaxis(img_true, -1, 0)
        phi, adj_phi = get_observation_operators(operator = deg_op, path_kernel = path_kernel, r = r)
        Id, _ = get_observation_operators("Id", path_kernel, r)
        img_obsrv = phi(img_true)
        if(deg_op == 'blur' or deg_op == 'Id'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, Id)
        elif (deg_op == 'random_sampling'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, phi)        
        if(poisson_noise):
            img_obsrv = apply_poisson_noise(img_obsrv, poisson_alpha)
        if(deg_op == 'blur' or deg_op == 'Id'):
            img_obsrv = add_salt_and_pepper_noise(img_obsrv, sp_nl, Id)
        elif (deg_op == 'random_sampling'):
            img_obsrv = add_salt_and_pepper_noise(img_obsrv, sp_nl, phi)        
        x_0 = np.copy(img_obsrv)
        if(poisson_noise):
            x_0 = x_0 / poisson_alpha
        
        # =====================================
        # Run evaluation
        # =====================================
        img_sol, s_sol, c_evolution, psnr_evolution, ssim_evolution, average_time = iteration.test_iter(x_0, img_obsrv, img_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, gaussian_nl, sp_nl, poisson_alpha, path_prox, max_iter, method, ch, r)
        if(poisson_noise):
            img_obsrv = img_obsrv / poisson_alpha

        # =====================================
        # Save results
        # =====================================
        filename = (path_img[path_img.rfind('\\'):])[1:]
        psnr[index] = psnr_evolution[-1]
        ssim[index] = ssim_evolution[-1]
        cpu_time[index] = average_time
        psnr_obsrv = eval_psnr(img_true, img_obsrv)
        ssim_obsrv = eval_ssim(img_true, img_obsrv)
        results[index] = {'filename' : filename, 'c_evolution': c_evolution, 'PSNR_evolution' : psnr_evolution, 'SSIM_evolution' : ssim_evolution, 'GROUND_TRUTH': img_true, 'OBSERVATION' : img_obsrv, 'RESULT': img_sol, 'REMOVED_SPARSE': s_sol, 'PSNR' : psnr_evolution[-1], 'SSIM' : ssim_evolution[-1], 'CPU_time' : average_time, 'PSNR_observation' : psnr_obsrv, 'SSIM_observation' : ssim_obsrv}

        # =====================================
        # Save images
        # =====================================
        pictures = [img_true, img_obsrv, img_sol]
        path_saveimg_base = method + '_' + deg_op + '_' + str(gaussian_nl).ljust(5, '0') + '_(' + filename + ')' + '_alpha' + str(alpha_n).ljust(5, '0') + '_lambda' + str(myLambda).ljust(5, '0')
        if (add_timestamp):
            path_saveimg_base = path_saveimg_base + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        path_pictures = [path_result + '\GROUND_TRUTH_' + path_saveimg_base,  path_result + '\OBSERVATION_' + path_saveimg_base, path_result + '\RESULT_' + path_saveimg_base]
        save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')

        # =====================================
        # Plot graphs if necessary
        # =====================================
        if(result_output):
            x = np.arange(0, max_iter, 1)
            plt.title('PSNR')
            plt.plot(x, psnr_evolution)
            # plt.gca().set_yscale('log')
            # plt.plot(x, c_evolution) 
            plt.xlabel('iteration')
            plt.ylabel('PSNR')
            plt.show()
        
        timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(timestamp_commandline + '  (' + str(index+1) + '/' + str(len(path_images)) + ') PSNR:' + str(psnr[index].round(3)).ljust(6, '0') + '    SSIM:' + str(ssim[index].round(3)).ljust(6, '0') +  '   ' + filename)


    # =====================================
    # Save all results
    # =====================================
    timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
#    params = {'architecture':architecture, 'gamma1': gamma1, 'gamma2': gamma2, 'alpha_n': alpha_n, 'gaussian_nl':gaussian_nl, 'sp_nl':sp_nl, 'poisson-noise':poisson_noise, 'poisson_alpha':poisson_alpha, 'alpha_n':alpha_n, 'alpha_s':alpha_s, 'max_iter':max_iter, 'myLambda': myLambda, 'r':r,  'deg_op': deg_op, 'method':method, 'ch':ch, 'm1':m1, 'm2':m2, 'gammaInADMMStep1':gammaInADMMStep1}
    algorithm, denoiser = get_algorithm_denoiser (method)
    summary = {'Average_PSNR':np.mean(psnr), 'PSNR':psnr, 'Average_SSIM':np.mean(ssim), 'SSIM' : ssim, 'Average_time':np.average(cpu_time) , 'Cpu_time': cpu_time, 'algorithm' : algorithm, 'denoiser' : denoiser}
    datas = {'experimental_settings' : experimental_settings_all, 'method' : method_all, 'configs' : configs_all, 'results' : results, 'summary' : summary}
    np.save(path_result + '\DATA_' + path_saveimg_base, datas)

    print(timestamp_commandline + '  Average_PSNR:' + str(np.mean(psnr).round(3)) + '  Average_SSIM:' + str(np.mean(ssim).round(3)) + '    Algorithm:' +  method + '   Observation:' + deg_op + '   Gaussian noise level:' + str(gaussian_nl).ljust(5, '0'))

    return datas

def main():
    experiment_data_list = []
    filepath = config['path_result'] + 'SUMMARY(' + str(datetime.datetime.now().strftime("%Y%m%d %H%M%S %f")) + ').txt'
    touch_textfile (filepath)

    noise_level_list = [50, 100, 200, 300, 400]
    obs_list = ['blur', 'random_sampling']
    method_list_G = ['C-Proposed', 'C-PnPADMM-DnCNN', 'C-RED-DnCNN', 'C-PnP-unstable-DnCNN']
    for nl in noise_level_list:
        for obs in obs_list:
            if (obs == 'blur'):
                max_iter = 1200
            elif (obs == 'random_sampling'):
                max_iter = 3000
            settings =  {'gaussian_nl' : 0, 'sp_nl' : 0, 'poisson_noise' : True, 'poisson_alpha' : nl, 'deg_op' : obs, 'r' : 0.8}
            configs = {'add_timestamp' : True, 'ch' : 1}
            for method_G in method_list_G:
                for i in range(0,10):
                    # C-Proposed blurもrandom_samplingも0～2まで
                    # ADMMも0～2まで
                    # REDは0～0.5まで
                    # Unstableは0～10まで
                    if (method_G == 'C-Proposed' or method_G == 'C-PnPADMM-DnCNN'):
                        myLambda = (i + 1) * 0.2
                    elif (method_G == 'C-RED-DnCNN'):
                        myLambda = (i + 1) * 0.05
                    elif (method_G == 'C-PnP-unstable-DnCNN'):
                        myLambda = (i + 1)
                    architecture = 'DnCNN_nobn_nch_1_nlev_0.01'
                    if (method_G == 'C-PnP-unstable-DnCNN'):
                        architecture = 'dncnn_15'
                    experiment_data_list.append ({'settings' : settings, 'method' : {'method' : method_G, 'architecture' : architecture, 'max_iter' : max_iter, 'gamma1' : 0.0005, 'gamma2' : 1999, 'myLambda' : myLambda, 'm1' : 25, 'm2' : 12, 'gammaInADMMStep1' : 1}, 'configs' : configs})


    for experiment_data in experiment_data_list:
        data = test_all_images(experiment_data['settings'], experiment_data['method'], experiment_data['configs'])
        write_textfile (filepath, data)
    add_footer_textfile (filepath, data)



if (__name__ == '__main__'):

    main()


### ファイル名に調整するパラメータの値を入れるようにする（上書きされるため）