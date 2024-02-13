import cv2, datetime, glob, json, os, iteration
import numpy as np
import matplotlib.pyplot as plt
from operators import get_observation_operators
from utils.utils_image import save_imgs
from utils.utils_noise import add_salt_and_pepper_noise, add_gaussian_noise, apply_poisson_noise
from utils.utils_eval import eval_psnr, eval_ssim

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def test_all_images (max_iter = 1000, gaussian_nl = 0.01, sp_nl = 0, poisson_noise = False, poisson_alpha = 300, gamma1 = 0.99, gamma2 = 0.99, alpha_n = 0.95, alpha_s = 0.95, myLambda=1, result_output =False, m1=25, m2=1, method = 'ours-A', architecture = 'DnCNN_nobn_nch_3_nlev_0.01', r=0.8, deg_op = 'blur', ch = 3, gammaInADMMStep1=10, add_timestamp = True):
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
        results[index] = {'filename' : filename, 'c': c_evolution, 'PSNR_evolution' : psnr_evolution, 'SSIM_evolution' : ssim_evolution, 'GROUND_TRUTH': img_true, 'OBSERVATION' : img_obsrv, 'RESULT': img_sol, 'REMOVED_SPARSE': s_sol, 'PSNR' : psnr_evolution[-1], 'SSIM' : ssim_evolution[-1], 'CPU_time' : average_time, 'PSNR_observation' : psnr_obsrv, 'SSIM_observation' : ssim_obsrv}

        # =====================================
        # Save images
        # =====================================
        pictures = [img_true, img_obsrv, img_sol]
        path_saveimg_base = method + '_' + deg_op + '_' + str(gaussian_nl).ljust(5, '0') + '_(' + filename + ')'
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
    params = {'architecture':architecture, 'gamma1': gamma1, 'gamma2': gamma2, 'alpha_n': alpha_n, 'gaussian_nl':gaussian_nl, 'sp_nl':sp_nl, 'poisson-noise':poisson_noise, 'poisson_alpha':poisson_alpha, 'alpha_n':alpha_n, 'alpha_s':alpha_s, 'max_iter':max_iter, 'myLambda': myLambda, 'r':r,  'deg_op': deg_op, 'method':method, 'ch':ch, 'm1':m1, 'm2':m2, 'gammaInADMMStep1':gammaInADMMStep1}
    summary = {'Average_PSNR':np.mean(psnr), 'PSNR':psnr, 'Average_SSIM':np.mean(ssim), 'SSIM' : ssim, 'Average_time':np.average(cpu_time) , 'Cpu_time': cpu_time}
    datas = {'params' : params, 'results' : results, 'summary' : summary}
    np.save(path_result + '\DATA_' + path_saveimg_base, datas)

    print(timestamp_commandline + '  Average_PSNR:' + str(np.mean(psnr).round(3)) + '  Average_SSIM:' + str(np.mean(ssim).round(3)) + '    Algorithm:' +  method + '   Observation:' + deg_op + '   Gaussian noise level:' + str(gaussian_nl).ljust(5, '0'))

    return datas

def main():
    Gaussian_noise_level_list = [0.005]
    sparse_noise_level_list = [0.0]
    observation_operator_list = ['blur']
    method_list = ['ours-A', 'comparisonA-1']

    for Gaussian_noise_level in Gaussian_noise_level_list:
        for sparse_noise_level in sparse_noise_level_list:
            for obsevation_operator in observation_operator_list:
                for method in method_list:
                    # test_all_imagesには実験設定とパラメータに関する２つの引数のみを用意して、オブジェクトでそれを渡すようにする　実験が終了したときに出力するデータファイルにそれを保存する
                    # 実験ごとに、そのままcsvで読み込めるようなテキストファイルを出力するようにする
                    # アルゴリズム部分のリファクタリングを進める
                    # 実験タイプ(A,B,C)と手法番号(1, 2, 3)を指定すれば、デノイザー(DnCNN)やアルゴリズム(PnP-PDS, RED, ADMM etc.)を取得できるような関数を作る（これの内容もテキストに保存する）。ノイズレベルなどは実験タイプごとにリスト化して保存しておく
                    datas = test_all_images(gaussian_nl=Gaussian_noise_level, sp_nl=sparse_noise_level, poisson_noise=False, poisson_alpha = 0, max_iter = 12, gamma1 = 0.99, gamma2 = 0.99, r=1, alpha_n = 0.95, myLambda=1, architecture='DnCNN_nobn_nch_3_nlev_0.01', deg_op = obsevation_operator, method = method, ch = 3)

if (__name__ == '__main__'):
    main()
