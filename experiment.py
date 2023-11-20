from test import eval_restoration
import numpy as np
import matplotlib.pyplot as plt

def plot_graph():
    ## Thanks to https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
#    plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
#    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 0.8 # axis line width
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.loc"] = 'best' # edgeの色を変更
#    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
#    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    plt.rcParams["figure.figsize"] = [34,5.5]  
    
    # plot
    outputfileName = 'PSNR_comparisonA-1(blur)_nl0.01_446.jpeg.npy'
    y_1 = np.load('./ICASSP-result/' + outputfileName)
    y_2 = np.load('./ICASSP-result/PSNR_ours-A(blur)_nl0.01_446.jpeg.npy')
    every = 40 #200, 40
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[1::every]
    y_2 = y_2[1::every]
    fig = plt.figure()
    fig_1 = fig.add_subplot(141)
    myMarkersize = 14
    fig_1.plot(x, y_1, marker='o', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="DnCNN-PnP-FBS")
    fig_1.plot(x, y_2, marker='x', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="b", label="DnCNN-PnP-PDS")
    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel("PSNR [dB]")
    fig_1.set_title('Deblurring', pad=12)
    plt.grid(color="gainsboro")
    fig_1.legend(ncol=1, loc="best")

    # plot
    outputfileName = 'c_comparisonA-1(blur)_nl0.01_446.jpeg.npy'
    y_1 = np.load('./ICASSP-result/' + outputfileName)
    y_2 = np.load('./ICASSP-result/c_ours-A(blur)_nl0.01_446.jpeg.npy')
    every = 40 #200, 40
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[1::every]
    y_2 = y_2[1::every]
    fig_2 = fig.add_subplot(142)
    myMarkersize = 14
    fig_2.plot(x, y_1, marker='o', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="DnCNN-PnP-FBS")
    fig_2.plot(x, y_2, marker='x', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="b", label="DnCNN-PnP-PDS")
    fig_2.set_xlabel(r"iterations")
    fig_2.set_ylabel(r"$c_n$")
    fig_2.set_title('Deblurring', pad=12)
    plt.yscale('log')
    plt.grid(color="gainsboro")
    fig_2.legend(ncol=1, loc="best")
    
    # plot
    outputfileName = 'PSNR_comparisonA-1(random_sampling)_nl0.01_446.jpeg.npy'
    y_1 = np.load('./ICASSP-result/' + outputfileName)
    y_2 = np.load('./ICASSP-result/PSNR_ours-A(random_sampling)_nl0.01_446.jpeg.npy')
    every = 200 #200, 40
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[1::every]
    y_2 = y_2[1::every]
    fig_3 = fig.add_subplot(143)
    myMarkersize = 14
    fig_3.plot(x, y_1, marker='o', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="DnCNN-PnP-FBS")
    fig_3.plot(x, y_2, marker='x', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="b", label="DnCNN-PnP-PDS")
    fig_3.set_xlabel(r"iterations")
    fig_3.set_ylabel("PSNR [dB]")
    fig_3.set_title('Inpainting', pad=12)
    plt.grid(color="gainsboro")
    fig_3.legend(ncol=1, loc="best")
    
    # plot
    outputfileName = 'c_comparisonA-1(random_sampling)_nl0.01_446.jpeg.npy'
    y_1 = np.load('./ICASSP-result/' + outputfileName)
    y_2 = np.load('./ICASSP-result/c_ours-A(random_sampling)_nl0.01_446.jpeg.npy')
    every = 200 #200, 40
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[1::every]
    y_2 = y_2[1::every]
    fig_4 = fig.add_subplot(144)
    myMarkersize = 14
    fig_4.plot(x, y_1, marker='o', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="DnCNN-PnP-FBS")
    fig_4.plot(x, y_2, marker='x', markersize=myMarkersize, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="b", label="DnCNN-PnP-PDS")
    fig_4.set_xlabel(r"iterations")
    fig_4.set_ylabel(r"$c_n$")
    plt.yscale('log')
    fig_4.set_title('Inpainting', pad=12)
    plt.grid(color="gainsboro")
    fig_4.legend(ncol=1, loc="best")

    outputfileName = "general_result"
    # save
    fig.savefig('./ICASSP-result/' + outputfileName + '.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./ICASSP-result/' + outputfileName + '.eps', bbox_inches="tight", pad_inches=0.05)
    
if (__name__ == '__main__'):
    plot_graph()
#    psnr = eval_restoration(gaussian_nl=0.01, sp_nl=0, max_iter = 300, gamma1 = 0.99, gamma2 = 0.1, r=0.8, alpha_n = 0.9, alpha_s = 0, result_output=False, architecture='preDnCNN_nobn_nch_3_nlev_0.01', deg_op = 'blur', method = 'ours-A')
#    print(psnr, np.mean(psnr))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-FBS(blur)_nl0.02_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-PDS(random_sampling)_nl0.02_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(blur)_nl0.02_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(random_sampling)_nl0.005_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(random_sampling)_nl0.01_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(random_sampling)_nl0.02_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_comparisonA-2(blur)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-FBS(blur)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-PDS(blur)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_comparisonA-1(blur)_nl0.05_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(blur)_nl0.05_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_comparisonA-2(random_sampling)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-FBS(random_sampling)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_BM3D-PnP-PDS(random_sampling)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_comparisonA-1(random_sampling)_nl0.015_265.jpeg.npy', allow_pickle=True))
#    print(np.load('./ICASSP-result/RESULT_AND_PARAMS_ours-A(random_sampling)_nl0.015_265.jpeg.npy', allow_pickle=True))
