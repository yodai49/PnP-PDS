from test import eval_restoration
import numpy as np
import matplotlib.pyplot as plt

def plot_graph():
    ## Thanks to https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 22 # 全体のフォントサイズが変更されます。
#    plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
#    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.loc"] = 'best' # edgeの色を変更
#    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
#    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    
    every = 25
    y_1 = np.load('./ICASSP-result/PSNR_comparisonA-1(blur)_nl0.01_079.jpeg.npy')
    y_2 = np.load('./ICASSP-result/PSNR_ours-A(blur)_nl0.01_079.jpeg.npy')
    #y_3 = np.load('./ICASSP-result/PSNR_comparisonC-2(random_sampling)_nl0.01_077.jpeg.npy')
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[::every]
    y_2 = y_2[::every]

    # plot
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    fig_1.plot(x, y_1, marker='o', markersize=14, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="PnP-FBS")
    fig_1.plot(x, y_2, marker='x', markersize=14, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="b", label="PnP-PDS")
    #fig_1.plot(x, y_3, marker='v', markersize=10, markevery = 250,  markeredgewidth=1., markeredgecolor='k', color="b", label="PnP-FBS")

    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel(r"PSNR[dB]")
#    plt.yscale('log')
    plt.grid(color="gainsboro")

    fig_1.legend(ncol=1, loc="best")

    # save
    fig.savefig('./ICASSP-result/test_1.png', bbox_inches="tight", pad_inches=0.05)
    fig.savefig('./ICASSP-result/test_1.eps', bbox_inches="tight", pad_inches=0.05)
    
if (__name__ == '__main__'):
    plot_graph()
#    psnr = eval_restoration(gaussian_nl=0.01, sp_nl=0, max_iter = 300, gamma1 = 0.99, gamma2 = 0.1, r=0.8, alpha_n = 0.9, alpha_s = 0, result_output=False, architecture='preDnCNN_nobn_nch_3_nlev_0.01', deg_op = 'blur', method = 'ours-A')
#    print(psnr, np.mean(psnr))
