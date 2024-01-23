import matplotlib.pyplot as plt
import numpy as np

def plot_graph():
    ## Reference: https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 14 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 12 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
#    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
#    plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    
    every = 100
    y_1 = np.load('./result-PDS/PSNR_ours-B(random_sampling)_nl0.005_4.jpeg(20231223-151351).npy')
    y_2 = np.load('./result-PDS/PSNR_ours-B(random_sampling)_nl0.005_4.jpeg(20231223-151351).npy')
    y_3 = np.load('./result-PDS/PSNR_ours-B(random_sampling)_nl0.005_4.jpeg(20231223-151138).npy')
    y_4 = np.load('./result-PDS/PSNR_ours-B(random_sampling)_nl0.005_4.jpeg(20231223-150952).npy')
    x = np.linspace(0, y_1.size, (int)(y_1.size / every))
    y_1 = y_1[::every]
    y_2 = y_2[::every]
    y_3 = y_3[::every]
    y_4 = y_4[::every]

    # plot
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    fig_1.plot(x, y_1, marker='o', markersize=7, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label="Proposed")
    fig_1.plot(x, y_2, marker='x', markersize=7, markevery = 1,  markeredgewidth=1., markeredgecolor='k', color="b", label="PnP with BM3D")
    fig_1.plot(x, y_3, marker='v', markersize=7, markevery = 1,  markeredgewidth=1., markeredgecolor='k', color="g", label="ADMM with DnCNN")
    fig_1.plot(x, y_4, marker='.', markersize=7, markevery = 1,  markeredgewidth=1., markeredgecolor='k', color="m", label="RED with DnCNN")

    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel(r"PSNR")
#    plt.yscale('log')
#    plt.ylim(15, 27)
    plt.grid(color="gainsboro")

#    fig_1.legend(ncol=1, bbox_to_anchor=(0., 1.025, 1., 0.102), loc="lower right")
    fig_1.legend(ncol=1, loc="best")

    # save
    plt.show()
#    fig.savefig('./ICASSP-result/test.png', bbox_inches="tight", pad_inches=0.05)
#    fig.savefig('./ICASSP-result/test.eps', bbox_inches="tight", pad_inches=0.05)

    #print
#    params = np.load('./result-PDS/RESULT_AND_PARAMS_comparisonC-2(blur)_nl0.0_ll.png(20231204-193413).npy', allow_pickle=True)
#    print(params)


if (__name__ == '__main__'):
    plot_graph()