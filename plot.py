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
    
    plot_content = 'PSNR'
    every = 1
    plot_content_data = {'PSNR' : {'title' : 'PSNR', 'key' : 'PSNR_evolution'}, 
                        'c' :  {'title' : '$c_n$', 'key' : 'c_evolution'}}
 
    filename_list = ['./result/result-test/DATA_C-Proposed_blur_00000_(01.png)_20240215-210346-533908.npy']
    y_list = [None] * len(filename_list)
    data_list = [None] * len(filename_list)
    method_name_list = [None] * len(filename_list)
    for filename in filename_list:
        data_list[filename_list.index(filename)] = np.load(filename, allow_pickle=True).item()
        y_list[filename_list.index(filename)] = data_list[filename_list.index(filename)]['results'][0][plot_content_data[plot_content]['key']]
        method_name_list[filename_list.index(filename)] = str(data_list[filename_list.index(filename)]['summary']['algorithm']) + ' (' + str(data_list[filename_list.index(filename)]['summary']['denoiser']) + ')'
    x = np.linspace(0, y_list[0].size, (int)(y_list[0].size / every))
    for filename in filename_list:
        y_list[filename_list.index(filename)] = y_list[filename_list.index(filename)][::every]

    # plot
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    fig_1.set_xlabel(r"iterations")
    fig_1.set_ylabel(plot_content_data[plot_content]['title'])
    for y in y_list:
        fig_1.plot(x, y, marker='o', markersize=7, markevery = 1, markeredgewidth=1., markeredgecolor='k', color="r", label=method_name_list[y_list.index(y)])

    if (plot_content == 'c'):
        plt.yscale('log')
    plt.grid(color="gainsboro")

#    fig_1.legend(ncol=1, bbox_to_anchor=(0., 1.025, 1., 0.102), loc="lower right")
    fig_1.legend(ncol=1, loc="best")

    # save
    plt.show()
#    fig.savefig('./ICASSP-result/test.png', bbox_inches="tight", pad_inches=0.05)
#    fig.savefig('./ICASSP-result/test.eps', bbox_inches="tight", pad_inches=0.05)

if (__name__ == '__main__'):
    plot_graph()