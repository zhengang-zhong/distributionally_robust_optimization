import numpy as np
import matplotlib.pyplot as plt

def plot_tex_setting():
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')  # für \text{..}
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['figure.subplot.bottom'] = .265
    plt.rcParams['figure.subplot.bottom'] = .265
    plt.rcParams['figure.subplot.left'] = .21
    plt.rcParams['figure.subplot.top'] = .995
    plt.rcParams['figure.subplot.right'] = .98

    plt.rcParams['figure.subplot.hspace'] = .5  # vertikaler Abstand
    plt.rcParams['figure.subplot.wspace'] = .5  # horizontaler Abstand
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 13

if __name__ == "__main__":

    N_loop = 50
    delta_t = 0.1
    i_state_ub = 0.4
    N_sim = 70

    x_tra_list1 = []
    file_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_collect_data/mass_spring_collect_data_x_tra_N_sample=1.txt"
    with open(file_path, "r") as f:
        i = 0
        for line in f:
            current_data = line[:-1]
            current_data = current_data[1:-1]
            current_data = current_data.split(',')
            current_data = [float(i) for i in current_data]
            x_tra_list1.append(current_data)
    f.close()
    x_tra1_array = np.array(x_tra_list1)
    x_tra1_75_per = np.percentile(x_tra1_array, 75,axis=0)
    x_tra1_25_per = np.percentile(x_tra1_array, 25, axis=0)
    x_tra1_ave = np.average(np.array(x_tra1_array), axis=0)

    x_tra_list3 = []
    file_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_collect_data/mass_spring_collect_data_x_tra_N_sample=3.txt"
    with open(file_path, "r") as f:
        i = 0
        for line in f:
            current_data = line[:-1]
            current_data = current_data[1:-1]
            current_data = current_data.split(',')
            current_data = [float(i) for i in current_data]
            x_tra_list3.append(current_data)
    f.close()
    x_tra3_array = np.array(x_tra_list3)
    x_tra3_75_per = np.percentile(x_tra3_array, 75,axis=0)
    x_tra3_25_per = np.percentile(x_tra3_array, 25, axis=0)
    x_tra3_ave = np.average(np.array(x_tra3_array), axis=0)


    x_tra_list5 = []
    file_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_collect_data/mass_spring_collect_data_x_tra_N_sample=5.txt"
    with open(file_path, "r") as f:
        i = 0
        for line in f:
            current_data = line[:-1]
            current_data = current_data[1:-1]
            current_data = current_data.split(',')
            current_data = [float(i) for i in current_data]
            x_tra_list5.append(current_data)
    f.close()
    x_tra5_array = np.array(x_tra_list5)
    x_tra5_75_per = np.percentile(x_tra5_array, 75,axis=0)
    x_tra5_25_per = np.percentile(x_tra5_array, 25, axis=0)
    x_tra5_ave = np.average(np.array(x_tra5_array), axis=0)




    Nt = np.shape(x_tra1_ave[::2])[0]
    t_plot = [delta_t * i for i in range(Nt)]

    plot_x_list = [x_tra1_ave[::2],  x_tra3_ave[::2],x_tra5_ave[::2]]
    plot_v_list = [x_tra1_ave[1::2],  x_tra3_ave[1::2], x_tra5_ave[1::2]]


    plot_tex_setting()
    fig = plt.figure(figsize=(9, 6))
    spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])

    legend = []
    ax1.plot(t_plot, plot_x_list[0], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=1$')
    ax1.plot(t_plot, plot_x_list[1], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=3$')
    ax1.plot(t_plot, plot_x_list[2], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=5$')
    ax1.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
    ax1.set_ylabel(r'{\fontsize{14}{8}\selectfont Displacement, }$ x_1 [m]$ ')
    ax1.fill_between(t_plot, x_tra1_25_per[0::2], x_tra1_75_per[0::2],alpha=0.2)
    ax1.fill_between(t_plot, x_tra3_25_per[0::2], x_tra3_75_per[0::2],alpha=0.2)
    ax1.fill_between(t_plot, x_tra5_25_per[0::2], x_tra5_75_per[0::2],alpha=0.2)
    ax1.legend(loc="lower right", bbox_to_anchor=(0.94, 0))
    ax1.set_xlim(0, N_sim * delta_t)

    ax2.plot(t_plot, plot_v_list[0], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=1$',)
    ax2.plot(t_plot, plot_v_list[1], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=3$',)
    ax2.plot(t_plot, plot_v_list[2], label=r'{\fontsize{14}{8}\selectfont} $N_{init}=5$',)
    ax2.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
    ax2.set_ylabel(r'{\fontsize{14}{8}\selectfont Velocity, }$ x_2 [m/s]$ ')
    ax2.hlines(i_state_ub, 0, N_sim * delta_t, zorder=10, color="k")
    ax2.set_xlim(0, N_sim * delta_t)
    ax2.fill_between(t_plot, x_tra1_25_per[1::2], x_tra1_75_per[1::2],alpha=0.2)
    ax2.fill_between(t_plot, x_tra3_25_per[1::2], x_tra3_75_per[1::2],alpha=0.2)
    ax2.fill_between(t_plot, x_tra5_25_per[1::2], x_tra5_75_per[1::2],alpha=0.2)


    fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot9.pdf",
                bbox_inches='tight')

    plt.show()