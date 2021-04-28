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
    N_sample_range = 10
    delta_t = 0.1
    i_state_ub = 0.4
    N_sim = 75


    x_tra_list = []
    for i in range(N_sample_range):
        file_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_N=var_eps=1/" + str(
            i + 1) + "_x_tra.txt"
        x_tra_temp_list = []
        with open(file_path,"r") as f:
            i = 0
            for line in f:
                current_data = line[:-1]
                current_data = current_data[1:-1]
                current_data = current_data.split(',')
                current_data = [float(i) for i in current_data]
                x_tra_temp_list.append(current_data)
        f.close()
        x_tra_list += [x_tra_temp_list]

    vio_range = 4.0

    violate_list = []
    violate_ave_list = []
    violate_25_list = []
    violate_75_list = []
    for x_tra_temp_list in x_tra_list:
        violate_temp_list = []
        for x_tra_temp in x_tra_temp_list:
            traj_temp = x_tra_temp[1::2]
            N_sample = np.shape(traj_temp)[0]
            N_violate = 0

            for j in range(int(vio_range/delta_t)):
                value = traj_temp[j]
                if value > i_state_ub:
                    N_violate += 1

            violate_temp_list += [N_violate]
        # violate_list += [np.array(violate_temp_list)]
        # violate_ave_list += [np.average(violate_temp_list,axis=0)]
        violate_list += [violate_temp_list]
        violate_ave_list += [np.average(np.array(violate_temp_list),axis=0)]

    violate_perc = [i*100/(int(vio_range/delta_t)) for i in violate_ave_list]
    plot_tex_setting()
    fig = plt.figure(figsize=(5, 3))
    spec = fig.add_gridspec(nrows=1, ncols=1, height_ratios=[1])
    ax1 = fig.add_subplot(spec[0, 0])

    x_axis_range = range(1,11)

    # ax1.boxplot([np.array(list)*100/int(vio_range/delta_t) for list in violate_list])


    ax1.plot(x_axis_range, violate_perc)
    ax1.set_xlabel(r'{\fontsize{14}{8}\selectfont Number of Sample} ')
    ax1.set_ylabel(r'{\fontsize{14}{8}\selectfont Violation, }$\%$ ')
    ax1.set_xticks(np.arange(min(x_axis_range), max(x_axis_range) + 1, 1.0))
    ax1.set_xlim(min(x_axis_range), max(x_axis_range))
    fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot4.pdf",
              bbox_inches='tight')

    plt.show()