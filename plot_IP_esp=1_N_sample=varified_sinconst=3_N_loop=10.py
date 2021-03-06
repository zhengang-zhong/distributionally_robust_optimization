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

N_sample_range = 10
N_loop = 10
delta_t = 0.1
i_state_ub = 0.5
N_sim = 100

x_traj_list = []
for i in range(N_sample_range):
    x_traj_temp = []
    file_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_N=var_eps=1/" + str(i+1) + "_x_tra.txt"
    with open(file_path,"r") as f:
        i = 0
        for line in f:
            current_data = line[:-1]
            current_data = current_data[1:-1]
            current_data = current_data.split(',')
            current_data = [float(i) for i in current_data]
            x_traj_temp.append(current_data)
    f.close()
    x_traj_list += [x_traj_temp]

plot_traj_ave_list = []
for plot_traj in x_traj_list:
    plot_traj_ave_list += [np.average(np.array(plot_traj),axis=0)]

Nt = np.shape(plot_traj_ave_list[0][::4])[0]
t_plot = [delta_t*i for i in range(Nt)]

plot_x_list = []
plot_v_list = []
plot_ang_list = []
plot_ang_v_list = []
for plot in plot_traj_ave_list:
    plot_x_list += [plot[::4]]
    plot_v_list += [plot[1::4]]
    plot_ang_list += [plot[2::4]]
    plot_ang_v_list += [plot[3::4]]


N_plot = len(plot_x_list)

plot_tex_setting()
fig = plt.figure(figsize=(11, 8))
spec = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[2, 0])
ax4 = fig.add_subplot(spec[3, 0])

legend = []
for i in range(N_plot):
    ax1.plot(t_plot, plot_x_list[i], label=r'{\fontsize{14}{8}\selectfont} $N =' + str(i+1)+'$')

ax1.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax1.set_ylabel(r'{\fontsize{14}{8}\selectfont Displacement, }$ x_1 [m]$ ')
ax1.legend(loc="lower right",bbox_to_anchor=(0.94,0))
ax1.set_xlim(0, N_sim * delta_t)

for i in range(N_plot):
    ax2.plot(t_plot, plot_v_list[i], label=r'{\fontsize{14}{8}\selectfont} $N =' + str(i+1)+'$')
ax2.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax2.set_ylabel(r'{\fontsize{14}{8}\selectfont Velocity, }$ x_2 [m/s]$ ')
ax2.set_xlim(0, N_sim * delta_t)

for i in range(N_plot):
    ax3.plot(t_plot, plot_ang_list[i], label=r'{\fontsize{14}{8}\selectfont} $N =' + str(i+1)+'$')
ax3.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax3.set_ylabel(r'{\fontsize{14}{8}\selectfont Displacement, }$ x_3 [???]$ ')
ax3.set_xlim(0, N_sim * delta_t)

for i in range(N_plot):
    ax4.plot(t_plot, plot_ang_v_list[i], label=r'{\fontsize{14}{8}\selectfont} $N =' + str(i+1)+'$')
ax4.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax4.set_ylabel(r'{\fontsize{14}{8}\selectfont Velocity, }$ x_4 [1/s]$ ')
ax4.hlines(i_state_ub,0,N_sim * delta_t, zorder=10, color = "k")
ax4.set_xlim(0, N_sim * delta_t)


fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot6.pdf",
          bbox_inches='tight')

plt.show()