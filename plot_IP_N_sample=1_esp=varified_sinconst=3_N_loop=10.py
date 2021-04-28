import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


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


N_loop = 10
delta_t = 0.1
i_state_ub = 0.5
N_sim = 100

x_tra_001 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/001_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_001.append(current_data)
f.close()



x_tra_01 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/01_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_01.append(current_data)
f.close()

x_tra_1 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/1_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_1.append(current_data)
f.close()

x_tra_3 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/3_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_3.append(current_data)
f.close()

x_tra_5 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/5_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_5.append(current_data)
f.close()

x_tra_10 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/10_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_10.append(current_data)
f.close()

x_tra_100 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/inverted_pendulum_experiment_eps=var_N_sample=1/100_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_100.append(current_data)
f.close()


x_tra_001_ave_plot = np.average(np.array(x_tra_001),axis=0)
x_tra_01_ave_plot = np.average(np.array(x_tra_01),axis=0)
x_tra_1_ave_plot = np.average(np.array(x_tra_1),axis=0)
x_tra_3_ave_plot = np.average(np.array(x_tra_3),axis=0)
x_tra_5_ave_plot = np.average(np.array(x_tra_5),axis=0)
x_tra_10_ave_plot = np.average(np.array(x_tra_10),axis=0)
x_tra_100_ave_plot = np.average(np.array(x_tra_100),axis=0)





Nt = np.shape(x_tra_001_ave_plot[::4])[0]
t_plot = [delta_t*i for i in range(Nt)]

plot_x = [x_tra_001_ave_plot[::4]]+[x_tra_01_ave_plot[::4]]+[x_tra_1_ave_plot[::4]]+[x_tra_3_ave_plot[::4]]+[x_tra_5_ave_plot[::4]]+[x_tra_10_ave_plot[::4]]+[x_tra_100_ave_plot[::4]]
plot_v = [x_tra_001_ave_plot[1::4]]+[x_tra_01_ave_plot[1::4]]+[x_tra_1_ave_plot[1::4]]+[x_tra_3_ave_plot[1::4]]+[x_tra_5_ave_plot[1::4]]+[x_tra_10_ave_plot[1::4]]+[x_tra_100_ave_plot[1::4]]
plot_ang = [x_tra_001_ave_plot[2::4]]+[x_tra_01_ave_plot[2::4]]+[x_tra_1_ave_plot[2::4]]+[x_tra_3_ave_plot[2::4]]+[x_tra_5_ave_plot[2::4]]+[x_tra_10_ave_plot[2::4]]+[x_tra_100_ave_plot[2::4]]
plot_ang_v = [x_tra_001_ave_plot[3::4]]+[x_tra_01_ave_plot[3::4]]+[x_tra_1_ave_plot[3::4]]+[x_tra_3_ave_plot[3::4]]+[x_tra_5_ave_plot[3::4]]+[x_tra_10_ave_plot[3::4]]+[x_tra_100_ave_plot[3::4]]
N_plot = len(plot_x)

plot_tex_setting()
fig = plt.figure(figsize=(11, 8))
spec = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[2, 0])
ax4 = fig.add_subplot(spec[3, 0])

legend = []

N_line = 7
colors = pl.cm.inferno(np.linspace(0.3,0.8,N_line))

ax1.plot(t_plot, plot_x[0], 'b', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 0.01$', color=colors[6])
ax1.plot(t_plot, plot_x[1], 'g', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 0.1$', color=colors[5])
ax1.plot(t_plot, plot_x[2], 'r', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 1$', color=colors[4])
ax1.plot(t_plot, plot_x[3], 'c', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 3$',  color=colors[3])
ax1.plot(t_plot, plot_x[4], 'm', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 5$',  color=colors[2])
ax1.plot(t_plot, plot_x[5], 'y', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 10$',  color=colors[1])
ax1.plot(t_plot, plot_x[6], 'orange', label=r'{\fontsize{12}{8}\selectfont} $\varepsilon = 100$',  color=colors[0])
# ax1.set_xlabel(r'{\fontsize{12}{8}\selectfont Time, }$ t [s]$ ')
ax1.set_ylabel(r'{\fontsize{12}{8}\selectfont Displacement, }$ x_1 [m]$ ')

# ax1.legend(loc="upper right",bbox_to_anchor=(0.94,-0.8),ncol=2)
ax1.legend(loc="upper right",ncol=3)
ax1.set_xlim(0, N_sim * delta_t)


for i in range(7):
    ax2.plot(t_plot, plot_v[i], color=colors[N_line-1-i])
    ax3.plot(t_plot, plot_ang[i], color=colors[N_line-1-i])
    ax4.plot(t_plot, plot_ang[i], color=colors[N_line-1-i])
# ax2.plot(t_plot, plot_v[0], 'b')
# ax2.plot(t_plot, plot_v[1], 'g')
# ax2.plot(t_plot, plot_v[2], 'r')
# ax2.plot(t_plot, plot_v[3], 'c')
# ax2.plot(t_plot, plot_v[4], 'm')
# ax2.plot(t_plot, plot_v[5], 'y')
# ax2.plot(t_plot, plot_v[6], 'orange')
# ax2.set_xlabel(r'{\fontsize{12}{8}\selectfont Time, }$ t [s]$ ')
ax2.set_ylabel(r'{\fontsize{12}{8}\selectfont Velocity, }$ x_2 [m/s]$ ')
ax2.set_xlim(0, N_sim * delta_t)

# ax3.plot(t_plot, plot_ang[0], 'b')
# ax3.plot(t_plot, plot_ang[1], 'g')
# ax3.plot(t_plot, plot_ang[2], 'r')
# ax3.plot(t_plot, plot_ang[3], 'c')
# ax3.plot(t_plot, plot_ang[4], 'm')
# ax3.plot(t_plot, plot_ang[5], 'y')
# ax3.plot(t_plot, plot_ang[6], 'orange')
# ax3.set_xlabel(r'{\fontsize{12}{8}\selectfont Time, }$ t [s]$ ')
ax3.set_ylabel(r'{\fontsize{12}{8}\selectfont Rotation, }$ x_3 [rad]$ ')
ax3.set_xlim(0, N_sim * delta_t)


# ax4.plot(t_plot, plot_ang_v[0], 'b')
# ax4.plot(t_plot, plot_ang_v[1], 'g')
# ax4.plot(t_plot, plot_ang_v[2], 'r')
# ax4.plot(t_plot, plot_ang_v[3], 'c')
# ax4.plot(t_plot, plot_ang_v[4], 'm')
# ax4.plot(t_plot, plot_ang_v[5], 'y')
# ax4.plot(t_plot, plot_ang_v[6], 'orange')
ax4.set_xlabel(r'{\fontsize{12}{8}\selectfont Time, }$ t [s]$ ')
ax4.set_ylabel(r'{\fontsize{12}{8}\selectfont Angular Velocity, }$ x_4 [1/s]$ ')

ax4.hlines(i_state_ub,0,N_sim * delta_t, zorder=10, color = "k")
ax4.set_xlim(0, N_sim * delta_t)





fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot5.pdf",
          bbox_inches='tight')

plt.show()