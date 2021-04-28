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


N_loop = 10
delta_t = 0.1
i_state_ub = 0.4
N_sim = 70

x_tra_001 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_001_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_001.append(current_data)
f.close()



x_tra_01 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_01_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_01.append(current_data)
f.close()

x_tra_1 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_1_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_1.append(current_data)
f.close()

x_tra_3 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_3_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_3.append(current_data)
f.close()

x_tra_5 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_5_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_5.append(current_data)
f.close()

x_tra_10 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_10_x_tra.txt", "r") as f:
    i = 0
    for line in f:
        current_data = line[:-1]
        current_data = current_data[1:-1]
        current_data = current_data.split(',')
        current_data = [float(i) for i in current_data]
        x_tra_10.append(current_data)
f.close()

x_tra_100 =[]
with open("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/mass_spring_experiment_100_x_tra.txt", "r") as f:
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





Nt = np.shape(x_tra_001_ave_plot[::2])[0]
t_plot = [delta_t*i for i in range(Nt)]

plot_x = [x_tra_001_ave_plot[::2]]+[x_tra_01_ave_plot[::2]]+[x_tra_1_ave_plot[::2]]+[x_tra_3_ave_plot[::2]]+[x_tra_5_ave_plot[::2]]+[x_tra_10_ave_plot[::2]]+[x_tra_100_ave_plot[::2]]
plot_v = [x_tra_001_ave_plot[1::2]]+[x_tra_01_ave_plot[1::2]]+[x_tra_1_ave_plot[1::2]]+[x_tra_3_ave_plot[1::2]]+[x_tra_5_ave_plot[1::2]]+[x_tra_10_ave_plot[1::2]]+[x_tra_100_ave_plot[1::2]]
N_plot = len(plot_x)

plot_tex_setting()
fig = plt.figure(figsize=(11, 8))
spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0])

legend = []

ax1.plot(t_plot, plot_x[0], 'b', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 0.01$')
ax1.plot(t_plot, plot_x[1], 'g', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 0.1$')
ax1.plot(t_plot, plot_x[2], 'r', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 1$')
ax1.plot(t_plot, plot_x[3], 'c', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 3$')
ax1.plot(t_plot, plot_x[4], 'm', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 5$')
ax1.plot(t_plot, plot_x[5], 'y', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 10$')
ax1.plot(t_plot, plot_x[6], 'orange', label=r'{\fontsize{14}{8}\selectfont} $\varepsilon = 100$')
ax1.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax1.set_ylabel(r'{\fontsize{14}{8}\selectfont Displacement, }$ x_1 [m]$ ')

ax1.legend(loc="lower right",bbox_to_anchor=(0.94,0))

ax1.set_xlim(0, N_sim * delta_t)

ax2.plot(t_plot, plot_v[0], 'b')
ax2.plot(t_plot, plot_v[1], 'g')
ax2.plot(t_plot, plot_v[2], 'r')
ax2.plot(t_plot, plot_v[3], 'c')
ax2.plot(t_plot, plot_v[4], 'm')
ax2.plot(t_plot, plot_v[5], 'y')
ax2.plot(t_plot, plot_v[6], 'orange')
ax2.set_xlabel(r'{\fontsize{14}{8}\selectfont Time, }$ t [s]$ ')
ax2.set_ylabel(r'{\fontsize{14}{8}\selectfont Velocity, }$ x_2 [m/s]$ ')
ax2.hlines(i_state_ub,0,N_sim * delta_t, zorder=10, color = "k")
ax2.set_xlim(0, N_sim * delta_t)

fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot1.pdf",
          bbox_inches='tight')

plt.show()