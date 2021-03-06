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


vio_range = 1.0

violate_list_001 = []
for i in range (N_loop):
    traj_temp = x_tra_001[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_001 += [N_violate]
violate_ave_001 = np.average(np.array(violate_list_001),axis=0)

violate_list_01 = []
for i in range (N_loop):
    traj_temp = x_tra_01[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_01 += [N_violate]
violate_ave_01 = np.average(np.array(violate_list_01),axis=0)

violate_list_1 = []
for i in range (N_loop):
    traj_temp = x_tra_1[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_1 += [N_violate]
violate_ave_1 = np.average(np.array(violate_list_1),axis=0)

violate_list_3 = []
for i in range (N_loop):
    traj_temp = x_tra_3[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_3 += [N_violate]
violate_ave_3 = np.average(np.array(violate_list_3),axis=0)

violate_list_5 = []
for i in range (N_loop):
    traj_temp = x_tra_5[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_5 += [N_violate]
violate_ave_5 = np.average(np.array(violate_list_5),axis=0)

violate_list_10 = []
for i in range (N_loop):
    traj_temp = x_tra_10[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_10 += [N_violate]
violate_ave_10 = np.average(np.array(violate_list_10),axis=0)

violate_list_100 = []
for i in range (N_loop):
    traj_temp = x_tra_100[i][3::4]
    N_sample = np.shape(traj_temp)[0]

    N_violate = 0
    for j in range(int(vio_range/delta_t)):
        value = traj_temp[j]
        if value > i_state_ub:
            N_violate += 1
    violate_list_100 += [N_violate]
violate_ave_100 = np.average(np.array(violate_list_100),axis=0)


violate_perc = [violate_ave_001/int(vio_range/delta_t)*100, violate_ave_01/int(vio_range/delta_t)*100, violate_ave_1/int(vio_range/delta_t)*100, violate_ave_3/int(vio_range/delta_t)*100,
                violate_ave_5/int(vio_range/delta_t)*100,violate_ave_10/int(vio_range/delta_t)*100, violate_ave_100/int(vio_range/delta_t)*100]

plot_tex_setting()
fig = plt.figure(figsize=(5,3))
spec = fig.add_gridspec(nrows=1, ncols=1, height_ratios=[1])
ax1 = fig.add_subplot(spec[0, 0])



# x_axis_range = range(len(violate_perc))
x_axis_range = [0.01,0.1,1,3,5,10,100]
ax1.plot(x_axis_range, violate_perc)
# labels = ["1","0.01", "0.1", "1", "3", "5", "10", "100"]
# ax1.set_xticklabels(labels)
ax1.set_xlabel(r'{\fontsize{14}{8}\selectfont Radius, }$\varepsilon $ ')
ax1.set_ylabel(r'{\fontsize{14}{8}\selectfont Violation, }$\%$ ')
ax1.set_xscale('log')
fig.savefig("/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/plot7.pdf",
          bbox_inches='tight')

plt.show()