import numpy as np
from DRO_model import Model
from DRO_simulation import Simulation
import sys
import os


def mass_string_ode(t, x, u):
    m = 2  # [kg]
    k1 = 3  # [N/m]
    k2 = 2  # [N/m]

    A = np.array([[0, 1], [-k2 / m, -k1 / m]])
    B = np.array([[0], [1 / m]])

    dot_x = A @ x + B @ u

    return dot_x

def gene_sample(N, d, N_sample, sin_const):
    # Generate data: const * sinx

    w_sample = []
    for i in range(N_sample * N):
        w_temp = sin_const * np.sin(np.random.randn(d, 1))
        w_sample += [w_temp]
    return w_sample

if __name__ == "__main__":
    file_name =(os.path.splitext(sys.argv[0])[0]).split("/")[-1]

    m = 2 #[kg]
    k1 = 3 # [N/m]
    k2 = 2 # [N/m]

    A = np.array([[0,1],[-k2/m, -k1/m]])
    B = np.array([[0],[1/m]])
    delta_t = 0.1

    x_init = np.array([[-2],[0]])


    Ck = np.array([[1e-3, 0],[0, 0]])
    D = np.array([[1, 0]])
    E = np.array([[0,1e-3]])
    N = 5

    model = Model(A, B, Ck, D, E, x_init, N, delta_t)


    Q = np.diag([10, 1])
    Qf = np.diag([15, 1])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    mu_w = np.vstack([1] + [mu] * N)
    M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
    beta = 0.95
    N_sample = 5
    i_th_state = 1
    i_state_ub = 0.4
    epsilon = 1
    sin_const = 3
    N_sim = 70
    N_loop = 50 # 50
    N_sample_max = 10


    data_set_init = gene_sample(N, d, N_sample, sin_const)



    # Test3: fix N_sample = 1, sin_const = 3, beta = 0.95, i_ub = 0.4, N_loop = 10, eps = 1
    # sample 100 trajectories with collected data
    result_x = []
    result_u = []
    for i in range(N_loop):
        data_set = data_set_init
        sim = Simulation(model, Q, Qf, R, mu, x_init, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                         i_state_ub=i_state_ub, epsilon=epsilon,
                         sin_const=sin_const, N_sim=N_sim, mode="collect", data_set=data_set, N_sample_max=N_sample_max)
        result_x += [sim.x_sim]
        result_u += [sim.u_sim]
        print("#" +str(i) + " sim " + "is done")

    write_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/" + file_name + "_" + "x_tra" + "_N_sample=" + str(N_sample) + ".txt"
    with open(write_path, 'w') as f:
        for listitem in result_x:
            f.write('%s\n' % listitem)
    f.close()
    write_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/" + file_name + "_" + "u_tra" + "_N_sample=" + str(N_sample) + ".txt"
    with open(write_path, 'w') as f:
        for listitem in result_u:
            f.write('%s\n' % listitem)
    f.close()



