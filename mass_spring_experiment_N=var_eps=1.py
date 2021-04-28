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


    # Q = np.diag([10, 1])
    # Qf = np.diag([15, 1])
    # R = np.diag([1])
    #
    # d = model.d
    # mu = np.zeros([d, 1])
    # mu_w = np.vstack([1] + [mu] * N)
    # M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
    # beta = 0.95
    # N_sample = 3
    # i_th_state = 1
    # i_state_ub = 0.4
    # epsilon = 1
    # sin_const = 3
    # N_sim = 60

    # sim = Simulation(model, Q, Qf, R, mu, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    # sin_const = sin_const, N_sim=N_sim, mode = "gene")
    # print(sim.x_sim)

    Q = np.diag([10, 1])
    Qf = np.diag([15, 1])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    mu_w = np.vstack([1] + [mu] * N)
    M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
    beta = 0.95
    N_sample = 1
    i_th_state = 1
    i_state_ub = 0.4
    epsilon = 1
    sin_const = 3
    N_sim = 75
    N_loop = 100
    # Test2: fix esp = 1, sin_const = 3, beta = 0.95, i_ub = 0.4, N_loop = 1.
    # change N_sample
    N_sample_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for N_sample_temp in N_sample_test:
        result_x = []
        result_u = []
        N_sample = N_sample_temp
        for i in range(N_loop):
            sim = Simulation(model, Q, Qf, R, mu, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state,
                             i_state_ub = i_state_ub, epsilon = epsilon,sin_const = sin_const, N_sim=N_sim, mode = "gene")
            result_x += [sim.x_sim]
            result_u += [sim.u_sim]
            print("#" +str(i) + " sim of " + str(N_sample) + " is done")

        N_sample_str = str(N_sample)

        write_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/" + file_name + "/" + N_sample_str+ "_" + "x_tra" + ".txt"
        with open(write_path, 'w') as f:
            for listitem in result_x:
                f.write('%s\n' % listitem)
        f.close()
        write_path = "/Users/zhengangzhong/Dropbox/PhD/documents/paper_writing/CDC2021/result/" + file_name + "/" + N_sample_str + "_" + "u_tra" +".txt"
        with open(write_path, 'w') as f:
            for listitem in result_u:
                f.write('%s\n' % listitem)
        f.close()