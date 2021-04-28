import numpy as np
from DRO_model import Model
from DRO_simulation import Simulation
import sys
import os



def inverted_pendulum_ode(t, x, u):
    M  = 1.096
    m = 0.109
    l = 0.25
    b = 0.1
    g = 9.81
    I = 0.0034

    p = I * (M + m) + M * m * l ** 2

    A = np.array([[0, 1, 0, 0],
                  [0, -(I+m*l**2)*b/p,  (m**2*g*l**2)/p,   0],
                  [0, 0, 0, 1],
                  [0, -(m*l*b)/p,       m*g*l*(M+m)/p,  0]])
    B = np.array([[0], [(I+m*l**2)/p], [0], [m*l/p]])

    dot_x = A @ x + B @ u

    return dot_x


if __name__ == "__main__":
    file_name =(os.path.splitext(sys.argv[0])[0]).split("/")[-1]

    M  = 1.096
    m = 0.109
    l = 0.25
    b = 0.1
    g = 9.81
    I = 0.0034

    p = I * (M + m) + M * m * l ** 2

    A = np.array([[0, 1, 0, 0],
                  [0, -(I+m*l**2)*b/p,  (m**2*g*l**2)/p,   0],
                  [0, 0, 0, 1],
                  [0, -(m*l*b)/p,       m*g*l*(M+m)/p,  0]])
    B = np.array([[0], [(I+m*l**2)/p], [0], [m*l/p]])

    delta_t = 0.1

    # x_init = np.array([[0], [0], [-np.pi/2], [0]])
    x_init = np.array([[0], [0], [-0.5], [0]])
    # C = np.diag([0, 0, 0, 1e-2])
    C = np.array([[0, 0],[0, 0],[0, 0],[1e-2, 0]])
    D = np.array([[0, 0, 1, 0]])
    # D = np.array([[1, 0, 0, 0], [0,0,1,0]])
    E = np.array([[0, 1e-2]])
    # E = np.array([[0, 0, 1e-2, 0]])
    N = 5
    model = Model(A, B, C, D, E, x_init, N, delta_t)

    Q = np.diag([1000, 1, 1500, 1])
    Qf = np.diag([1000, 1, 1500, 1])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    beta = 0.95
    N_sample = 1
    i_th_state = 3
    i_state_ub = 0.5
    epsilon = 1
    sin_const = 3
    N_sim = 100
    N_loop = 10

    # Test2: fix esp = 1, sin_const = 3, beta = 0.95, i_ub = 0.5, N_loop = 1.
    # change N_sample
    # N_sample_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_sample_test = [3, 4, 5, 6, 7, 8, 9, 10]
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