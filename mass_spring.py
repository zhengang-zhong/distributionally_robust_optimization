import numpy as np
from DRO_model import Model
from DRO_optproblem import Opt_problem
from DRO_simulation import Simulation
import mosek
import cvxpy as cp


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
    sigma = 1
    beta = 0.95
    N_sample = 1
    i_th_state = 1
    i_state_ub = 0.5
    epsilon = 1
    sin_const = 2
    N_sim = 60

    # Generate date, mu and sigma are given
    # sim = Simulation(model, Q, Qf, R, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    # sin_const = sin_const, N_sim=N_sim, mode = "gene", mu = mu, sigma = sigma, est = False)
    # print(sim.x_sim)

    # Generate date, mu and sigma are estimated from data
    sim = Simulation(model, Q, Qf, R, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    sin_const = sin_const, N_sim=N_sim, mode = "gene", mu = mu, sigma = sigma, est = True)
    print(sim.x_sim)

    # Collect date, mu and sigma are given
    # data_set = gene_sample(N, d, N_sample, sin_const)
    # print(data_set)
    # N_sample_max = 10
    # sim = Simulation(model, Q, Qf, R, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    # sin_const = sin_const, N_sim=N_sim, mode = "collect", data_set = data_set, N_sample_max = N_sample_max, mu = mu, sigma = sigma, est = False)
    # print(sim.x_sim)

    # Collect date, mu and sigma are estimated from data
    # data_set = gene_sample(N, d, N_sample, sin_const)
    # print(data_set)
    # N_sample_max = 10
    # sim = Simulation(model, Q, Qf, R, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    # sin_const = sin_const, N_sim=N_sim, mode = "collect", data_set = data_set, N_sample_max = N_sample_max, mu = mu, sigma = sigma, est = True)
    # print(sim.x_sim)

    sim.plot_state()

    # opt_problem = Opt_problem(model, Q, Qf, R, mu, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon, sin_const = sin_const)
    # W_sample, W_sample_ext =gene_disturbance(N, d, N_sample, sin_const)
    # opt_problem.W_sample_matrix.value = W_sample
    # prob = opt_problem.prob
    # prob.solve(solver = cp.MOSEK)
    # H_cal_dec = opt_problem.H_cal_dec
    # print(H_cal_dec.value)