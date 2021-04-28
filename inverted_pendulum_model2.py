import numpy as np
# from DRO_model import Model_nonlinear
from DRO_model import Model
from DRO_optproblem import Opt_problem
# from DRO_simulation import Simulation_nonlinear
from DRO_simulation import Simulation
from scipy.linalg import expm
import mosek
import cvxpy as cp
import casadi as ca
import sys
import os

if __name__ == "__main__":
    M  = 1.096
    m = 0.109
    l = 0.25
    b = 0.1
    g = 9.81
    I = 0.0034

    p = I * (M + m) + M * m * l ** 2

    A = np.array([[0, 1, 0, 0],
                  [29.8615, 0, 0, 0],
                  [0, 0, 0, 1],
                  [-0.9401, 0, 0, 0]])
    B = np.array([[0], [-1.15741], [0], [0.416667]])

    delta_t = 0.1

    # x_init = np.array([[0], [0], [-np.pi/2], [0]])
    x_init = np.array([[-0.2], [0], [0], [0]])
    # C = np.diag([0, 0, 0, 1e-2])
    # C = np.array([[0, 0],[0, 0],[1e-2, 0],[0, 0]])
    C = np.array([[0, 0],[1e-2, 0],[0, 0],[0, 0]])
    D = np.array([[1, 0, 0, 0]])
    # D = np.array([[1, 0, 0, 0], [0,0,1,0]])
    E = np.array([[0, 1e-2]])
    # E = np.array([[0, 0, 1e-2, 0]])
    N = 5
    model = Model(A, B, C, D, E, x_init, N, delta_t)

    Q = np.diag([1500, 1, 1000, 1])
    Qf = np.diag([1500, 1, 1000, 1])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    sigma = 1
    beta = 0.95
    N_sample = 3
    i_th_state = 1
    i_state_ub = 0.5
    epsilon = 10

    sin_const = 3
    N_sim = 120
    sim = Simulation(model, Q, Qf, R, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    sin_const = sin_const, N_sim=N_sim, mode = "gene", mu = mu, sigma = sigma, est = False)
    print(sim.x_sim)
    sim.plot_state()