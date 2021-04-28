import numpy as np
from DRO_model import Model_nonlinear
from DRO_optproblem import Opt_problem
from DRO_simulation import Simulation_nonlinear
from scipy.linalg import expm
import mosek
import cvxpy as cp
import casadi as ca
import sys
import os


def inverted_pendulum_nonlinear_ode(x, u):
    M = 2.4
    m = 0.23
    l = 0.36
    g = 9.81

    dx1_dt = x[1]
    dx2_dt = (u * ca.cos(x[0]) - (M + m) * g * ca.sin(x[0]) + m * l * ca.cos(x[0]) * ca.sin(x[0]) * x[1] ** 2) / (
                m * l * ca.cos(x[0]) ** 2 - (M + m) * l)
    dx3_dt = x[3]
    dx4_dt = (u + m * l * ca.sin(x[0]) * x[1] ** 2 - m * g * ca.cos(x[0]) * ca.sin(x[0])) / (
                M + m - m * ca.cos(x[0]) ** 2)
    rhs = [dx1_dt,
           dx2_dt,
           dx3_dt,
           dx4_dt
           ]

    return ca.vertcat(*rhs)


if __name__ == "__main__":
    x_SX = ca.SX.sym("x_SX", 4)
    u_SX = ca.SX.sym("u_SX", 1)

    ode = ca.Function("ode_func", [x_SX, u_SX], [inverted_pendulum_nonlinear_ode(x_SX, u_SX)])
    delta_t = 0.5
    N = 5

    # x_init = np.array([[0], [0], [-np.pi/2], [0]])
    x_init = np.array([[-0.5], [0], [0], [0]])
    u_init =  np.array([[0]])
    # C = np.diag([0, 0, 0, 1e-2])
    # C = np.array([[0, 0],[0, 0],[1e-2, 0],[0, 0]])
    C = np.array([[0, 0],[1e-2, 0],[0, 0],[0, 0]])
    D = np.array([[1, 0,0 , 0]])
    # D = np.array([[1, 0, 0, 0], [0,0,1,0]])
    E = np.array([[0, 1e-2]])
    # E = np.array([[0, 0, 1e-2, 0]])
    N = 5
    model = Model_nonlinear(ode, C, D, E, x_init,u_init, N, delta_t)

    Q = np.diag([1500, 1, 1000, 1])
    Qf = np.diag([1500, 1, 1000, 1])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    sigma = 1
    beta = 0.95
    N_sample = 1
    i_th_state = 1
    i_state_ub = 0.2
    epsilon = 10

    sin_const = 3
    N_sim = 50
    sim = Simulation_nonlinear(model, Q, Qf, R, x_init, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                               i_state_ub=i_state_ub, epsilon=epsilon,
                               sin_const=sin_const, N_sim=N_sim, mode="gene", mu=mu, sigma=sigma, est=False)
    print(sim.x_sim)
    sim.plot_state()
