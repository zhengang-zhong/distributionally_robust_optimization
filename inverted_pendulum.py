import numpy as np
from DRO_model import Model
from DRO_optproblem import Opt_problem
from DRO_simulation import Simulation
import mosek
import cvxpy as cp


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

    x_init = np.array([[0], [0], [-np.pi], [0]])
    C = np.diag([1e-2, 0, 0, 0])
    D = np.array([[1, 0, 0, 0]])
    # D = np.array([[1, 0, 0, 0], [0,0,1,0]])
    E = np.array([[1e-2, 0, 0, 0]])
    N = 5
    model = Model(A, B, C, D, E, x_init, N, delta_t)

    Q = np.diag([1000, 0, 200, 0])
    Qf = np.diag([1000, 0, 200, 0])
    R = np.diag([1])

    d = model.d
    mu = np.zeros([d, 1])
    beta = 0.95
    N_sample = 10
    i_th_state = 1
    i_state_ub = 6.2
    epsilon = 1
    sin_const = 2
    N_sim = 30

    sim = Simulation(model, Q, Qf, R, mu, x_init, beta = beta, N_sample = N_sample, i_th_state = i_th_state, i_state_ub = i_state_ub, epsilon = epsilon,
    sin_const = sin_const, N_sim=N_sim, mode = "gene")
    print(sim.x_sim)