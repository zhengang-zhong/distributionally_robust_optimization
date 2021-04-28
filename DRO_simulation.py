import numpy as np
from DRO_optproblem import Opt_problem
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt

class Simulation:
    '''
    @Arg

        mode: "collect" data and incorporate the collected data into constraint
              "gene" data at each time instant and use fixed number of data to solve opt problem


    '''

    def __init__(self, model, Q, Qf, R, x_init, beta = 0.95, N_sample = 10, i_th_state = 1, i_state_ub = 0.05, epsilon = 1,
                 sin_const = 1, N_sim=80, mode = "collect", est = False, data_set = None, mu = None, sigma = None, N_sample_max = None):

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.N_sample = N_sample
        self.i_th_state = i_th_state
        self.i_state_ub = i_state_ub
        self.epsilon = epsilon
        self.sin_const = sin_const
        self.model = model

        if mode is "collect":
            self.x_sim, self.u_sim = self.simulation_collect(x_init,N_sim, data_set, N_sample_max, est)
        elif mode is "gene":
            self.x_sim, self.u_sim = self.simulation_gene(x_init,N_sim, est)

        else:
            print("Error Mode")

    def simulation_collect(self,x_init,N_sim, data_set, N_sample_max, est):
        Ak = self.model.Ak
        Bk = self.model.Bk
        C = self.model.C
        D = self.model.D
        E = self.model.E
        N = self.model.N

        Q = self.Q
        Qf = self.Qf
        R = self.R
        mu = self.mu
        sigma = self.sigma
        beta = self.beta

        i_th_state = self.i_th_state
        i_state_ub = self.i_state_ub
        epsilon = self.epsilon
        sin_const = self.sin_const

        delta_t = self.model.delta_t
        d = self.model.d

        t0 = 0
        xk = x_init
        uk = 0
        t = t0
        h = delta_t

        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []

        N_sample = self.N_sample
        # N_sample_max
        for i in range(N_sim):
            if i % N == 0 and i > 0 and N_sample <= N_sample_max:
                N_sample += 1

            self.model.change_xinit(xk)
            opt_problem = Opt_problem(self.model, Q, Qf, R, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                                      i_state_ub=i_state_ub, epsilon=epsilon, sin_const=sin_const, collect = True, est= est, data_set = data_set, mu = mu, sigma = sigma)
            # W_sample, W_sample_ext = self.gene_disturbance(N, d, N_sample, sin_const)
            # opt_problem.W_sample_matrix.value = W_sample
            prob = opt_problem.prob
            #         print(W_sample_matrix)
            #     print( prob.solve(verbose=True))

            prob.solve(solver=cp.MOSEK)

            wk = sin_const * np.sin(np.random.randn(d, 1))
            data_set += [wk]
            uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ xk + E @ wk)
            u_list += uk.flatten().tolist()
            # print("current state and input", xk, uk)
            # x_kp1 = self.simulation_Euler(Ak, Bk, xk, uk)
            x_kp1 = Ak @ xk + Bk @ uk
            xk = x_kp1
            xk += C @ wk
            x_list += xk.flatten().tolist()
        return x_list, u_list


    def simulation_gene(self, x_init, N_sim, est):
        Ak = self.model.Ak
        Bk = self.model.Bk
        C = self.model.C
        D = self.model.D
        E = self.model.E
        N = self.model.N

        Q = self.Q
        Qf = self.Qf
        R = self.R
        mu = self.mu
        sigma = self.sigma
        beta = self.beta
        N_sample = self.N_sample
        i_th_state = self.i_th_state
        i_state_ub = self.i_state_ub
        epsilon = self.epsilon
        sin_const = self.sin_const

        delta_t = self.model.delta_t
        d = self.model.d

        t0 = 0
        xk = x_init
        uk = 0
        t = t0
        h = delta_t

        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []

        for i in range(N_sim):
            #     if i % N == 0:
            self.model.change_xinit(xk)

            opt_problem = Opt_problem(self.model, Q, Qf, R, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                                      i_state_ub=i_state_ub, epsilon=epsilon, sin_const=sin_const, mu = mu, sigma = sigma, est = est)
            # W_sample, W_sample_ext = self.gene_disturbance(N, d, N_sample, sin_const)
            # opt_problem.W_sample_matrix.value = W_sample
            prob = opt_problem.prob
            #         print(W_sample_matrix)
            #     print( prob.solve(verbose=True))

            prob.solve(solver=cp.MOSEK)
            # print("opt value:", prob.value)
            #     print( prob.solve(verbose=True))
            #         prob.solve(solver = cp.MOSEK,verbose = True, mosek_params = {mosek.dparam.basis_tol_s:1e-9, mosek.dparam.ana_sol_infeas_tol:0})
            #         print(Ax @ x_init +  Bx @ H.value @ W_sample_matrix_ext[:,0:1]  + Cx @ W_sample_matrix[:,0:1])
            #         print("status:", prob.status)
            # print("Controller", opt_problem.H_cal_dec.value[0,0], opt_problem.H_cal_dec.value[0,1])
            #         print("dual:", constraint[0].dual_value)
            #         print("gamma", gamma_matrix[0].value,  gamma_matrix[1].value,  gamma_matrix[2].value,  gamma_matrix[3].value)
            # print("lambda",opt_problem.lambda_var.value)
            #         print("lambda time epsilon",lambda_var.value * epsilon)
            # print("si",opt_problem.si_var.value)
            # print("si average",np.sum(opt_problem.si_var.value)/N_sample)
            # print("state_constraint", np.mean(opt_problem.si_var.value) + opt_problem.lambda_var.value * epsilon)
            # print("state",(self.model.Bx @ opt_problem.H_cal_dec.value @ (self.model.Cy_tilde + self.model.Ey_tilde) + self.model.Cx_tilde) @ opt_problem.W_sample_matrix_ext)
                    # print("disturbance data", W_sample_matrix)
            wk = sin_const * np.sin(np.random.randn(d, 1))
            uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ xk + E @ wk)
            u_list += uk.flatten().tolist()
            print("current state and input", xk, uk)
            # x_kp1 = self.simulation_Euler(Ak, Bk, xk, uk)
            x_kp1 = Ak @ xk + Bk @ uk
            # x_kp1 = self.RK4_np(self.inverted_pendulum_ode, xk, uk, t, h)
            xk = x_kp1
            xk += C @ wk
            x_list += xk.flatten().tolist()
        return x_list, u_list

    # def inverted_pendulum_ode(self, t, x, u): # this is for the controler design based on the linear model to control nonlinear system
    #     M = 2.4
    #     m = 0.23
    #     l = 0.36
    #     g = 9.81
    #
    #     x = x.flatten()
    #     u = u.flatten()
    #     # print("x and u",x,u)
    #     dx1_dt = x[1]
    #     dx2_dt = (u[0] * np.cos(x[0]) - (M + m) * g * np.sin(x[0]) + m * l * np.cos(x[0]) * np.sin(x[0]) * x[1] ** 2) / (
    #                 m * l * np.cos(x[0]) ** 2 - (M + m) * l)
    #     dx3_dt = x[3]
    #     dx4_dt = (u[0] + m * l * np.sin(x[0]) * x[1] ** 2 - m * g * np.cos(x[0]) * np.sin(x[0])) / (
    #                 M + m - m * np.cos(x[0]) ** 2)
    #     rhs = np.array([[dx1_dt],
    #            [dx2_dt],
    #            [dx3_dt],
    #            [dx4_dt]])
    #
    #     return rhs

    def gene_disturbance(self, N, d, N_sample, sin_const):
        # Generate data: const * sinx

        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N*d))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        W_sample_matrix_ext = np.vstack( [np.ones([1, N_sample]),W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext

    def simulation_Euler(self, Ak, Bk, x, u):
        '''
        Simulate with implicit Euler
        x[k+1] = (I - delta_t * A)^{-1} @ x[k] + (I - delta_t * A)^{-1} @ (delta_t * B) @ u[k]
        '''
        x_next = Ak @ x + Bk @ u

        return x_next

    def RK4_np(self, f, x, u, t, h):
        """
        Runge-Kutta 4th order solver using numpy array data type.

        Args:
            f: A function returning first order ODE in 2D numpy array (Nx x 1).
            x: Current value (list or numpy array).
            t: Current time.
            h: Step length.
        Returns:
            x_next: Vector of next value in 2D numpy array (Nx x 1)
        """
        # x = np.reshape(x, (np.shape(x)[0], -1))  # Reshape x to col vector in np 2D array
        k1 = f(t, x, u)
        k2 = f(t + h / 2, x + h / 2 * k1, u)
        k3 = f(t + h / 2, x + h / 2 * k2, u)
        k4 = f(t + h, x + h * k3, u)
        x = np.reshape(x, (np.shape(x)[0], -1))  # Reshape x to col vector in np 2D array
        x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        print("xnext",x_next)
        return x_next

    def plot_state(self):
        delta_t = self.model.delta_t
        n = self.model.n

        x_traj = self.x_sim

        Nt = np.shape(x_traj[::n])[0]
        t_plot = [delta_t * i for i in range(Nt)]

        plt.figure(1, figsize=(10, 20))
        plt.clf()
        for i in range (n):
            plt.subplot( str(n) + str(1) + str(i + 1) )
            plt.grid()
            x_traj_temp = x_traj[i::n]
            plt.plot(t_plot, x_traj_temp)
            plt.ylabel('x' + str(i + 1))

        plt.xlabel('t')
        plt.show()




class Simulation_nonlinear:
    '''
    @Arg

        mode: "collect" data and incorporate the collected data into constraint
              "gene" data at each time instant and use fixed number of data to solve opt problem


    '''

    def __init__(self, model, Q, Qf, R, x_init, beta = 0.95, N_sample = 10, i_th_state = 1, i_state_ub = 0.05, epsilon = 1,
                 sin_const = 1, N_sim=80, mode = "collect", est = False, data_set = None, mu = None, sigma = None, N_sample_max = None):

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.N_sample = N_sample
        self.i_th_state = i_th_state
        self.i_state_ub = i_state_ub
        self.epsilon = epsilon
        self.sin_const = sin_const
        self.model = model

        if mode is "collect":
            self.x_sim, self.u_sim = self.simulation_collect(x_init,N_sim, data_set, N_sample_max, est)
        elif mode is "gene":
            self.x_sim, self.u_sim = self.simulation_gene(x_init,N_sim, est)

        else:
            print("Error Mode")

    def simulation_collect(self,x_init,N_sim, data_set, N_sample_max, est):
        Ak = self.model.Ak
        Bk = self.model.Bk
        C = self.model.C
        D = self.model.D
        E = self.model.E
        N = self.model.N

        Q = self.Q
        Qf = self.Qf
        R = self.R
        mu = self.mu
        sigma = self.sigma
        beta = self.beta

        i_th_state = self.i_th_state
        i_state_ub = self.i_state_ub
        epsilon = self.epsilon
        sin_const = self.sin_const

        delta_t = self.model.delta_t
        d = self.model.d

        t0 = 0
        xk = x_init
        uk = 0
        t = t0
        h = delta_t

        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []

        N_sample = self.N_sample
        # N_sample_max
        for i in range(N_sim):
            if i % N == 0 and i > 0 and N_sample <= N_sample_max:
                N_sample += 1

            self.model.change_xinit(xk,uk)
            opt_problem = Opt_problem(self.model, Q, Qf, R, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                                      i_state_ub=i_state_ub, epsilon=epsilon, sin_const=sin_const, collect = True, est= est, data_set = data_set, mu = mu, sigma = sigma)
            # W_sample, W_sample_ext = self.gene_disturbance(N, d, N_sample, sin_const)
            # opt_problem.W_sample_matrix.value = W_sample
            prob = opt_problem.prob
            #         print(W_sample_matrix)
            #     print( prob.solve(verbose=True))

            prob.solve(solver=cp.MOSEK)

            wk = sin_const * np.sin(np.random.randn(d, 1))
            data_set += [wk]
            uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ xk + E @ wk)
            u_list += uk.flatten().tolist()

            # x_kp1 = self.simulation_Euler(Ak, Bk, xk, uk)
            x_kp1 = np.array(self.model.ode_disc_func(xk, uk))
            xk = x_kp1
            xk += C @ wk
            x_list += xk.flatten().tolist()
            print("current state and input", xk, uk)
        return x_list, u_list


    def simulation_gene(self, x_init, N_sim, est):
        Ak = self.model.Ak
        Bk = self.model.Bk
        C = self.model.C
        D = self.model.D
        E = self.model.E
        N = self.model.N

        Q = self.Q
        Qf = self.Qf
        R = self.R
        mu = self.mu
        sigma = self.sigma
        beta = self.beta
        N_sample = self.N_sample
        i_th_state = self.i_th_state
        i_state_ub = self.i_state_ub
        epsilon = self.epsilon
        sin_const = self.sin_const

        delta_t = self.model.delta_t
        d = self.model.d

        t0 = 0
        xk = x_init
        uk = 0
        t = t0
        h = delta_t

        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []

        for i in range(N_sim):
            #     if i % N == 0:
            self.model.change_xinit(xk,uk)
            print(self.model.A)
            opt_problem = Opt_problem(self.model, Q, Qf, R, beta=beta, N_sample=N_sample, i_th_state=i_th_state,
                                      i_state_ub=i_state_ub, epsilon=epsilon, sin_const=sin_const, mu = mu, sigma = sigma, est = est)
            # W_sample, W_sample_ext = self.gene_disturbance(N, d, N_sample, sin_const)
            # opt_problem.W_sample_matrix.value = W_sample
            prob = opt_problem.prob
            #         print(W_sample_matrix)
            #     print( prob.solve(verbose=True))

            prob.solve(solver=cp.MOSEK)
            # print("opt value:", prob.value)
            #     print( prob.solve(verbose=True))
            #         prob.solve(solver = cp.MOSEK,verbose = True, mosek_params = {mosek.dparam.basis_tol_s:1e-9, mosek.dparam.ana_sol_infeas_tol:0})
            #         print(Ax @ x_init +  Bx @ H.value @ W_sample_matrix_ext[:,0:1]  + Cx @ W_sample_matrix[:,0:1])
            #         print("status:", prob.status)
            # print("Controller", opt_problem.H_cal_dec.value[0,0], opt_problem.H_cal_dec.value[0,1])
            #         print("dual:", constraint[0].dual_value)
            #         print("gamma", gamma_matrix[0].value,  gamma_matrix[1].value,  gamma_matrix[2].value,  gamma_matrix[3].value)
            # print("lambda",opt_problem.lambda_var.value)
            #         print("lambda time epsilon",lambda_var.value * epsilon)
            # print("si",opt_problem.si_var.value)
            # print("si average",np.sum(opt_problem.si_var.value)/N_sample)
            # print("state_constraint", np.mean(opt_problem.si_var.value) + opt_problem.lambda_var.value * epsilon)
            # print("state",(self.model.Bx @ opt_problem.H_cal_dec.value @ (self.model.Cy_tilde + self.model.Ey_tilde) + self.model.Cx_tilde) @ opt_problem.W_sample_matrix_ext)
                    # print("disturbance data", W_sample_matrix)
            wk = sin_const * np.sin(np.random.randn(d, 1))
            uk = opt_problem.H_cal_dec.value[0, 0] + opt_problem.H_cal_dec.value[0, 1] * (D @ xk + E @ wk)
            u_list += uk.flatten().tolist()

            # x_kp1 = self.simulation_Euler(Ak, Bk, xk, uk)
            # x_kp1 = np.array(self.model.ode_disc_func(xk, uk))
            # A = np.array(self.model.jacobian_disc_x(xk, uk))
            # B = np.array(self.model.jacobian_disc_u(xk, uk))
            A, B = self.model.disc_nonlinear_system(xk, uk, delta_t)
            x_kp1 = A @ xk + B @ uk
            xk = x_kp1
            xk += C @ wk
            x_list += xk.flatten().tolist()
            print("current state and input", xk, uk)
        return x_list, u_list



    def gene_disturbance(self, N, d, N_sample, sin_const):
        # Generate data: const * sinx

        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N*d))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        W_sample_matrix_ext = np.vstack( [np.ones([1, N_sample]),W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext


    def RK4_np(self, f, x, u, t, h):
        """
        Runge-Kutta 4th order solver using numpy array data type.

        Args:
            f: A function returning first order ODE in 2D numpy array (Nx x 1).
            x: Current value (list or numpy array).
            t: Current time.
            h: Step length.
        Returns:
            x_next: Vector of next value in 2D numpy array (Nx x 1)
        """
        x = np.reshape(x, (np.shape(x)[0], -1))  # Reshape x to col vector in np 2D array
        k1 = f(t, x, u)
        k2 = f(t + h / 2, x + h / 2 * k1, u)
        k3 = f(t + h / 2, x + h / 2 * k2, u)
        k4 = f(t + h, x + h * k3, u)
        x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def plot_state(self):
        delta_t = self.model.delta_t
        n = self.model.n

        x_traj = self.x_sim

        Nt = np.shape(x_traj[::n])[0]
        t_plot = [delta_t * i for i in range(Nt)]

        plt.figure(1, figsize=(10, 20))
        plt.clf()
        for i in range (n):
            plt.subplot( str(n) + str(1) + str(i + 1) )
            plt.grid()
            x_traj_temp = x_traj[i::n]
            plt.plot(t_plot, x_traj_temp)
            plt.ylabel('x' + str(i + 1))

        plt.xlabel('t')
        plt.show()
