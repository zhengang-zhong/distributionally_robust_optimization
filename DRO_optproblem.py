import numpy as np
import cvxpy as cp
import mosek


class Opt_problem:
    def __init__(self, model, Q, Qf, R, mu, beta = 0.95, N_sample = 5, i_th_state = 1, i_state_ub = 0.05, epsilon = 1, sin_const = 1):

        N = model.N

        n = model.n
        m = model.m
        d = model.d
        r = model.r
        Nx = model.Nx
        Nu = model.Nu
        Ny = model.Ny
        Nw = model.Nw
        # Stack system matrices
        Ax = model.Ax
        Bx = model.Bx
        Cx = model.Cx
        Ay = model.Ay
        By = model.By
        Cy = model.Cy
        Ey = model.Ey
        Cx_tilde = model.Cx_tilde
        Cy_tilde = model.Cy_tilde
        Ey_tilde = model.Ey_tilde
        D_tilde = model.D_tilde

        self.Q = Q
        self.Qf = Qf
        self.R = R

        Jx, Ju, eigval, eigvec, H_cal_dec, H, H_new_matrix, H_new, loss_func = self.define_loss_func(n, m, d, r, Nu, Nw, Bx, Cx_tilde, N, Q, Qf, R, mu, beta, sin_const)
        # W_sample_matrix, W_sample_matrix_ext = self.disturbance_para(N, d, N_sample, sin_const)
        W_sample_matrix, W_sample_matrix_ext = self.gene_disturbance(N, d, N_sample, sin_const)
        lambda_var, gamma_matrix, si_var, constraint = self.define_constraint(W_sample_matrix, W_sample_matrix_ext, eigvec, H_cal_dec, H, H_new, n, d, Bx, Cx_tilde, Cy_tilde,
                          Ey_tilde, N, N_sample, sin_const, i_th_state, i_state_ub, epsilon)
        self.H_cal_dec = H_cal_dec
        self.W_sample_matrix = W_sample_matrix
        self.W_sample_matrix_ext = W_sample_matrix_ext

        self.obj = cp.Minimize(loss_func)
        self.prob = cp.Problem(self.obj, constraint)

    def gene_disturbance(self, N, d, N_sample, sin_const):
        # Generate data: const * sinx

        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N * d))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        W_sample_matrix_ext = np.vstack([np.ones([1, N_sample]), W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext

    def disturbance_para(self, N, d, N_sample, sin_const):
        W_sample_matrix = cp.Parameter((N * d, N_sample))
        W_sample_matrix_ext = cp.vstack([np.ones([1, N_sample]),W_sample_matrix])
        return W_sample_matrix, W_sample_matrix_ext

    def define_loss_func(self, n, m, d, r, Nu, Nw, Bx, Cx_tilde, N, Q, Qf, R, mu, beta, sin_const):
        # Define decision variables for POB affine constrol law

        sigma = 1  # var of normal distribution

        H_cal_dec = cp.Variable((Nu, 1))
        #     print(H_cal_dec)
        # \mathcal{H} matrix
        for i in range(N):
            H_col = cp.Variable((Nu - (i * m), r))
            if i > 0:
                H_col = cp.vstack([np.zeros([i * m, r]), H_col])
            #         print(H_col)
            H_cal_dec = cp.hstack([H_cal_dec, H_col])
        #     print(H_cal_dec)
        #     print(np.shape(H_cal_dec))
        # Define intermediate decision variables for objective function
        H = cp.Variable((Nu, Nw + 1))

        # Define loss function
        #     beta = 0.95
        Jx = np.zeros([(N + 1) * n, (N + 1) * n])
        for i in range(N):
            Jx[i * n: (i + 1) * n, i * n: (i + 1) * n] = beta ** i * Q
        Jx[N * n:, N * n:] = beta ** N * Qf

        Ju = np.zeros([N * m, N * m])
        for i in range(N):
            Ju[i * m: (i + 1) * m, i * m: (i + 1) * m] = beta ** i * R

        # This is only for var = 1 and mean = 0. Should be modified.
        mu_w = np.vstack([1] + [mu] * N)
        #     M_w = mu_w @ mu_w.T + np.diag([0] + [1] * N * d)
        M_w = np.diag([1] + [sin_const ** 2 * (1 - np.exp(-2 * sigma ** 2)) / 2] * N * d)

        # Intermediate decision variables. Since CVXPY does not support quadratic obj of decision variable matrix.
        H_new_matrix = []
        #     for i in range(Nw+1):
        #         H_new_matrix += [cp.Variable([Nu,1])]
        #     H_new = cp.hstack(H_new_matrix)
        for i in range(Nu):
            H_new_matrix += [cp.Variable([1, Nw + 1])]
        H_new = cp.vstack(H_new_matrix)

        #     print(H_new.shape)
        # Reformulate the quadratic term

        eigval, eigvec = np.linalg.eig(Ju + Bx.T @ Jx @ Bx)
        eigval_mat = np.diag(eigval)
        #     print(Ju + Bx.T @ Jx @ Bx - eigvec @ eigval_mat @ np.linalg.inv(eigvec))
        # Loss function
        loss_func = 0

        N_eig = np.shape(eigval)[0]
        I = np.diag([1] * (Nw + 1))
        for i in range(N_eig):
            # Reformulate Tr(H.T @ (Ju + Bx.T @ Jx @ Bx)@ H ) @ M_w
            #         print(np.shape(H_new_matrix[i].T))
            loss_func += eigval[i] * M_w[i, i] * cp.quad_form(H_new_matrix[i].T,
                                                              I)  # When M_w is identity matrix. Otherwise reformulate system matrix or this line
        loss_func += cp.trace(2 * Cx_tilde.T @ Jx @ Bx @ H @ M_w)
        loss_func += cp.trace(Cx_tilde.T @ Jx @ Cx_tilde @ M_w)
        # Reformulate mu_w.T @ (H.T @ (Ju + Bx.T @ Jx @ Bx)@ H ) @ mu_w
        loss_func += eigval[0] * cp.quad_form(H_new_matrix[0].T, I) + 2 * mu_w.T @ Cx_tilde.T @ Jx @ Bx @ H @ mu_w
        loss_func += mu_w.T @ Cx_tilde.T @ Jx @ Cx_tilde @ mu_w

        return Jx, Ju, eigval, eigvec, H_cal_dec, H, H_new_matrix, H_new, loss_func


    def define_constraint(self, W_sample_matrix, W_sample_matrix_ext, eigvec, H_cal_dec, H, H_new, n, d, Bx, Cx_tilde, Cy_tilde,
                          Ey_tilde, N, N_sample, sin_const, i_th_state, i_state_ub, epsilon):
        constraint = []
        constraint += [H_new == np.linalg.inv(eigvec) @ H]
        #     constraint += [H_new == eigvec.T @ H ]
        constraint += [H == H_cal_dec @ (Cy_tilde + Ey_tilde)]

        #     i_th_state = 1 # 0 for first element, 1 for second element
        #     i_state_ub = 0.05

        d_supp = np.vstack((sin_const * np.ones([N * d, 1]), sin_const * np.ones([N * d, 1])))
        C_supp = np.vstack((np.diag([1] * N * d), np.diag([-1] * N * d)))

        lambda_var = cp.Variable()

        gamma_shape = np.shape(d_supp)[0]
        gamma_matrix = []
        for i in range(N_sample):
            for j in range(N):
                gamma_var = cp.Variable([gamma_shape, 1])
                gamma_matrix += [gamma_var]
        # k in N, i in N_sample
        # bk + <ak,xi_i>
        X_constraint = (Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) + Cx_tilde) @ W_sample_matrix_ext
        si_var = cp.Variable([N_sample, 1])

        for i in range(N_sample):
            for j in range(N):
                #             print(N_sample)
                constraint_temp = X_constraint[n * (j + 1) + i_th_state, i] + gamma_matrix[i * N + j].T @ (
                            d_supp - C_supp @ W_sample_matrix[:, [i]])
                constraint += [constraint_temp <= si_var[i, 0]]

        ak_matrix = (Bx @ H_cal_dec @ (Cy_tilde + Ey_tilde) + Cx_tilde)[:, 1:]
        for i in range(N_sample):
            for j in range(N):
                constraint_temp = C_supp.T @ gamma_matrix[i * N + j] - ak_matrix[n * (j + 1) + i_th_state:n * (
                            j + 1) + i_th_state + 1, :].T
                constraint_temp = C_supp.T @ gamma_matrix[i * N + j] - ak_matrix[ [n * (j + 1) + i_th_state], :].T
                constraint += [cp.norm_inf(constraint_temp) <= lambda_var]

        for i in range(N_sample):
            for j in range(N):
                constraint += [gamma_matrix[i * N + j] >= 0]
        constraint += [lambda_var * epsilon + 1 / N_sample * cp.sum(si_var) <= i_state_ub]
        return lambda_var, gamma_matrix, si_var, constraint




