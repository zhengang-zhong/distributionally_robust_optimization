import numpy as np
import casadi as ca
from numpy.linalg import matrix_power
from scipy.linalg import expm
class Model:
    '''
    This is for linear systems


    '''

    def __init__(self, A, B, C, D, E, x_init, N, delta_t):
        '''

        x_next = Ak x_k + Bk u_k + Ck w_k
        y_k = D x_k + E w_k

        Args:
            A: Continuous time
            B: Continuous time
            C: Discrete time
            D: Discrete time
            E: Discrete time

        '''

        self.A = A
        self.B = B

        self.delta_t = delta_t
        self.N = N

        self.n = np.shape(A)[1]  # Dimension of state
        self.m = np.shape(B)[1]  # Dimension of input

        Ak, Bk = self.disc_linear_system(A, B, delta_t)
        self.Ak = Ak
        self.Bk = Bk
        self.C = C
        self.D = D
        self.E = E

        n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde  = self.stack_system(Ak, Bk, C, D, E, x_init, N)
        # Define relevant dimension
        self.n = n
        self.m = m
        self.d = d
        self.r = r
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Nw = Nw
        # Stack system matrices
        self.Ax = Ax
        self.Bx = Bx
        self.Cx = Cx
        self.Ay = Ay
        self.By = By
        self.Cy = Cy
        self.Ey = Ey
        self.Cx_tilde = Cx_tilde
        self.Cy_tilde = Cy_tilde
        self.Ey_tilde = Ey_tilde
        self.D_tilde = D_tilde


    def disc_linear_system(self, A, B, delta_t):
        '''
        Discrete a linear system with implicit Euler
        x[k+1] = (I - delta_t * A)^{-1} @ x[k] + (I - delta_t * A)^{-1} @ (delta_t * B) @ u[k]

        Returns:
            Ak
            Bk

        '''
        Nx = np.shape(A)[0]
        Ix = np.identity(Nx)

        Ak = np.linalg.inv(Ix - delta_t * A)
        Bk = np.linalg.inv(Ix - delta_t * A) @ (delta_t * B)

        return Ak, Bk
    # def disc_linear_system(self, A, B, delta_t):
    #     '''
    #     '''
    #
    #     Ak = expm(A*delta_t)
    #     Bk = (expm(A * delta_t) - np.identity(self.n)) @ np.linalg.inv(A) @ B
    #
    #     return Ak, Bk

    def stack_system(self, A, B, C, D, E, x_init, N):
        '''
        Stack system matrix for N prediction horizon

        x_next = A x_k + B u_k + C w_k
        y_k = D x_k + E w_k

        '''

        n = np.shape(A)[1]  # Dimension of state
        m = np.shape(B)[1]  # Dimension of input
        d = np.shape(C)[1]  # Dimension of disturbance
        r = np.shape(D)[0]  # Dimension of output

        #     print(Nx,Nu,Nw,Ny)
        Nx = (N + 1) * n
        Nu = N * m
        Ny = N * r
        Nw = N * d

        Ax = np.zeros([Nx, n])
        Bx = np.zeros([Nx, Nu])
        Cx = np.zeros([Nx, Nw])
        Ay = np.zeros([Ny, n])
        By = np.zeros([Ny, Nu])
        Cy = np.zeros([Ny, Nw])
        Ey = np.zeros([Ny, Nw])
        Cx_tilde = np.zeros([Nx, Nw + 1])
        Cy_tilde = np.zeros([Ny + 1, Nw + 1])
        Ey_tilde = np.zeros([Ny + 1, Nw + 1])
        D_tilde = np.zeros([Ny + 1, Nx])
        #     H

        # Ax
        for i in range(N + 1):
            Ax[i * n:(i + 1) * n, :] = matrix_power(A, i)
        # Bx
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                Bx[(i + 1) * n: (i + 2) * n, (i - j) * m: (i - j + 1) * m] = mat_temp
                mat_temp = A @ mat_temp
        # Cx
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cx[(i + 1) * n: (i + 2) * n, (i - j) * d: (i - j + 1) * d] = mat_temp
                mat_temp = A @ mat_temp
        # Ay
        for i in range(N):
            Ay[i * r:(i + 1) * r, :] = D @ matrix_power(A, i)
            # By
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                By[(i + 1) * r: (i + 2) * r, (i - j) * m: (i - j + 1) * m] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Cy
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cy[(i + 1) * r: (i + 2) * r, (i - j) * d: (i - j + 1) * d] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Ey
        for i in range(N):
            Ey[i * r: (i + 1) * r, i * d: (i + 1) * d] = E
        # Cx_tilde
        Cx_tilde[:, [0]] = Ax @ x_init
        Cx_tilde[:, 1:] = Cx
        # Cy_tilde
        Cy_tilde[r:, [0]] = Ay @ x_init
        Cy_tilde[r:, 1:] = Cy
        # Ey_tilde
        Ey_tilde[0, 0] = 1
        Ey_tilde[1:, 1:] = Ey
        # D_tilde
        for i in range(N):
            D_tilde[1 + i * r: 1 + (i + 1) * r, i * n: (i + 1) * n] = D

        return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde



    def change_xinit(self, x_k):
        '''
        x_next = A x_k + B u_k + C w_k
        y_k = D x_k + E w_k
        '''

        n = self.n
        m = self.m
        d = self.d
        r = self.r

        #     print(Nx,Nu,Nw,Ny)
        Nx = self.Nx
        Nu = self.Nu
        Ny = self.Ny
        Nw = self.Nw

        N = self.N

        A = self.Ak
        B = self.Bk
        C = self.C
        D = self.D
        E = self.E


        Ax = self.Ax
        Bx = self.Bx
        Cx = self.Cx
        Ay = self.Ay
        By = self.By
        Cy = self.Cy
        Ey = self.Ey

        Cx_tilde = self.Cx_tilde
        Cy_tilde = self.Cy_tilde
        Ey_tilde = self.Ey_tilde
        D_tilde = self.D_tilde

        # Cx_tilde
        Cx_tilde[:, [0]] = Ax @ x_k
        # Cx_tilde[:, 1:] = Cx
        # Cy_tilde
        Cy_tilde[r:, [0]] = Ay @ x_k
        # Cy_tilde[r:, 1:] = Cy

        # Stack system matrices

        self.Cx_tilde = Cx_tilde
        self.Cy_tilde = Cy_tilde

        # return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde




class Model_nonlinear:
    '''
    This is for nonlinear systems

    '''

    def __init__(self, ode, C, D, E, x_init,u_init, N, delta_t):
        '''

        Args:
            ode: the continuous-time system dynamics in CASADI
        '''

        self.ode_func = ode
        n = ode.size1_in(0)
        m = ode.size1_in(1)

        x_SX = ca.SX.sym("x_SX", n)
        u_SX = ca.SX.sym("u_SX", m)

        self.delta_t = delta_t
        self.N = N
        self.n = n
        self.m = m

        self.jacobian_ode_x = ca.Function('jacobian_ode_x', [x_SX, u_SX],
                                           [ca.jacobian(ode(x_SX, u_SX), x_SX)])

        self.jacobian_ode_u = ca.Function('jacobian_ode_u', [x_SX, u_SX],
                                       [ca.jacobian(ode(x_SX, u_SX), u_SX)])

        self.linear_ode_func = ca.Function('linear_ode_func', [x_SX, u_SX],
                                       [self.jacobian_ode_x(x_SX, u_SX) @ x_SX +   self.jacobian_ode_u(x_SX, u_SX) @ u_SX])

        Ak, Bk = self.disc_nonlinear_system(x_init, u_init, delta_t)

        # Ak, Bk = self.disc_linear_system(self.jacobian_ode_x(x_init,u_init), self.jacobian_ode_u(x_init,u_init), delta_t)
        # A = self.jacobian_ode_x(x_init, u_init)
        # B = self.jacobian_ode_u(x_init, u_init)
        #
        # Ak = ca.expm(A*delta_t)
        # Bk = ca.expm(A * delta_t - ca.SX.eye(Nx)) @ B
        # self.jacobian_disc_x = ca.Function('jacobian_disc_x', [x_SX, u_SX],[Ak])
        # self.jacobian_disc_u = ca.Function('jacobian_disc_u', [x_SX, u_SX], [Bk])
        # self.ode_disc_func = ca.Function("ode_disc_func",[x_SX,u_SX], [Ak @ x_SX + Bk @ u_SX])

        # ode_disc = self.integrator_rk4(self.linear_ode_func , x_SX, u_SX, delta_t)
        # ode_disc_func = ca.Function("ode_disc_func",[x_SX,u_SX], [ode_disc])
        # self.ode_disc = ode_disc
        # self.ode_disc_func = ode_disc_func
        # self.jacobian_disc_x = ca.Function('jacobian_disc_x', [x_SX, u_SX],
        #                                    [ca.jacobian(self.ode_disc_func(x_SX, u_SX), x_SX)])
        #
        # self.jacobian_disc_u = ca.Function('jacobian_disc_u', [x_SX, u_SX],
        #                                [ca.jacobian(self.ode_disc_func(x_SX, u_SX), u_SX)])




        # ode_disc = self.integrator_rk4(ode, x_SX, u_SX, delta_t)
        # #
        # ode_disc_func = ca.Function("ode_disc_func",[x_SX,u_SX], [ode_disc])
        # self.ode_disc = ode_disc
        # self.ode_disc_func = ode_disc_func
        # #
        # self.jacobian_disc_x = ca.Function('jacobian_disc_x', [x_SX, u_SX],
        #                                    [ca.jacobian(self.ode_disc_func(x_SX, u_SX), x_SX)])
        # #
        # self.jacobian_disc_u = ca.Function('jacobian_disc_u', [x_SX, u_SX],
        #                                [ca.jacobian(self.ode_disc_func(x_SX, u_SX), u_SX)])
        #
        # Ak = np.array(self.jacobian_disc_x(x_init,u_init))
        # Bk = np.array(self.jacobian_disc_u(x_init, u_init))

        self.Ak = Ak
        self.Bk = Bk
        self.C = C
        self.D = D
        self.E = E

        n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde  = self.stack_system(Ak, Bk, C, D, E, x_init, N)
        # Define relevant dimension
        self.n = n
        self.m = m
        self.d = d
        self.r = r
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.Nw = Nw
        # Stack system matrices
        self.Ax = Ax
        self.Bx = Bx
        self.Cx = Cx
        self.Ay = Ay
        self.By = By
        self.Cy = Cy
        self.Ey = Ey
        self.Cx_tilde = Cx_tilde
        self.Cy_tilde = Cy_tilde
        self.Ey_tilde = Ey_tilde
        self.D_tilde = D_tilde


    def stack_system(self, A, B, C, D, E, x_init, N):
        '''
        Stack system matrix for N prediction horizon

        x_next = A x_k + B u_k + C w_k
        y_k = D x_k + E w_k

        '''

        n = np.shape(A)[1]  # Dimension of state
        m = np.shape(B)[1]  # Dimension of input
        d = np.shape(C)[1]  # Dimension of disturbance
        r = np.shape(D)[0]  # Dimension of output

        #     print(Nx,Nu,Nw,Ny)
        Nx = (N + 1) * n
        Nu = N * m
        Ny = N * r
        Nw = N * d

        Ax = np.zeros([Nx, n])
        Bx = np.zeros([Nx, Nu])
        Cx = np.zeros([Nx, Nw])
        Ay = np.zeros([Ny, n])
        By = np.zeros([Ny, Nu])
        Cy = np.zeros([Ny, Nw])
        Ey = np.zeros([Ny, Nw])
        Cx_tilde = np.zeros([Nx, Nw + 1])
        Cy_tilde = np.zeros([Ny + 1, Nw + 1])
        Ey_tilde = np.zeros([Ny + 1, Nw + 1])
        D_tilde = np.zeros([Ny + 1, Nx])
        #     H

        # Ax
        for i in range(N + 1):
            Ax[i * n:(i + 1) * n, :] = matrix_power(A, i)
        # Bx
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                Bx[(i + 1) * n: (i + 2) * n, (i - j) * m: (i - j + 1) * m] = mat_temp
                mat_temp = A @ mat_temp
        # Cx
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cx[(i + 1) * n: (i + 2) * n, (i - j) * d: (i - j + 1) * d] = mat_temp
                mat_temp = A @ mat_temp
        # Ay
        for i in range(N):
            Ay[i * r:(i + 1) * r, :] = D @ matrix_power(A, i)
            # By
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                By[(i + 1) * r: (i + 2) * r, (i - j) * m: (i - j + 1) * m] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Cy
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cy[(i + 1) * r: (i + 2) * r, (i - j) * d: (i - j + 1) * d] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Ey
        for i in range(N):
            Ey[i * r: (i + 1) * r, i * d: (i + 1) * d] = E
        # Cx_tilde
        Cx_tilde[:, [0]] = Ax @ x_init
        Cx_tilde[:, 1:] = Cx
        # Cy_tilde
        Cy_tilde[r:, [0]] = Ay @ x_init
        Cy_tilde[r:, 1:] = Cy
        # Ey_tilde
        Ey_tilde[0, 0] = 1
        Ey_tilde[1:, 1:] = Ey
        # D_tilde
        for i in range(N):
            D_tilde[1 + i * r: 1 + (i + 1) * r, i * n: (i + 1) * n] = D

        return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde



    def change_xinit(self, x_k, u_k):
        '''
        x_next = A x_k + B u_k + C w_k
        y_k = D x_k + E w_k
        '''

        n = self.n
        m = self.m
        d = self.d
        r = self.r

        #     print(Nx,Nu,Nw,Ny)
        Nx = self.Nx
        Nu = self.Nu
        Ny = self.Ny
        Nw = self.Nw

        N = self.N

        # A = np.array(self.jacobian_disc_x(x_k, u_k))
        # B = np.array(self.jacobian_disc_u(x_k, u_k))
        A, B = self.disc_nonlinear_system(x_k, u_k, self.delta_t)

        C = self.C
        D = self.D
        E = self.E

        self.A = A
        self.B = B

        Ax = np.zeros([Nx, n])
        Bx = np.zeros([Nx, Nu])
        Cx = np.zeros([Nx, Nw])
        Ay = np.zeros([Ny, n])
        By = np.zeros([Ny, Nu])
        Cy = np.zeros([Ny, Nw])
        Ey = np.zeros([Ny, Nw])
        Cx_tilde = np.zeros([Nx, Nw + 1])
        Cy_tilde = np.zeros([Ny + 1, Nw + 1])
        Ey_tilde = np.zeros([Ny + 1, Nw + 1])
        D_tilde = np.zeros([Ny + 1, Nx])


        # Ax
        for i in range(N + 1):
            Ax[i * n:(i + 1) * n, :] = matrix_power(A, i)
        # Bx
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                Bx[(i + 1) * n: (i + 2) * n, (i - j) * m: (i - j + 1) * m] = mat_temp
                mat_temp = A @ mat_temp
        # Cx
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cx[(i + 1) * n: (i + 2) * n, (i - j) * d: (i - j + 1) * d] = mat_temp
                mat_temp = A @ mat_temp
        # Ay
        for i in range(N):
            Ay[i * r:(i + 1) * r, :] = D @ matrix_power(A, i)
            # By
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                By[(i + 1) * r: (i + 2) * r, (i - j) * m: (i - j + 1) * m] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Cy
        for i in range(N):
            mat_temp = C
            for j in range(i + 1):
                Cy[(i + 1) * r: (i + 2) * r, (i - j) * d: (i - j + 1) * d] = D @ mat_temp
                mat_temp = A @ mat_temp
        # Ey
        for i in range(N):
            Ey[i * r: (i + 1) * r, i * d: (i + 1) * d] = E

        self.Ax = Ax
        self.Bx = Bx
        self.Cx = Cx
        self.Ay = Ay
        self.By = By
        self.Cy = Cy
        self.Ey = Ey


        # Cx_tilde
        Cx_tilde[:, [0]] = Ax @ x_k
        Cx_tilde[:, 1:] = Cx
        # Cy_tilde
        Cy_tilde[r:, [0]] = Ay @ x_k
        Cy_tilde[r:, 1:] = Cy
        # Ey_tilde
        Ey_tilde[0, 0] = 1
        Ey_tilde[1:, 1:] = Ey
        # D_tilde
        for i in range(N):
            D_tilde[1 + i * r: 1 + (i + 1) * r, i * n: (i + 1) * n] = D

        # Stack system matrices
        self.Ey_tilde = Ey_tilde
        self.D_tilde = D_tilde

        self.Cx_tilde = Cx_tilde
        self.Cy_tilde = Cy_tilde

        # return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde

    def integrator_rk4(self, f, x, u, delta_t):

        """
        Runge-Kutta 4th order solver using casadi.
        Args:
            f: First order ODE in casadi function (Nx + Nt -> Nx).
            t: Current time.
            x: Current value.
            u: Current input.
            delta_t: Step length.
        Returns:
            x_next: Vector of next value in casadi DM
        """
        k1 = f(x, u)
        k2 = f(x + delta_t / 2 * k1, u)
        k3 = f(x + delta_t / 2 * k2, u)
        k4 = f(x + delta_t * k3, u)
        x_next = x + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next






    def disc_linear_system(self, A, B, delta_t):
        '''
        Discrete a linear system with implicit Euler
        x[k+1] = (I - delta_t * A)^{-1} @ x[k] + (I - delta_t * A)^{-1} @ (delta_t * B) @ u[k]

        Returns:
            Ak
            Bk

        '''
        Nx = np.shape(A)[0]
        Ix = np.identity(Nx)

        Ak = np.linalg.inv(Ix - delta_t * A)
        Bk = np.linalg.inv(Ix - delta_t * A) @ (delta_t * B)

        return Ak, Bk

    def disc_nonlinear_system(self, xk, uk, delta_t):
        '''

        '''
        A = np.array(self.jacobian_ode_x(xk, uk))
        B = np.array(self.jacobian_ode_u(xk, uk))
        Nx = np.shape(A)[0]
        Ix = np.identity(Nx)
        Ak = np.linalg.inv(Ix - delta_t * A)
        Bk = np.linalg.inv(Ix - delta_t * A) @ (delta_t * B)

        return Ak, Bk

    # def disc_nonlinear_system(self, xk, uk, delta_t):
    #     '''
    #
    #     '''
    #     A = np.array(self.jacobian_ode_x(xk, uk))
    #     B = np.array(self.jacobian_ode_u(xk, uk))
    #     # A[1,2] = 0.000001
    #     Ak = expm(A*delta_t)
    #     # print(A)
    #     Bk = (expm(A * delta_t) - np.identity(self.n)) @ np.linalg.inv(A) @ B
    #
    #     return Ak, Bk

    # def disc_nonlinear_system(self, xk, uk, delta_t):
    #
    #     """
    #     Runge-Kutta 4th order solver using casadi.
    #     Args:
    #         f: First order ODE in casadi function (Nx + Nt -> Nx).
    #         t: Current time.
    #         x: Current value.
    #         u: Current input.
    #         delta_t: Step length.
    #     Returns:
    #         x_next: Vector of next value in casadi DM
    #     """
    #     f = self.linear_ode_func
    #     k1 = f(xk, uk)
    #     k2 = f(xk + delta_t / 2 * k1, uk)
    #     k3 = f(xk + delta_t / 2 * k2, uk)
    #     k4 = f(xk + delta_t * k3, uk)
    #     x_next = xk + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #
    #     return x_next