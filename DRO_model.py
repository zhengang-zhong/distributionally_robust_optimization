import numpy as np
from numpy.linalg import matrix_power
class Model:

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

        Ak, Bk = self.disc_linear_system(A, B, delta_t)
        self.Ak = Ak
        self.Bk = Bk
        self.C = C
        self.D = D
        self.E = E

        n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde = self.stack_system(Ak, Bk, C, D, E, x_init, N)
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
        Cx_tilde[:, 0: 1] = Ax @ x_init
        Cx_tilde[:, 1:] = Cx
        # Cy_tilde
        Cy_tilde[r:, 0: 1] = Ay @ x_init
        Cy_tilde[r:, 1:] = Cy
        # Ey_tilde
        Ey_tilde[0, 0] = 1
        Ey_tilde[1:, 1:] = Ey
        # D_tilde
        for i in range(N):
            D_tilde[1 + i * r: 1 + (i + 1) * r, i * d: (i + 1) * d] = D

        return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde



    def change_xinit(self, x_init):
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

        A = self.Ak
        B = self.Bk
        C = self.C
        D = self.D
        E = self.E

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
            D_tilde[1 + i * r: 1 + (i + 1) * r, i * d: (i + 1) * d] = D


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
        # return n, m, d, r, Nx, Nu, Ny, Nw, Ax, Bx, Cx, Ay, By, Cy, Ey, Cx_tilde, Cy_tilde, Ey_tilde, D_tilde