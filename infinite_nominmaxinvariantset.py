import numpy as np
from scipy.optimize import linprog
from scipy import linalg as lin
from math import *
from matplotlib import pyplot as plt
import casadi


Q_x = 1
Q_y = 1
R_u = 0.1
N = 2

x_0 = 1.1
y_0 = -2.1

x_ref = 0
y_ref = 0

u_max = 2
u_min = -u_max

A= np.mat([[1.7, 1], [0, 0.95]])
B= np.mat([[0.5], [1]])

def lqr_p_k_cal(a: np.matrix, b: np.matrix, q: np.matrix, r: np.matrix):
    p = np.mat(lin.solve_discrete_are(a, b, q, r))  # 求Riccati方程
    k = lin.inv(r + b.T * p * b) * b.T * p * a

    return p, k

p, k =lqr_p_k_cal(A, B, casadi.diagcat(Q_x, Q_y),
                  casadi.diagcat(R_u))

Ak=A-B*k


def domain_of_xf(
                 u_min,
                 u_max,
                 k: np.matrix
                 ):

    xf_u_a = np.vstack((-k, k))
    xf_u_b = np.mat([[-u_min, u_max]])  # 由于U是线段，因此计算出KZ的最大最小值进行闵可夫斯基差

    xf_a = np.mat(xf_u_a)
    xf_b = np.mat(xf_u_b)
    # 下面的返回值偷懒了，因为知道X-Z，U-KZ的形式是x，u的取值范围，所以只返回最值，正常情况应返回不等式组
    return xf_a, xf_b


A_D, b_D=domain_of_xf(u_min, u_max,k)


def collinear(a: np.matrix, b: np.matrix):
    c = np.hstack((a, b.T))
    delete_line = []

    for i in range(0, c.shape[0]):
        for j in range(i + 1, c.shape[0]):
            test_mat = np.vstack((c[i, :], c[j, :]))
            if np.linalg.matrix_rank(test_mat) < 2:
                delete_line.append(j)

    c = np.delete(c, delete_line, 0)

    new_a = np.mat(np.delete(c, -1, 1))
    new_b = np.mat(c[:, -1].T)

    return new_a, new_b

def xf_cal( d_a: np.matrix, d_b: np.matrix, a: np.matrix):
    t = 0
    a_cons, b_cons = d_a, d_b  # 初始状态

    while True:
        max_res = []

        for i in range(0, d_a.shape[0]):
            c = d_a[i, :] * (a ** (t + 1))
            bounds = [(None, None)] * d_a.shape[1]
            res = linprog(-c, A_ub=a_cons, b_ub=b_cons, bounds=bounds, method='revised simplex')
            max_res.append(-res.fun)
        # 检验Ot+1是否与Ot相等

        t = t + 1
        if ((np.mat(max_res) - d_b) <= 0).all():
            break  # 若相等则跳出循环

        a_cons = np.mat(np.vstack((a_cons, d_a * a ** t)))
        b_cons = np.mat(np.hstack((b_cons, d_b)))

        # 若不是则令Ot = Ot+1继续循环

    a_cons, b_cons = collinear(a_cons, b_cons)
    # 计算方法是增加t，直到Ot == Ot+1，于是有O∞ = Ot
    return a_cons, b_cons
    # 计算终端约束区域Xf
A_Xf= np.mat([[1, 0], [0, 1]])
b_Xf= np.mat([[0.1], [0.1]]).reshape(1,-1)

# A_Xf, b_Xf = xf_cal(A_D, b_D, Ak)
# print(A_Xf)
# print(b_Xf.shape)
def dm_to_array(dm):
    return np.array(dm.full())

def shift_timestep( state, control, f):
    next_state = f(state, control[:, 0])
    next_control = casadi.horzcat(control[:, 1:],
                                  casadi.reshape(control[:, -1], -1, 1))

    return  next_state, next_control



x = casadi.SX.sym('x')
y = casadi.SX.sym('y')
states = casadi.vertcat(x, y)
n_states = states.numel()

u= casadi.SX.sym('u')
controls = u
n_controls = controls.numel()

#定义
X = casadi.SX.sym('X', n_states, N + 1)
U = casadi.SX.sym('U', n_controls, N)
P = casadi.SX.sym('P', 2 * n_states)
Q = casadi.diagcat(Q_x, Q_y)
R = casadi.diagcat(R_u)
A = casadi.SX(A)
B = casadi.SX(B)  # 状态矩阵

st_fun_nom = A @ states + B @ controls
f_nom = casadi.Function('f_nom', [states, controls], [st_fun_nom])  # 对应状态方程中的f()

cost = 0
g = X[:, 0] - P[:n_states]

for k in range(N):
    state = X[:, k]
    control = U[:, k]
    cost = cost + (state - P[n_states:]).T @ Q @ (state - P[n_states:]) + \
            control.T @ R @ control
    next_state = X[:, k + 1]
    predicted_state = f_nom(state, control)
    g = casadi.vertcat(g, next_state - predicted_state)

cost=cost++ 1 / 2 * X[:, N].T @ p @ X[:, N] # 还应加上终端代价函数
g = casadi.vertcat(g, A_Xf @ X[:, N])  # 还有终端区域约束，输入约束加在bounds里
opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

nlp_prob = {
    'f': cost,
    'x': opt_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'sb': 'yes',
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0,
}

solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))
ubx = casadi.DM.zeros((n_states * (N + 1) + n_controls * N, 1))

lbx[0:n_states * (N + 1):n_states] = -casadi.inf
lbx[1:n_states * (N + 1):n_states] = -casadi.inf

ubx[0:n_states * (N + 1):n_states] = casadi.inf
ubx[1:n_states * (N + 1):n_states] = casadi.inf

lbx[n_states * (N + 1):] = -1 # u 的下界
ubx[n_states * (N + 1):] = 1  # u 的上界

xf_num = A_Xf.shape[0]
lbg = casadi.DM.zeros((n_states * (N+1) +  xf_num, 1))
ubg = casadi.DM.zeros((n_states * (N+1) +  xf_num, 1))

for i in range(0, xf_num):
    lbg[n_states * (N+1)  + i] = -0.1
    ubg[n_states * (N+1)  + i] = float(b_Xf[0, i])  # x(N)在终端区域内

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx
}

#初始状态
state_0 = casadi.DM([x_0, y_0])
state_ref = casadi.DM([x_ref, y_ref])

u0 = casadi.DM.zeros((n_controls, N))
X0 = casadi.repmat(state_0, 1, N + 1)

cat_states = dm_to_array(X0)
cat_controls = dm_to_array(u0[:, 0])

State=dm_to_array(state_0)


if __name__ == '__main__':

    for i in range (20):
        args['p'] = casadi.vertcat(state_0, state_ref)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, n_states * (N + 1), 1),
                                    casadi.reshape(u0, n_controls * N, 1))
        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                     lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = casadi.reshape(sol['x'][:n_states * (N + 1)], n_states, N + 1)

        cat_states = np.dstack((cat_states, dm_to_array(X0)))
        cat_controls = np.dstack((cat_controls, dm_to_array(u[:, 0])))

        state_0, u0 = shift_timestep(state_0, u, f_nom)
        State=np.hstack((State,state_0))

        X0 = casadi.horzcat(X0[:, 1:], casadi.reshape(X0[:, -1], -1, 1))


plt.plot(State[0,:], label="x1")
plt.plot(State[1,:], label="x2")
plt.show()

plt.plot(cat_controls[0,0,:], label="x2")
plt.show()
