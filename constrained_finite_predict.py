import numpy as np
from scipy.optimize import linprog
from scipy import linalg as lin
from math import *
from matplotlib import pyplot as plt
import casadi


Q_x = 1
Q_y = 1
R_u = 0.1
N = 10

x_0 = 1.1
y_0 = -2.1

x_ref = 0
y_ref = 0

u_max = 2
u_min = -u_max

A= np.mat([[1.7, 1], [0, 1]])
B= np.mat([[0.5], [1]])

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
Q = casadi.diagcat(Q_x, Q_y,)
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

args = {
    'lbg': casadi.DM.zeros((n_states * (N + 1))),
    'ubg': casadi.DM.zeros((n_states * (N + 1))),
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
