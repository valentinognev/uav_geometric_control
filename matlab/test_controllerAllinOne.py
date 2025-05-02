import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
import matplotlib.pyplot as plt

# % Copyright (c) 2020 Flight Dynamics and Control Lab
# %
# % Permission is hereby granted, free of charge, to any person obtaining a
# % copy of this software and associated documentation files (the
# % "Software"), to deal in the Software without restriction, including
# % without limitation the rights to use, copy, modify, merge, publish,
# % distribute, sublicense, and/or sell copies of the Software, and to permit
# % persons to whom the Software is furnished to do so, subject to the
# % following conditions:
# %
# % The above copyright notice and this permission notice shall be included
# %  in all copies or substantial portions of the Software.
# %
# % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# % OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# % MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# % IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# % CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# % TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# % SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def test_controllerAllinOne():
    # %% Simulation parameters
    t = np.arange(0, 10.01, 0.01)
    N = len(t)

    # Quadrotor
    J1 = 0.02
    J2 = 0.02
    J3 = 0.04
    param = {
        'J': np.diag([J1, J2, J3]),
        'm': 2,
        'd': 0.169,
        'ctf': 0.0135,
        'x_delta': np.array([0.5, 0.8, -1]),
        'R_delta': np.array([0.2, 1.0, -0.1]),
        'g': 9.81
    }

    # %% Controller gains
    k = {
        'x': 10,
        'v': 8,
        'i': 10,
        'R': 1.5,
        'W': 0.35,
        'I': 10,
        'y': 0.8,
        'wy': 0.15,
        'yI': 2
    }
    param['c1'] = 1.5
    param['sigma'] = 10
    param['c2'] = 2
    param['c3'] = 2

    # %% Initial conditions
    x0 = np.zeros(3)
    v0 = np.zeros(3)
    R0 = expm(hat(np.array([0, 0, 1])) * np.pi/2)
    W0 = np.zeros(3)
    X0 = np.concatenate([x0, v0, W0, R0.flatten(), np.zeros(6)])

    # %% Numerical integration
    X = odeint(func=eom, y0=X0, t=t, args=(k, param), rtol=1e-6, atol=1e-6)

    # %% Post processing
    # Create empty arrays to save data
    e, d, R, f, M = generate_output_arrays(N)

    # Unpack the outputs of ode45 function
    x = X[:, 0:3].T
    v = X[:, 3:6].T
    W = X[:, 6:9].T
    ei = X[:, 18:21].T
    eI = X[:, 21:24].T

    for i in range(N):
        R[:, :, i] = X[i, 9:18].reshape(3, 3)
        des = command(t[i])
        f[i], M[:, i], _, _, err, calc = position_control(X[i, :], des, k, param)

        # Unpack errors
        e['x'][:, i] = err['x']
        e['v'][:, i] = err['v']
        e['R'][:, i] = err['R']
        e['W'][:, i] = err['W']
        e['y'][i] = err['y']
        e['Wy'][i] = err['Wy']

        # Unpack desired values
        d['x'][:, i] = des['x']
        d['v'][:, i] = des['v']
        d['b1'][:, i] = des['b1']
        d['R'][:, :, i] = calc['R']

    # Plot data
    linetype = 'k'
    linewidth = 1
    xlabel_ = 'time (s)'
    
    plot_3x1(t, e['R'], '', xlabel_, 'e_R', linetype, linewidth)
    plot_3x1(t, e['x'], '', xlabel_, 'e_x', linetype, linewidth)
    plot_3x1(t, e['v'], '', xlabel_, 'e_v', linetype, linewidth)

    fig = plot_3x1(t, eI * np.array([k['I'], k['I'], k['yI']])[:, None], '', xlabel_, 'e', linetype, linewidth)
    plot_3x1(t, np.tile(param['R_delta'], (N, 1)).T, '', xlabel_, 'e_I', 'r', linewidth, fig=fig)

    fig = plot_3x1(t, ei * k['i'], '', xlabel_, 'e_i', linetype, linewidth)
    plot_3x1(t, np.tile(param['x_delta'], (N, 1)).T, '', xlabel_, 'e_i', 'r', linewidth, fig=fig)

    fig = plot_3x1(t, x, '', xlabel_, 'x', linetype, linewidth)
    plot_3x1(t, d['x'], '', xlabel_, 'x', 'r', linewidth, fig=fig)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x[0, :], x[1, :], x[2, :], 'k')
    ax.plot3D(d['x'][0, :], d['x'][1, :], d['x'][2, :], 'r')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlabel('$x_1$')
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    plt.show()

    print('')

def eom(X, t, k, param):
    e3 = np.array([0, 0, 1])
    m = param['m']
    J = param['J']

    _, v, R, W, _, _ = split_to_states(X)

    desired = command(t)
    f, M, ei_dot, eI_dot, _, _ = position_control(X, desired, k, param)

    xdot = v
    vdot = param['g'] * e3 - f / m * R @ e3 + param['x_delta'] / m
    Wdot = np.linalg.inv(J) @ (-hat(W) @ J @ W + M + param['R_delta'])
    Rdot = R @ hat(W)

    return np.concatenate([xdot, vdot, Wdot, Rdot.flatten(), ei_dot, eI_dot])

def command(t):
    return command_Lissajous(t)

def position_control(X, desired, k, param):
    use_decoupled = False

    # Unpack states
    x, v, R, W, ei, eI = split_to_states(X)

    sigma = param['sigma']
    c1 = param['c1']
    m = param['m']
    g = param['g']
    e3 = np.array([0, 0, 1])

    error_ = {
        'x': x - desired['x'],
        'v': v - desired['v']
    }

    A = -k['x'] * error_['x'] - k['v'] * error_['v'] - m * g * e3 + m * desired['x_2dot'] - k['i'] * sat(sigma, ei)
    ei_dot = error_['v'] + c1 * error_['x']
    b3 = R @ e3
    f = -np.dot(A, b3)
    ea = g * e3 - f / m * b3 - desired['x_2dot'] + param['x_delta'] / m
    A_dot = -k['x'] * error_['v'] - k['v'] * ea + m * desired['x_3dot'] - k['i'] * satdot(sigma, ei, ei_dot)

    ei_ddot = ea + c1 * error_['v']
    b3_dot = R @ hat(W) @ e3
    f_dot = -np.dot(A_dot, b3) - np.dot(A, b3_dot)
    eb = -f_dot / m * b3 - f / m * b3_dot - desired['x_3dot']
    A_ddot = -k['x'] * ea - k['v'] * eb + m * desired['x_4dot'] - k['i'] * satdot(sigma, ei, ei_ddot)

    b3c, b3c_dot, b3c_ddot = deriv_unit_vector(-A, -A_dot, -A_ddot)

    A2 = -hat(desired['b1']) @ b3c
    A2_dot = -hat(desired['b1_dot']) @ b3c - hat(desired['b1']) @ b3c_dot
    A2_ddot = -hat(desired['b1_2dot']) @ b3c - 2 * hat(desired['b1_dot']) @ b3c_dot - hat(desired['b1']) @ b3c_ddot

    b2c, b2c_dot, b2c_ddot = deriv_unit_vector(A2, A2_dot, A2_ddot)

    b1c = hat(b2c) @ b3c
    b1c_dot = hat(b2c_dot) @ b3c + hat(b2c) @ b3c_dot
    b1c_ddot = hat(b2c_ddot) @ b3c + 2 * hat(b2c_dot) @ b3c_dot + hat(b2c) @ b3c_ddot

    Rc = np.column_stack([b1c, b2c, b3c])
    Rc_dot = np.column_stack([b1c_dot, b2c_dot, b3c_dot])
    Rc_ddot = np.column_stack([b1c_ddot, b2c_ddot, b3c_ddot])

    Wc = vee(Rc.T @ Rc_dot)
    Wc_dot = vee(Rc.T @ Rc_ddot - hat(Wc) @ hat(Wc))

    W3 = np.dot(R @ e3, Rc @ Wc)
    W3_dot = np.dot(R @ e3, Rc @ Wc_dot) + np.dot(R @ hat(W) @ e3, Rc @ Wc)

    # Run attitude controller
    if use_decoupled:
        M, eI_dot, error_['b'], error_['W'], error_['y'], error_['Wy'] = attitude_control_decoupled_yaw(
            R, W, eI, b3c, b3c_dot, b3c_ddot, b1c, W3, W3_dot, k, param)
        error_['R'] = 0.5 * vee(Rc.T @ R - R.T @ Rc)
    else:
        M, eI_dot, error_['R'], error_['W'] = attitude_control(
            R, W, eI, Rc, Wc, Wc_dot, k, param)
        error_['y'] = 0
        error_['Wy'] = 0

    # Saving data
    calculated = {
        'b3': b3c,
        'b3_dot': b3c_dot,
        'b3_ddot': b3c_ddot,
        'b1': b1c,
        'R': Rc,
        'W': Wc,
        'W_dot': Wc_dot,
        'W3': np.dot(R @ e3, Rc @ Wc),
        'W3_dot': np.dot(R @ e3, Rc @ Wc_dot) + np.dot(R @ hat(W) @ e3, Rc @ Wc)
    }

    return f, M, ei_dot, eI_dot, error_, calculated

def attitude_control(R, W, eI, Rd, Wd, Wd_dot, k, param):
    eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
    eW = W - R.T @ Rd @ Wd

    kR = np.diag([k['R'], k['R'], k['y']])
    kW = np.diag([k['W'], k['W'], k['wy']])

    M = -kR @ eR - kW @ eW - k['I'] * eI + hat(R.T @ Rd @ Wd) @ param['J'] @ R.T @ Rd @ Wd + param['J'] @ R.T @ Rd @ Wd_dot
    eI_dot = eW + param['c2'] * eR

    return M, eI_dot, eR, eW

def attitude_control_decoupled_yaw(R, W, eI, b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, k, param):
    J = param['J']
    c2 = param['c2']
    c3 = param['c3']

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    b1 = R @ e1
    b2 = R @ e2
    b3 = R @ e3

    w = W[0] * b1 + W[1] * b2
    b3_dot = hat(w) @ b3

    wd = hat(b3d) @ b3d_dot
    wd_dot = hat(b3d) @ b3d_ddot

    eb = hat(b3d) @ b3
    ew = w + hat(b3) @ hat(b3) @ wd
    tau = -k['R'] * eb - k['W'] * ew - J[0, 0] * np.dot(b3, wd) * b3_dot - J[0, 0] * hat(b3) @ hat(b3) @ wd_dot - k['I'] * eI[0] * b1 - k['I'] * eI[1] * b2

    tau1 = np.dot(b1, tau)
    tau2 = np.dot(b2, tau)

    M1 = tau1 + J[2, 2] * W[2] * W[1]
    M2 = tau2 - J[2, 2] * W[2] * W[0]

    ey = -np.dot(b2, b1c)
    ewy = W[2] - wc3

    M3 = -k['y'] * ey - k['wy'] * ewy - k['yI'] * eI[2] + J[2, 2] * wc3_dot

    eI_dot = np.array([
        np.dot(b1, c2 * eb + ew),
        np.dot(b2, c2 * eb + ew),
        c3 * ey + ewy
    ])

    return np.array([M1, M2, M3]), eI_dot, eb, ew, ey, ewy

def deriv_unit_vector(q, q_dot, q_ddot):
    nq = np.linalg.norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * np.dot(q, q_dot) / nq**3

    u_ddot = q_ddot / nq - q_dot / nq**3 * (2 * np.dot(q, q_dot)) - q / nq**3 * (np.dot(q_dot, q_dot) + np.dot(q, q_ddot)) + 3 * q / nq**5 * np.dot(q, q_dot)**2

    return u, u_dot, u_ddot

def hat(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def vee(S):
    return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

def sat(sigma, y):
    z = np.zeros_like(y)
    for k in range(len(y)):
        if y[k] > sigma:
            z[k] = sigma
        elif y[k] < -sigma:
            z[k] = -sigma
        else:
            z[k] = y[k]
    return z

def satdot(sigma, y, ydot):
    z = np.zeros_like(y)
    for k in range(len(y)):
        if y[k] > sigma or y[k] < -sigma:
            z[k] = 0
        else:
            z[k] = ydot[k]
    return z

def split_to_states(X):
    x = X[0:3]
    v = X[3:6]
    W = X[6:9]
    R = X[9:18].reshape(3, 3)
    ei = X[18:21]
    eI = X[21:24]
    return x, v, R, W, ei, eI

def generate_output_arrays(N):
    error_ = {
        'x': np.zeros((3, N)),
        'v': np.zeros((3, N)),
        'R': np.zeros((3, N)),
        'W': np.zeros((3, N)),
        'y': np.zeros(N),
        'Wy': np.zeros(N)
    }

    desired = {
        'x': np.zeros((3, N)),
        'v': np.zeros((3, N)),
        'b1': np.zeros((3, N)),
        'R': np.zeros((3, 3, N))
    }

    R = np.zeros((3, 3, N))
    f = np.zeros(N)
    M = np.zeros((3, N))

    return error_, desired, R, f, M

def plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth=1, font_size=10, fig=None):
    if fig is None:
        fig = plt.figure()
    axs = fig.subplots(3, 1)
    for i in range(3):
        axs[i].plot(x, y[i, :] if len(y.shape) > 1 else y, linetype, linewidth=linewidth)
    axs[1].set_ylabel(f'${ylabel_}$')
    axs[2].set_xlabel(xlabel_)
    fig.suptitle(title_)
    
    return fig

def command_Lissajous(t):
    A = 1
    B = 1
    C = 0.2
    d = np.pi / 2 * 0
    a = 2
    b = 3
    c = 2
    alt = -1

    desired = {
        'x': np.array([A * np.sin(a * t + d), B * np.sin(b * t), alt + C * np.cos(c * t)]),
        'v': np.array([A * a * np.cos(a * t + d), B * b * np.cos(b * t), C * c * -np.sin(c * t)]),
        'x_2dot': np.array([A * a**2 * -np.sin(a * t + d), B * b**2 * -np.sin(b * t), C * c**2 * -np.cos(c * t)]),
        'x_3dot': np.array([A * a**3 * -np.cos(a * t + d), B * b**3 * -np.cos(b * t), C * c**3 * np.sin(c * t)]),
        'x_4dot': np.array([A * a**4 * np.sin(a * t + d), B * b**4 * np.sin(b * t), C * c**4 * np.cos(c * t)]),
        'b1': np.array([np.cos(2*np.pi/10 * t), np.sin(2*np.pi/10 * t), 0]),
        'b1_dot': 2*np.pi/10 * np.array([-np.sin(2*np.pi/10 * t), np.cos(2*np.pi/10 * t), 0]),
        'b1_2dot': (2*np.pi/10)**2 * np.array([-np.cos(2*np.pi/10 * t), -np.sin(2*np.pi/10 * t), 0])
    }
    return desired

def command_line(t):
    height = 0
    w = 2 * np.pi / 10
    
    desired = {
        'x': np.array([0.5 * t, 0, -height]),
        'v': np.array([0.5, 0, 0]),
        'x_2dot': np.zeros(3),
        'x_3dot': np.zeros(3),
        'x_4dot': np.zeros(3),
        'b1': np.array([np.cos(w * t), np.sin(w * t), 0]),
        'b1_dot': w * np.array([-np.sin(w * t), np.cos(w * t), 0]),
        'b1_2dot': w**2 * np.array([-np.cos(w * t), -np.sin(w * t), 0])
    }
    return desired

if __name__ == "__main__":
    test_controllerAllinOne()
    plt.show()