import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

def run_brescianini():
    """Main function to run the Brescianini controller simulation."""
    
    # Simulation parameters
    t = np.arange(0, 10.01, 0.01)
    N = len(t)

    # Quadrotor parameters
    J1 = 0.02
    J2 = 0.02
    J3 = 0.04
    param = {
        'J': np.diag([J1, J2, J3]),
        'm': 0.5,
        'g': 9.81,
        'd': 0.169,
        'c_tf': 0.0135,
        'use_disturbances': False,  # Change to True to use disturbances
        'W_x': np.eye(3),
        'theta_x': np.array([1, 0.8, -1]),
        'W_R': np.eye(3),
        'theta_R': np.array([0.1, 0.1, -0.1]),
        'kp_xy': 1,
        'kd_xy': 1.4,
        'kp_z': 1,
        'kd_z': 1.4
    }

    # Controller gains for Brescianini
    # Position
    kx = 1
    kv = 1.4
    k = {
        'x': np.diag([kx, kx, 1]),
        'v': np.diag([kv, kv, 1.4])
    }

    # Initial conditions
    x0 = np.array([0, 20, -40])  # for line
    # x0 = np.array([1, -1, 0])  # for circle
    v0 = np.array([0, 0, 0])
    # R0 = expm((np.pi - 0.01) * hat(np.array([0, 0, 1])))
    R0 = np.array([[0, 1, 0],
                   [-1, 0, 0],
                   [0, 0, 1]])  # for line
    W0 = np.array([0, 0, 0])
    X0 = np.concatenate([x0, v0, W0, R0.flatten()])

    # Numerical integration
    sol = odeint(lambda X, t: eom_brescianini(t, X, k, param), X0, t, 
                rtol=1e-6, atol=1e-6)
    X = sol.T

    # Output arrays
    e, d, R, f, M = generate_output_arrays(N)
    thr = np.zeros((4, N))

    # Post processing
    x = X[:3, :]
    v = X[3:6, :]
    W = X[6:9, :]

    avg_ex = 0
    avg_eR = 0
    avg_f = 0

    converge_t = 0
    is_converged = False
    converge_ex = 0.02

    for i in range(N):
        R[:, :, i] = X[9:18, i].reshape(3, 3)
        
        des = command(t[i])
        f[i], M[:, i], err, calc = position_control(X[:, i], des, k, param)
        
        f[i], M[:, i] = saturate_fM(f[i], M[:, i], param)
        thr[:, i] = fM_to_thr(f[i], M[:, i], param)
        
        # Unpack errors
        e['x'][:, i] = err['x']
        e['v'][:, i] = err['v']
        e['R'][:, i] = err['R']
        e['W'][:, i] = W[:, i] - calc['W']

        # Unpack desired values
        d['x'][:, i] = des['x']
        d['v'][:, i] = des['v']
        d['b1'][:, i] = des['b1']
        d['R'][:, :, i] = calc['R']
        d['W'][:, i] = calc['W']
        
        # Find normalized errors
        norm_ex = norm(err['x'])
        norm_eR = norm(err['R'])
        
        avg_ex += norm_ex
        avg_eR += norm_eR
        
        norm_f = norm(thr[:, i])
        avg_f += norm_f
        
        if norm_ex < converge_ex and not is_converged:
            converge_t = t[i]
            is_converged = True

    avg_ex /= N
    avg_eR /= N
    avg_f /= N

    print(f"avg_ex: {avg_ex}")
    print(f"avg_eR: {avg_eR}")
    print(f"avg_f: {avg_f}")
    print(f"converge_t: {converge_t}")

    # Plots
    linetype = 'k'
    linewidth = 1
    xlabel_ = 'time (s)'

    plot_3x1(t, e['R'], '', xlabel_, 'e_R', linetype, linewidth)
    plot_3x1(t, e['x'], '', xlabel_, 'e_x', linetype, linewidth)
    plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)

    # Save data
    np.savez('brescianini.npz', t=t, X=X, e=e, d=d, R=R, f=f, M=M, thr=thr)
    plt.show()
########################################################################################################
def eom_brescianini(t, X, k, param):
    """Equations of motion for the quadrotor."""
    e3 = np.array([0, 0, 1])
    m = param['m']
    J = param['J']

    x, v, R, W = split_to_states(X)

    desired = command(t)
    f, M, _, _ = position_control(X, desired, k, param)

    f, M = saturate_fM(f, M, param)

    x_dot = v
    v_dot = param['g'] * e3 - f / m * R @ e3
    if param['use_disturbances']:
        v_dot += param['W_x'] @ param['theta_x'] / m

    if param['use_disturbances']:
        W_dot = np.linalg.inv(J) @ (-hat(W) @ J @ W + M + param['W_R'] @ param['theta_R'])
    else:
        W_dot = np.linalg.inv(J) @ (-hat(W) @ J @ W + M)
    
    R_dot = R @ hat(W)

    return np.concatenate([x_dot, v_dot, W_dot, R_dot.flatten()])
########################################################################################################

def saturate_fM(f, tau, param):
    """Saturate force and moment commands."""
    l = param['d']
    c_tf = param['c_tf']

    u_max = 8
    u_min = 0.1

    tau_hat = np.zeros(3)
    tau_max_xy = (u_max - u_min) * l
    for i in range(2):
        tau_hat[i] = saturate(tau[i], -tau_max_xy, tau_max_xy)

    tau_hat_x = tau_hat[0]
    tau_hat_y = tau_hat[1]
    f_min = 4*u_min + abs(tau_hat_x)/l + abs(tau_hat_y)/l
    f_max = 4*u_max - abs(tau_hat_x)/l - abs(tau_hat_y)/l
    f_hat = saturate(f, f_min, f_max)

    tau_min_z_list = [
        c_tf*(4*u_min - f_hat + 2*abs(tau_hat_x)/l),
        c_tf*(-4*u_max + f_hat + 2*abs(tau_hat_y)/l)
    ]
    tau_min_z = max(tau_min_z_list)

    tau_max_z_list = [
        c_tf*(4*u_max - f_hat - 2*abs(tau_hat_x)/l),
        c_tf*(-4*u_min + f_hat - 2*abs(tau_hat_y)/l)
    ]
    tau_max_z = min(tau_max_z_list)

    tau_hat[2] = saturate(tau[2], tau_min_z, tau_max_z)
    return f_hat, tau_hat
########################################################################################################

def split_to_states(X):
    """Split state vector into components."""
    x = X[:3]
    v = X[3:6]
    W = X[6:9]
    R = X[9:18].reshape(3, 3)
    return x, v, R, W
########################################################################################################

def position_control(X, desired, k, param):
    """Position controller."""
    x, v, R, W = split_to_states(X)

    m = param['m']
    g = param['g']
    e3 = np.array([0, 0, 1])

    error = {
        'x': x - desired['x'],
        'v': v - desired['v']
    }

    A = -k['x'] @ error['x'] \
        - k['v'] @ error['v'] \
        - m*g*e3 \
        + m*desired['x_2dot']

    b3 = R @ e3
    f = -A @ b3
    ea = g*e3 - f/m*b3 - desired['x_2dot']
    A_dot = -k['x'] @ error['v'] - k['v'] @ ea + m*desired['x_3dot']

    b3_dot = R @ hat(W) @ e3
    f_dot = -A_dot @ b3 - A @ b3_dot
    eb = -f_dot/m*b3 - f/m*b3_dot - desired['x_3dot']
    A_ddot = -k['x'] @ ea - k['v'] @ eb + m*desired['x_4dot']

    b3c, b3c_dot, b3c_ddot = deriv_unit_vector(-A, -A_dot, -A_ddot)

    A2 = -hat(desired['b1']) @ b3c
    A2_dot = -hat(desired['b1_dot']) @ b3c - hat(desired['b1']) @ b3c_dot
    A2_ddot = (-hat(desired['b1_2dot']) @ b3c - 2*hat(desired['b1_dot']) @ b3c_dot 
              - hat(desired['b1']) @ b3c_ddot)

    b2c, b2c_dot, b2c_ddot = deriv_unit_vector(A2, A2_dot, A2_ddot)

    b1c = hat(b2c) @ b3c
    b1c_dot = hat(b2c_dot) @ b3c + hat(b2c) @ b3c_dot
    b1c_ddot = (hat(b2c_ddot) @ b3c + 2*hat(b2c_dot) @ b3c_dot 
               + hat(b2c) @ b3c_ddot)

    Rc = np.column_stack([b1c, b2c, b3c])
    Rc_dot = np.column_stack([b1c_dot, b2c_dot, b3c_dot])
    Rc_ddot = np.column_stack([b1c_ddot, b2c_ddot, b3c_ddot])

    Wc = vee(Rc.T @ Rc_dot)
    Wc_dot = vee(Rc.T @ Rc_ddot - hat(Wc) @ hat(Wc))

    # Run attitude controller
    M, _ = attitude_control_brescianini(R, W, Rc, Wc, Wc_dot, param)
    error['R'] = 0.5 * vee(Rc.T @ R - R.T @ Rc)

    # Saving data
    calculated = {
        'b3': b3c,
        'b3_dot': b3c_dot,
        'b3_ddot': b3c_ddot,
        'b1': b1c,
        'R': Rc,
        'W': Wc,
        'W_dot': Wc_dot
    }

    return f, M, error, calculated
########################################################################################################

def fM_to_thr(f, M, param):
    """Convert force and moment to individual thrusts."""
    d = param['d']
    ctf = param['c_tf']

    f_to_fM = np.array([
        [1, 1, 1, 1],
        [0, -d, 0, d],
        [d, 0, -d, 0],
        [-ctf, ctf, -ctf, ctf]
    ])

    return np.linalg.solve(f_to_fM, np.concatenate([[f], M]))
########################################################################################################

def command(t):
    """Generate command signal."""
    return command_point(t)
    # return command_circle(t)
########################################################################################################

def attitude_control_brescianini(R, w, Rd, wd, wd_dot, param):
    """Attitude controller."""
    J = param['J']
    kp_xy = param['kp_xy']
    kp_z = param['kp_z']
    kd_xy = param['kd_xy']
    kd_z = param['kd_z']

    q = Quaternion(matrix=R)
    qd = Quaternion(matrix=Rd)
    qe = qd * q.conjugate

    wd_bar = qe.rotation_matrix.T @ wd
    we = wd_bar - w
    wd_bar_dot = hat(we) @ wd_bar + qe.rotation_matrix.T @ wd_dot

    qe = qe.conjugate
    q0, q1, q2, q3 = qe.elements

    q0q3 = np.sqrt(q0*q0 + q3*q3)
    B = np.array([
        q0*q0 + q3*q3,
        q0*q1 - q2*q3,
        q0*q2 + q1*q3,
        0
    ])
    qe_red = B / q0q3
    qe_yaw = np.array([q0, 0, 0, q3]) / q0q3

    tilde_qe_red = qe_red[1:]
    tilde_qe_yaw = qe_yaw[1:]

    tau_ff = J @ wd_bar_dot - hat(J @ w) @ w

    Kd = np.diag([kd_xy, kd_xy, kd_z])
    tau = (kp_xy * tilde_qe_red + kp_z * np.sign(q0) * tilde_qe_yaw 
           + Kd @ we + tau_ff)

    error = {
        'qe': qe,
        'we': we
    }

    return tau, error
########################################################################################################

def vee(S):
    """Convert skew-symmetric matrix to vector."""
    return np.array([-S[1, 2], S[0, 2], -S[0, 1]])
########################################################################################################

def hat(x):
    """Convert vector to skew-symmetric matrix."""
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
########################################################################################################

def saturate(x, x_min, x_max):
    """Saturate a value between bounds."""
    if x > x_max:
        return x_max
    elif x < x_min:
        return x_min
    else:
        return x
########################################################################################################

def satdot(sigma, y, ydot):
    """Saturate derivative."""
    if y > sigma or y < -sigma:
        return 0
    else:
        return ydot
########################################################################################################

def plot_4x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    """Plot 4x1 subplots."""
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    for i in range(4):
        axs[i].plot(x, y[i, :], linetype, linewidth=linewidth)
    axs[-1].set_xlabel(xlabel_)
    fig.suptitle(title_)
    axs[1].set_ylabel(ylabel_)
    return fig, axs
########################################################################################################

def plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    """Plot 3x1 subplots."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    for i in range(3):
        axs[i].plot(x, y[i, :], linetype, linewidth=linewidth)
    axs[-1].set_xlabel(xlabel_)
    fig.suptitle(title_)
    axs[1].set_ylabel(ylabel_)
    return fig, axs
########################################################################################################

def generate_output_arrays(N):
    """Initialize output arrays."""
    error = {
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
        'R': np.zeros((3, 3, N)),
        'W': np.zeros((3, N))
    }

    R = np.zeros((3, 3, N))
    f = np.zeros(N)
    M = np.zeros((3, N))

    return error, desired, R, f, M
########################################################################################################

def deriv_unit_vector(q, q_dot, q_ddot):
    """Compute derivatives of a unit vector."""
    nq = norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * (q @ q_dot) / nq**3

    u_ddot = (q_ddot / nq - q_dot / nq**3 * (2 * q @ q_dot)
              - q / nq**3 * (q_dot @ q_dot + q @ q_ddot)
              + 3 * q / nq**5 * (q @ q_dot)**2)

    return u, u_dot, u_ddot
########################################################################################################

def command_point(t):
    """Line command generator."""
    height = 40

    desired = {
        'x': np.array([0, 20, -height]),
        'v': np.array([0, 0, 0]),
        'x_2dot': np.zeros(3),
        'x_3dot': np.zeros(3),
        'x_4dot': np.zeros(3),
        'w': 0,
        'w_dot': 0,
        'yaw': -np.pi/2,
        'b1': np.array([1, 0, 0]),
        'b1_dot': np.array([0, 0, 0]),
        'b1_2dot': np.array([0, 0, 0])
    }
    return desired
########################################################################################################

def command_line(t):
    """Line command generator."""
    height = 5
    v = 4
    w = 0

    desired = {
        'x': np.array([v*t, 0, -height]),
        'v': np.array([v, 0, 0]),
        'x_2dot': np.zeros(3),
        'x_3dot': np.zeros(3),
        'x_4dot': np.zeros(3),
        'w': w,
        'w_dot': 0,
        'yaw': w*t,
        'b1': np.array([np.cos(w*t), np.sin(w*t), 0]),
        'b1_dot': w * np.array([-np.sin(w*t), np.cos(w*t), 0]),
        'b1_2dot': w**2 * np.array([-np.cos(w*t), -np.sin(w*t), 0])
    }
    return desired
########################################################################################################

def command_circle(t):
    """Circle command generator."""
    rad = 1
    w = 2*np.pi / 10
    height = 1

    desired = {
        'x': rad * np.array([np.cos(w*t) - 1, np.sin(w*t), -height]),
        'v': w * rad * np.array([-np.sin(w*t), np.cos(w*t), 0]),
        'x_2dot': w**2 * rad * np.array([-np.cos(w*t), -np.sin(w*t), 0]),
        'x_3dot': w**3 * rad * np.array([np.sin(w*t), -np.cos(w*t), 0]),
        'x_4dot': w**4 * rad * np.array([np.cos(w*t), np.sin(w*t), 0]),
        'b1': np.array([np.cos(w*t), np.sin(w*t), 0]),
        'b1_dot': w * np.array([-np.sin(w*t), np.cos(w*t), 0]),
        'b1_2dot': w**2 * np.array([-np.cos(w*t), -np.sin(w*t), 0])
    }
    return desired
########################################################################################################

if __name__ == "__main__":
    run_brescianini()