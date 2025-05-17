import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
import matplotlib.pyplot as plt

def adaptiveGeometricController():
    # Simulation mode
    # Uncomment to use geometric adaptive decoupled-yaw controller.
    param = {}
    param['use_decoupled_controller'] = False

    # Uncomment to use geometric adaptive coupled-yaw controller in
    # Geometric adaptive tracking control of a quadrotor unmanned aerial 
    # vehicle on {SE(3)}, F. Goodarzi, D. Lee, and T. Lee.
    # param.use_decoupled_controller = False;
    
    # Disturbances
    # Uncomment to use disturbances.
    # param.use_disturbances = True;

    # Uncomment to remove disturbances.
    param['use_disturbances'] = True
    
    # Simulation parameters
    t = np.arange(0, 10.01, 0.01)
    N = len(t)

    # Quadrotor
    J1 = 0.02
    J2 = 0.02
    J3 = 0.04
    param['J'] = np.diag([J1, J2, J3])
    param['m'] = 2
    param['g'] = 9.81

    param['d'] = 0.169
    param['c_tf'] = 0.0135

    # Fixed disturbance
    # Controller gains
    k = {}
    if param['use_disturbances']:
        param['W_x'] = np.eye(3)
        param['theta_x'] = np.array([1, 0.8, -1]).reshape(-1,1)

        param['W_R'] = np.eye(3)
        param['theta_R'] = np.array([0.1, 0.1, -0.1]).reshape(-1,1)

        k['x'] = 15
        k['v'] = 9
        k['R'] = 4.1
        k['W'] = 2.9
        k['y'] = 0.8
        k['wy'] = 0.2    
    else:
        param['W_x'] = np.zeros((3,3))
        param['theta_x'] = np.zeros((3,1))

        param['W_R'] = np.eye(3)
        param['theta_R'] = np.zeros((3,1))
        
        k['x'] = 11
        k['v'] = 9
        k['R'] = 6
        k['W'] = 2
        k['y'] = 2
        k['wy'] = 0.8

    param['gamma_x'] = 1.25
    param['gamma_R'] = 0.45
    param['B_theta_x'] = 10
    
    param['c1'] = min(np.sqrt(k['x'] / param['m']), 4 * k['x'] * k['v'] / (k['v']**2 + 4 * param['m'] * k['x']))

    B2 = 1
    J1 = param['J'][0,0]
    param['c2'] = min(np.sqrt(k['R'] / J1), 4 * k['R'] * k['W'] / ((k['W'] + J1 * B2)**2 + 4 * J1 * k['R']))

    J3 = param['J'][2,2]
    param['c3'] = min(np.sqrt(k['y'] / J3), 4 * k['y'] * k['wy'] / (k['wy']**2 + 4 * J1 * k['R']))
    
    # Initial conditions
    x0 = np.array([1, -1, 0]).reshape(-1,1)  # for circle trajectory
    # x0 = np.array([0, 0, 0]).reshape(-1,1)  # for line trajectory

    v0 = np.zeros((3,1))
    e3 = np.array([0, 0, 1]).reshape(-1,1)
    R0 = expm((np.pi - 0.01) * hat(e3))
    W0 = np.zeros((3,1))

    X0 = np.vstack([x0, v0, W0, R0.reshape(-1,1), np.zeros((6,1))])
    
    # Numerical integration
    X = odeint(eom, X0.flatten(), t, args=(k, param), 
              rtol=1e-6, atol=1e-6)
    
    # Output arrays
    # Create empty arrays to save data
    e, d, R, f, M = generate_output_arrays(N)
    
    # Post processing
    x = X[:, 0:3].T
    v = X[:, 3:6].T
    W = X[:, 6:9].T
    theta_x = X[:, 18:21].T
    theta_R = X[:, 21:24].T

    b1 = np.zeros((3, N))
    b1c = np.zeros((3, N))

    thr = np.zeros((4, N))

    avg_ex = 0
    avg_eR = 0
    avg_f = 0

    converge_t = 0
    is_converged = False
    converge_ex = 0.02

    for i in range(N):
        R[:,:,i] = X[i,9:18].reshape(3,3)
        
        des = command(t[i])
        f[i], M_, _, _, err, calc = position_control(X[i,:].reshape(-1,1), des, k, param)
        # Unpack errors
        e['x'][:,i] = err['x'].flatten()
        e['v'][:,i] = err['v'].flatten()
        e['R'][:,i] = err['R'].flatten()
        e['W'][:,i] = err['W'].flatten()
        
        if param['use_decoupled_controller']:
            e['y'][i] = err['y']
            e['Wy'][i] = err['Wy']
        
        f[i], M_ = saturate_fM(f[i], M_, param)
        thr[:,i] = fM_to_thr(f[i], M_, param)
        M[:,i] = M_.flatten()
        
        # Unpack desired values
        d['x'][:,i] = des['x'].flatten()
        d['v'][:,i] = des['v'].flatten()
        d['b1'][:,i] = des['b1'].flatten()
        d['R'][:,:,i] = calc['R']
        b1[:,i] = R[:,:,i] @ np.array([1, 0, 0]).reshape(-1,1).flatten()
        b1c[:,i] = calc['b1'].flatten()
        
        norm_ex = np.linalg.norm(err['x'])
        norm_eR = np.linalg.norm(err['R'])
        
        # Find normalized errors
        avg_ex += norm_ex
        avg_eR += norm_eR
        
        norm_f = np.linalg.norm(thr[:,i])
        avg_f += norm_f
        
        if norm_ex < converge_ex:
            if not is_converged:
                converge_t = t[i]
                is_converged = True
    
    avg_ex = avg_ex / N
    avg_eR = avg_eR / N
    avg_f = avg_f / N
    
    print(f"avg_ex: {avg_ex}")
    print(f"avg_eR: {avg_eR}")
    print(f"avg_f: {avg_f}")
    print(f"converge_t: {converge_t}")

    # Plots
    linetype = 'k'
    linewidth = 1
    xlabel_ = 'time (s)'

    plt.figure(1)
    plot_3x1(t, e['R'], '', xlabel_, 'e_R', linetype, linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    plt.figure(2)
    plot_3x1(t, e['x'], '', xlabel_, 'e_x', linetype, linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    plt.figure(3)
    plot_3x1(t, x, '', xlabel_, 'x', linetype, linewidth)
    plot_3x1(t, d['x'], '', xlabel_, 'x', 'r:', linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    plt.figure(4)
    plot_3x1(t, theta_x - param['theta_x'], '', xlabel_, 
            '\\tilde\\theta_x', linetype, linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    plt.figure(5)
    plot_3x1(t, theta_R - param['theta_R'], '', xlabel_, 
            '\\tilde\\theta_R', linetype, linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    plt.figure(6)
    plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)
    # set(gca, 'FontName', 'Times New Roman');

    # Save data
    if param['use_decoupled_controller']:
        np.savez('decoupled.npz', **locals())
    else:
        np.savez('coupled.npz', **locals())

    plt.show()

def attitude_control_coupled(R, W, theta_R, Rd, Wd, Wddot, k, param):
    """
    Based on: 2010, Taeyoung Lee, Melvin Leok, and N. Harris McClamroch
    "Geometric tracking control of a quadrotor UAV on SE(3)"
    https://ieeexplore.ieee.org/abstract/document/5717652

    The control law is based on the attitude control law for rigid bodies
    with a coupling term that depends on the desired angular velocity
    and its derivative.
    """
    eR = 1/2 * vee(Rd.T @ R - R.T @ Rd)                     # (10)
    eW = W - R.T @ Rd @ Wd                                  # (11)
                                                            # (16)
    M = - k['R'] * eR \
          - k['W'] * eW \
          - param['W_R'] @ theta_R \
          + hat(R.T @ Rd @ Wd) @ param['J'] @ R.T @ Rd @ Wd \
          + param['J'] @ R.T @ Rd @ Wddot

    dot_theta_R = param['gamma_R'] * param['W_R'].T @ (eW + param['c2'] * eR)
    return M, dot_theta_R, eR, eW

def attitude_control(R, W, bar_theta_R, b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, k, param):
    """Unpack other parameters"""
    J = param['J']
    c2 = param['c2']
    c3 = param['c3']

    gamma_R = param['gamma_R']
    W_R = param['W_R']

    W_R_1 = W_R[0,:]
    W_R_2 = W_R[1,:]
    W_R_3 = W_R[2,:]
    
    """Body axes"""
    e1 = np.array([1, 0, 0]).reshape(-1,1)
    e2 = np.array([0, 1, 0]).reshape(-1,1)
    e3 = np.array([0, 0, 1]).reshape(-1,1)

    b1 = R @ e1
    b2 = R @ e2
    b3 = R @ e3
    
    """Roll/pitch dynamics"""
    kb = k['R']
    kw = k['W']

    w = W[0] * b1 + W[1] * b2                                          # (23)
    b3_dot = hat(w) @ b3                                               # (22)

    wd = hat(b3d) @ b3d_dot
    wd_dot = hat(b3d) @ b3d_ddot

    eb = hat(b3d) @ b3                                                 # (27)
    ew = w + hat(b3)@hat(b3) @ wd                                           # (28)
                                                    # (31)
    tau = - kb * eb \
              - kw * ew \
              - J[0,0] * (b3.T @ wd) * b3_dot \
              - J[0,0] * hat(b3)@hat(b3) @ wd_dot \
              - W_R_1 @ bar_theta_R * b1 - W_R_2 @ bar_theta_R * b2

    tau1 = b1.T @ tau
    tau2 = b2.T @ tau

    M1 = tau1 + J[2,2] * W[2] * W[1]                                   # (24)
    M2 = tau2 - J[2,2] * W[2] * W[0]                                   # (24)
    
    """Yaw dynamics"""
    ey = -b2.T @ b1c                                                   # (49)
    ewy = W[2] - wc3                                                   # (50)

                                             # (52)
    M3 = - k['y'] * ey \
              - k['wy'] * ewy \
              - W_R_3 @ bar_theta_R \
              + J[2,2] * wc3_dot
    
    """Attitude adaptive term"""
    ew_c2eb = ew + c2 * eb
    dot_theta_R = gamma_R * W_R_1.T * (ew_c2eb.T @ b1) \
                    + gamma_R * W_R_2.T * (ew_c2eb.T @ b2) \
                    + gamma_R * W_R_3.T * (ewy + c3 * ey)

    M = np.vstack([M1, M2, M3])
    return M, dot_theta_R.T, eb, ew, ey, ewy

def position_control(X, desired, k, param):
    x, v, R, W, bar_theta_x, bar_theta_R = split_to_states(X)

    c1 = param['c1']
    m = param['m']
    g = param['g']

    W_x = param['W_x']
    W_x_dot = np.zeros((3,3))
    W_x_2dot = np.zeros((3,3))

    e3 = np.array([0, 0, 1]).reshape(-1,1)

    error = {}
    error['x'] = x - desired['x']
    error['v'] = v - desired['v']
    A = - k['x'] * error['x'] \
          - k['v'] * error['v'] \
          - m * g * e3 \
          + m * desired['x_2dot'] \
          - W_x @ bar_theta_x
    
    gamma_x = param['gamma_x']
    c1 = param['c1']
    ev_c1ex = error['v'] + c1 * error['x']

    norm_theta_x = np.linalg.norm(bar_theta_x)
    if norm_theta_x < param['B_theta_x'] or \
        (norm_theta_x == param['B_theta_x'] and bar_theta_x.T @ W_x.T @ ev_c1ex <= 0):
        
        bar_theta_x_dot = gamma_x * W_x.T @ ev_c1ex
    else:
        I_theta = np.eye(3) - bar_theta_x @ bar_theta_x.T / (bar_theta_x.T @ bar_theta_x)
        bar_theta_x_dot = gamma_x * I_theta @ W_x.T @ ev_c1ex
    
    b3 = R @ e3
    f = -A.T @ b3
    ev_dot = g * e3 \
              - f / m * b3 \
              - desired['x_2dot'] \
              + W_x @ bar_theta_x / m
    A_dot = - k['x'] * error['v'] \
              - k['v'] * ev_dot \
              + m * desired['x_3dot'] \
              - W_x_dot @ bar_theta_x \
              - W_x @ bar_theta_x_dot
    
    norm_theta_x = np.linalg.norm(bar_theta_x)
    if norm_theta_x < param['B_theta_x'] or \
        (norm_theta_x == param['B_theta_x'] and bar_theta_x.T @ W_x.T @ ev_c1ex <= 0):
        
        bar_theta_x_2dot = gamma_x * W_x_dot.T @ ev_c1ex \
                            + gamma_x * W_x.T @ (ev_dot + c1 * error['v'])
    else:
        I_theta = np.eye(3) \
                    - bar_theta_x @ bar_theta_x.T / (bar_theta_x.T @ bar_theta_x)
        
        num = norm_theta_x * (bar_theta_x_dot @ bar_theta_x.T \
                            + bar_theta_x @ bar_theta_x_dot.T) \
                            - 2 * (bar_theta_x @ bar_theta_x.T) @ bar_theta_x_dot
        I_theta_dot = - num / norm_theta_x**3
        bar_theta_x_2dot = gamma_x * I_theta_dot @ W_x.T @ ev_c1ex \
                            + gamma_x * I_theta @ W_x_dot.T @ ev_c1ex \
                            + gamma_x * I_theta @ W_x.T @ (ev_dot + c1 * error['v'])
    
    b3_dot = R @ hat(W) @ e3
    f_dot = -A_dot.T @ b3 - A.T @ b3_dot
    ev_2dot = - f_dot / m * b3 - f / m * b3_dot - desired['x_3dot'] \
                + W_x_dot @ bar_theta_x / m + W_x @ bar_theta_x_dot / m
    A_ddot = - k['x'] * ev_dot \
              - k['v'] * ev_2dot \
              + m * desired['x_4dot'] \
              - W_x_2dot @ bar_theta_x \
              - 2 * W_x_dot @ bar_theta_x_dot \
              - W_x @ bar_theta_x_2dot
    
    b3c, b3c_dot, b3c_ddot = deriv_unit_vector(-A, -A_dot, -A_ddot)

    A2 = -hat(desired['b1']) @ b3c
    A2_dot = -hat(desired['b1_dot']) @ b3c - hat(desired['b1']) @ b3c_dot
    A2_ddot = - hat(desired['b1_2dot']) @ b3c \
                - 2 * hat(desired['b1_dot']) @ b3c_dot \
                - hat(desired['b1']) @ b3c_ddot

    b2c, b2c_dot, b2c_ddot = deriv_unit_vector(A2, A2_dot, A2_ddot)

    b1c = hat(b2c) @ b3c
    b1c_dot = hat(b2c_dot) @ b3c + hat(b2c) @ b3c_dot
    b1c_ddot = hat(b2c_ddot) @ b3c \
                + 2 * hat(b2c_dot) @ b3c_dot \
                + hat(b2c) @ b2c_ddot
    
    Rc = np.hstack([b1c, b2c, b3c])
    Rc_dot = np.hstack([b1c_dot, b2c_dot, b3c_dot])
    Rc_ddot = np.hstack([b1c_ddot, b2c_ddot, b3c_ddot])
    
    Wc = vee(Rc.T @ Rc_dot)
    Wc_dot = vee(Rc.T @ Rc_ddot - hat(Wc)@hat(Wc))
    
    W3 = (R @ e3).T @ (Rc @ Wc)
    W3_dot = (R @ e3).T @ (Rc @ Wc_dot) \
                + (R @ hat(W) @ e3).T @ (Rc @ Wc)
    
    """Run attitude controller"""
    if param['use_decoupled_controller']:
        M, theta_R_dot, error['R'], error['W'], error['y'], error['Wy'] \
            = attitude_control(
            R, W, bar_theta_R,
            b3c, b3c_dot, b3c_ddot, b1c, W3, W3_dot,
            k, param)
        
        # For comparison with non-decoupled controller
        error['R'] = 1/2 * vee(Rc.T @ R - R.T @ Rc)
    else:
        M, theta_R_dot, error['R'], error['W'] = attitude_control_coupled(
            R, W, bar_theta_R, Rc, Wc, Wc_dot, k, param)
    
    """Saving data"""
    calculated = {}
    calculated['b3'] = b3c
    calculated['b3_dot'] = b3c_dot
    calculated['b3_ddot'] = b3c_ddot
    calculated['b1'] = b1c
    calculated['R'] = Rc
    calculated['W'] = Wc
    calculated['W_dot'] = Wc_dot
    calculated['W3'] = (R @ e3).T @ (Rc @ Wc)
    calculated['W3_dot'] = (R @ e3).T @ (Rc @ Wc_dot) \
                            + (R @ hat(W) @ e3).T @ (Rc @ Wc)
    
    return f, M, bar_theta_x_dot, theta_R_dot, error, calculated

def command(t):
    # desired = command_line(t)
    return command_circle(t)

def eom(X, t, k, param):

    e3 = np.array([0, 0, 1]).reshape(-1,1)
    m = param['m']
    J = param['J']

    W_x = param['W_x']
    W_R = param['W_R']

    x, v, R, W, _, _ = split_to_states(X)

    desired = command(t)
    f, M, bar_theta_x_dot, bar_theta_R_dot, _, _ = position_control(X, desired, k, param)

    f, M = saturate_fM(f, M, param)

    xdot = v
    vdot = param['g'] * e3 \
              - f / m * R @ e3 + W_x @ param['theta_x'] / m
    Wdot = np.linalg.inv(J) @ (-hat(W) @ J @ W + M + W_R @ param['theta_R'])
    Rdot = R @ hat(W)

    if not param['use_disturbances']:
        bar_theta_x_dot = 0*bar_theta_x_dot
        bar_theta_R_dot = 0*bar_theta_R_dot

    Xdot = np.vstack([xdot, vdot, Wdot, Rdot.reshape(-1,1), 
                     bar_theta_x_dot, bar_theta_R_dot])
    return Xdot.flatten()

def fM_to_thr(f, M, param):
    d = param['d']
    ctf = param['c_tf']

    f_to_fM = np.array([
        [1, 1, 1, 1],
        [0, -d, 0, d],
        [d, 0, -d, 0],
        [-ctf, ctf, -ctf, ctf]
    ])

    thr = np.linalg.solve(f_to_fM, np.vstack([f, M]))
    return thr.flatten()

def saturate_fM(f, M, param):
    thr = fM_to_thr(f, M, param)

    max_f = 8
    min_f = 0.1

    for i in range(4):
        if thr[i] > max_f:
            thr[i] = max_f
        elif thr[i] < min_f:
            thr[i] = min_f

    return thr_to_fM(thr, param)

def split_to_states(X):
    X = X.reshape(-1,1)
    x = X[0:3]
    v = X[3:6]
    W = X[6:9]
    R = X[9:18].reshape(3,3)
    bar_theta_x = X[18:21]
    bar_theta_R = X[21:24]
    return x, v, R, W, bar_theta_x, bar_theta_R

def thr_to_fM(thr, param):
    d = param['d']
    ctf = param['c_tf']

    f_to_fM = np.array([
        [1, 1, 1, 1],
        [0, -d, 0, d],
        [d, 0, -d, 0],
        [-ctf, ctf, -ctf, ctf]
    ])

    fM = f_to_fM @ thr.reshape(-1,1)
    f = fM[0,0]
    M = fM[1:4]
    return f, M

def command_circle(t):
    rad = 1
    w = 2*np.pi / 10
    height = 1

    desired = {}
    desired['x'] = rad*np.array([np.cos(w*t) - 1, np.sin(w*t), -height]).reshape(-1,1)
    desired['v'] = w*rad*np.array([-np.sin(w*t), np.cos(w*t), 0]).reshape(-1,1)
    desired['x_2dot'] = w**2*rad*np.array([-np.cos(w*t), -np.sin(w*t), 0]).reshape(-1,1)
    desired['x_3dot'] = w**3*rad*np.array([np.sin(w*t), -np.cos(w*t), 0]).reshape(-1,1)
    desired['x_4dot'] = w**4*rad*np.array([np.cos(w*t), np.sin(w*t), 0]).reshape(-1,1)

    w = 2*np.pi / 40
    desired['b1'] = np.array([np.cos(w*t), np.sin(w*t), 0]).reshape(-1,1)
    desired['b1_dot'] = w*np.array([-np.sin(w*t), np.cos(w*t), 0]).reshape(-1,1)
    desired['b1_2dot'] = w**2*np.array([-np.cos(w*t), -np.sin(w*t), 0]).reshape(-1,1)
    return desired

def command_line(t):
    height = 5
    v = 4
    
    desired = {}
    desired['x'] = np.array([v*t, 0, -height]).reshape(-1,1)
    desired['v'] = np.array([v, 0, 0]).reshape(-1,1)
    desired['x_2dot'] = np.zeros((3,1))
    desired['x_3dot'] = np.zeros((3,1))
    desired['x_4dot'] = np.zeros((3,1))

    w = 0
    desired['w'] = w
    desired['w_dot'] = 0

    desired['yaw'] = w*t

    desired['b1'] = np.array([np.cos(w*t), np.sin(w*t), 0]).reshape(-1,1)
    desired['b1_dot'] = w * np.array([-np.sin(w*t), np.cos(w*t), 0]).reshape(-1,1)
    desired['b1_2dot'] = w**2 * np.array([-np.cos(w*t), -np.sin(w*t), 0]).reshape(-1,1)
    return desired

def deriv_unit_vector(q, q_dot, q_ddot):
    nq = np.linalg.norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * (q.T @ q_dot) / nq**3

    u_ddot = q_ddot / nq - q_dot / nq**3 * (2 * q.T @ q_dot)\
                - q / nq**3 * (q_dot.T @ q_dot + q.T @ q_ddot)\
                + 3 * q / nq**5 * (q.T @ q_dot)**2
    return u, u_dot, u_ddot

def generate_output_arrays(N):
    error = {}
    error['x'] = np.zeros((3, N))
    error['v'] = np.zeros((3, N))
    error['R'] = np.zeros((3, N))
    error['W'] = np.zeros((3, N))
    error['y'] = np.zeros(N)
    error['Wy'] = np.zeros(N)

    desired = {}
    desired['x'] = np.zeros((3, N))
    desired['v'] = np.zeros((3, N))
    desired['b1'] = np.zeros((3, N))
    desired['R'] = np.zeros((3, 3, N))

    R = np.zeros((3, 3, N))
    f = np.zeros(N)
    M = np.zeros((3, N))
    return error, desired, R, f, M

def hat(x):
    x = x.flatten()
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def vee(S):
    return np.array([-S[1,2], S[0,2], -S[0,1]]).reshape(-1,1)

def plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(x, y[i,:], linetype, linewidth=linewidth)  
        plt.grid(True)
    
    plt.xlabel(xlabel_)
    plt.title(title_)

    plt.subplot(3, 1, 2)
    plt.ylabel(f'${ylabel_}$')

def plot_4x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(x, y[i,:], linetype, linewidth=linewidth)  
        plt.grid(True)
    
    plt.xlabel(xlabel_)
    plt.title(title_)

    plt.subplot(4, 1, 2)
    plt.ylabel(f'${ylabel_}$')

if __name__ == "__main__":
    adaptiveGeometricController()