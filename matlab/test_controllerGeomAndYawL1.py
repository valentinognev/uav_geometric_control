import numpy as np
from scipy.integrate import odeint, ode
from scipy.linalg import expm, norm
from numpy import exp
import matplotlib.pyplot as plt
from copy import deepcopy

#
# Copyright (c) 2020 Flight Dynamics and Control Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
class State:
    def __init__(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.v_hat = np.zeros(3)
        self.omega_hat = np.zeros(3)
        self.omega = np.zeros(3)
        self.u_ad = np.zeros(4)
        self.u_b = np.zeros(4)
        self.sigma_m_hat = np.zeros(4)
        self.sigma_um_hat = np.zeros(2)
        self.lpf1 = np.zeros(4)
        self.lpf2 = np.zeros(4)
        self.t = 0

def test_controllerAllinOne():
    # addpath('aux_functions');
    # addpath('test_functions');

    ### Simulation parameters
    tend=10
    dt = 0.001


    # Quadrotor
    J1 = 3e-3    # 0.0000582857 # 0.02
    J2 = 1.8e-3  # 0.0000716914 # 0.02
    J3 = 3.2e-3  # 0.0001       # 0.04
    param = {
        'J': np.diag([J1, J2, J3]),
        'Jvar': 0.002*0,
        'Jbias': 0.1*0,
        'Jinv': np.linalg.inv(np.diag([J1, J2, J3])),
        'm': 0.62, # 0.075,       # 2,
        'x_delta': np.array([0.5, 0.8, -1])*0,
        'R_delta': np.array([0.2, 1.0, -0.1])*0,
        'g': 9.81,
        'L1on': True,
    }

    ### Controller gains
    k = {
        'x': np.diag(np.array([14, 15, 15])),    # np.diag(np.array([0.62, 0.45, 0.69])),  # np.diag(np.array([10, 10, 10])),
        'v': np.diag(np.array([1.5, 0.9, 1.1])), # np.diag(np.array([0.1, 0.1, 0.15])),  # 8,
        'i': 0, #10,
        'c1': 1.5,
        'sigma': 10,
        'R': np.diag(np.array([5.5, 3.5, 1.5]))*1e-1,  #np.diag(np.array([0.105, 0.105, 0.002])),  # 1.5,
        'W': np.diag(np.array([3.5, 3, 0.4]))*1e-2,  #np.diag(np.array([0.0017, 0.0017, 0.0005]))*1.3,  # 0.35,
        'I': 10*0,
        'c2': 2,
        'y': 0.8*0,
        'wy': 0.15*0,
        'yI': 2*0,
        'c3': 2,
        'Asv': -5,         # As for the velocity state
        'Asomega':-10,     # As for the rotational velocity state
        'ctoffq1Thrust': 10, # LPF1's cutoff frequency for thrust in the L1 adaptive controller (rad/s)
        'ctoffq1Moment': 10, # LPF1's cutoff frequency for moment in the L1 adaptive controller (rad/s)
        'ctoffq2Moment': 2, # LPF2's cutoff frequency for moment in the L1 adaptive controller (rad/s)
    }

    ### Initial conditions
    x0 = np.array([0, 0, 0])
    v0 = np.zeros(3)
    # R0 = expm(hat(np.array([0, 0, 0])) * np.pi/2)
    R0= np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    W0 = np.zeros(3)
    X0 = np.concatenate([x0, v0, W0, R0.flatten(), np.zeros(6)])

    ### Numerical integration
    # X = odeint(eom, X0, t, args=(k, param), rtol=1e-6, atol=1e-6, nsteps=1)
    prevState = State()
 
    def local_eom(t, X, prevState):
        Xdot, newState = eom(t, X, k, param, deepcopy(prevState))
        return Xdot
    
    r = ode(local_eom).set_integrator('dop853', rtol=1, atol=1, nsteps=10).set_initial_value(X0, 0)
    while r.successful() and r.t < tend:
        if r.t>0.1:
            pass
        r.set_f_params(prevState)
        r.integrate(r.t + dt)
        Xdot, prevState = eom(r.t, r.y, k, param, deepcopy(prevState))
        X = np.vstack((X, r.y)) if 'X' in locals() else r.y

    errcode = r.get_return_code()
    
    ### Post processing
    # Create empty arrays to save data
    N = len(X)
    e, d, R, f, M = generate_output_arrays(N)

    # Unpack the outputs of ode45 function
    x = X[:, 0:3].T
    v = X[:, 3:6].T
    W = X[:, 6:9].T
    ei = X[:, 18:21].T
    eI = X[:, 21:24].T
    t = np.linspace(0, tend, N)
    for i in range(N):
        R[:,:,i] = X[i,9:18].reshape(3, 3)

        des = command(t[i])
        f[i], M[:,i], _, _, err, calc = position_control(X[i,:], des, k, param)

        # Unpack errors
        e['x'][:,i] = err['x']
        e['v'][:,i] = err['v']
        e['R'][:,i] = err['R']
        e['W'][:,i] = err['W']
        e['y'][i] = err['y']
        e['Wy'][i] = err['Wy']

        # Unpack desired values
        d['x'][:,i] = des['x']
        d['v'][:,i] = des['v']
        d['b1'][:,i] = des['b1']
        d['R'][:,:,i] = calc['R']

    # Plot data
    linetype = 'k'
    linewidth = 1
    xlabel_ = 'time (s)'

    plot_3x1(t, e['R'], '', xlabel_, 'e_R', linetype, linewidth, fignum=1)
    # plt.gca().set(fontname='Times New Roman')

    plot_3x1(t, e['x'], '', xlabel_, 'e_x', linetype, linewidth, fignum=2)
    # plt.gca().set(fontname='Times New Roman')

    plot_3x1(t, e['v'], '', xlabel_, 'e_v', linetype, linewidth, fignum=3)
    # plt.gca().set(fontname='Times New Roman')

    plot_3x1(t, eI * (np.ones((len(t),1))*np.array([k['I'], k['I'], k['yI']])).T, '', xlabel_, 'e', linetype, linewidth, fignum=4)
    plot_3x1(t, param['R_delta'].reshape(3,1) * np.ones((3, N)), '', xlabel_, 'e_I', 'r', linewidth, fignum=4)
    # plt.gca().set(fontname='Times New Roman')

    plot_3x1(t, ei * k['i'], '', xlabel_, 'e_i', linetype, linewidth, fignum=5)
    plot_3x1(t, param['x_delta'].reshape(3,1) * np.ones((3, N)), '', xlabel_, 'e_i', 'r', linewidth, fignum=5)
    # plt.gca().set(fontname='Times New Roman')

    plot_3x1(t, x, '', xlabel_, 'x', linetype, linewidth, fignum=6)
    plot_3x1(t, d['x'], '', xlabel_, 'x', 'r', linewidth, fignum=6)
    # plt.gca().set(fontname='Times New Roman')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[0,:], x[1,:], x[2,:], 'k')
    ax.plot(d['x'][0,:], d['x'][1,:], d['x'][2,:], 'r')
    ax.set(ylabel='$x_2$', xlabel='$x_1$', zlabel='$x_3$')
    ax.set_ylim(ax.get_ylim()[::-1])  # YDir reverse
    ax.set_zlim(ax.get_zlim()[::-1])  # ZDir reverse
    ax.set_box_aspect([1,1,1])  # axis equal
    ax.grid(True)
    # plt.gca().set(fontname='Times New Roman')
    plt.show()
    print('')

def eom(t, X, k, param, prevState):
    if t>0.1:
        pass
    e3 = np.array([0, 0, 1])
    m = param['m']
    J = deepcopy(param['J'])
    jval= param['Jvar']  #*np.random.randn(1)[0]
    Jdist = np.array([[0,            jval,jval],
                      [jval,     0,            0       ],
                      [jval,     0,            0       ]])
    Jdist += np.eye(3) * param['Jbias']
    J += Jdist
    _, v, R, W, _, _ = split_to_states(X)

    desired = command(t)

    f, M, ei_dot, eI_dot, _, _ = position_control(X, desired, k, param)
    
    thrustMomentCmd = np.array([f, M[0], M[1], M[2]])
    cur_velocity_ned = v
    cur_omega = W
    
    if t>0 and param['L1on']:
        thrustMomentCmdCorrection = L1AdaptiveAugmentation(thrustMomentCmd, t, cur_velocity_ned, cur_omega, k, param, prevState)
        if np.isnan(thrustMomentCmdCorrection[0]) :
            thrustMomentCmdCorrection = L1AdaptiveAugmentation(thrustMomentCmd, t, cur_velocity_ned, cur_omega, k, param, prevState)
            print('nan')
        f = f + thrustMomentCmdCorrection[0]
        M = M + thrustMomentCmdCorrection[1:4]


    xdot = v
    vdot = param['g'] * e3 - f / m * R @ e3 + param['x_delta'] / m
    Wdot = np.linalg.solve(J, (-hat(W) @ J @ W + M + param['R_delta']))
    Rdot = R @ hat(W)
    
    if np.isnan(v[0]) or np.isnan(vdot[0]) or np.isnan(W[0]) or np.isnan(Wdot[0]):
        print('nan')

    return np.concatenate([xdot, vdot, Wdot, Rdot.flatten(), ei_dot, eI_dot]), prevState

def command(t):
    # desired = command_line(t)
    desired = command_Lissajous(t)
    # desired = command_point(t)
    return desired

def position_control(X, desired, k, param):
    """[f, M, ei_dot, eI_dot, error_, calculated] = position_control(X, desired, 
    k, param)
    
    Position controller that uses decoupled-yaw controller as the attitude
    controller
    
      Caluclates the force and moments required for a UAV to reach a given 
      set of desired position commands using a decoupled-yaw controller
      defined in https://ieeexplore.ieee.org/document/8815189.
      
      Inputs:
       X: (24x1 matrix) states of the system (x, v, R, W, ei, eI)
       desired: (struct) desired states
       k: (struct) control gains
       param: (struct) parameters such as m, g, J in a struct
    
      Outputs:
        f: (scalar) required motor force
        M: (3x1 matrix) control moment required to reach desired conditions
        ei_dot: (3x1 matrix) position integral change rate
        eI_dot: (3x1 matrix) attitude integral change rate
        error: (struct) errors for attitude and position control (for data
        logging)
        calculated: (struct) calculated desired commands (for data logging)
    """
    # Use this flag to enable or disable the decoupled-yaw attitude controller.
    use_decoupled = False

    # Unpack states
    x, v, R, W, ei, eI = split_to_states(X)

    sigma = k['sigma']
    c1 = k['c1']
    m = param['m']
    g = param['g']
    e3 = np.array([0, 0, 1])

    error_ = {}
    error_['x'] = x - desired['x']                                                # (11)
    error_['v'] = v - desired['v']                                                # (12)
    A = - k['x'] @ error_['x'] \
        - k['v'] @ error_['v'] \
        - m * g * e3 \
        + m * desired['x_2dot'] \
        - k['i'] * sat(sigma, ei)                                                  # (14)

    ei_dot = error_['v'] + c1 * error_['x']                                       # (13)
    b3 = R @ e3
    f = -np.dot(A, b3)
    ea = g * e3 \
        - f / m * b3 \
        - desired['x_2dot'] \
        + param['x_delta'] / m
    A_dot = - k['x'] @ error_['v'] \
        - k['v'] @ ea \
        + m * desired['x_3dot'] \
        - k['i'] * satdot(sigma, ei, ei_dot)                                           # (14)

    ei_ddot = ea + c1 * error_['v']
    b3_dot = R @ hat(W) @ e3                                                     # (22)
    f_dot = -np.dot(A_dot, b3) - np.dot(A, b3_dot)
    eb = - f_dot / m * b3 - f / m * b3_dot - desired['x_3dot']                   # (27)
    A_ddot = - k['x'] @ ea \
        - k['v'] @ eb \
        + m * desired['x_4dot'] \
        - k['i'] * satdot(sigma, ei, ei_ddot)

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
        + hat(b2c) @ b3c_ddot

    Rc = np.column_stack([b1c, b2c, b3c])
    Rc_dot = np.column_stack([b1c_dot, b2c_dot, b3c_dot])
    Rc_ddot = np.column_stack([b1c_ddot, b2c_ddot, b3c_ddot])

    Wc = vee(Rc.T @ Rc_dot)
    Wc_dot = vee(Rc.T @ Rc_ddot - hat(Wc)@hat(Wc))

    W3 = np.dot(R @ e3, Rc @ Wc)
    W3_dot = np.dot(R @ e3, Rc @ Wc_dot) + np.dot(R @ hat(W) @ e3, Rc @ Wc)

    ### Run attitude controller
    if use_decoupled:
        M, eI_dot, error_['b'], error_['W'], error_['y'], error_['Wy'] \
            = attitude_control_decoupled_yaw(
            R, W, eI,
            b3c, b3c_dot, b3c_ddot, b1c, W3, W3_dot,
            k, param)
        
        # Only used for comparison between two controllers
        error_['R'] = 1 / 2 * vee(Rc.T @ R - R.T @ Rc)
    else:
        M, eI_dot, error_['R'], error_['W'] = attitude_control(
            R, W, eI,
            Rc, Wc, Wc_dot,
            k, param)
        error_['y'] = 0
        error_['Wy'] = 0
    
    ### Saving data
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
    """[M, eI_dot, eR, eW] = attitude_control(R, W, eI, Rd, Wd, Wddot, k, param)
    
    Attitude controller
    
      Caluclates control moments for a given set of desired attitude commands 
      using a the controller defined in 
      https://ieeexplore.ieee.org/abstract/document/5717652
      
      Inputs:
       R: (3x3 matrix) current attitude in SO(3)
       W: (3x1 matrix) current angular velocity
       eI: (3x1 matrix) attitude integral error
       Rd: (3x3 matrix) desired attitude in SO(3)
       Wd: (3x1 matrix) desired body angular velocity
       Wd_dot: (3x1 matrix) desired body angular acceleration
       k: (struct) control gains
       param: (struct) parameters such as m, g, J in a struct
    
      Outputs:
        M: (3x1 matrix) control moment required to reach desired conditions
        eI_dot: (3x1 matrix) attitude integral change rate
        eR: (3x1 matrix) attitude error
        eW: (3x1 matrix) angular velocity error
    """
    eR = 1 / 2 * vee(Rd.T @ R - R.T @ Rd)                    # (10)
    eW = W - R.T @ Rd @ Wd                                   # (11)

    kR = k['R']   # np.diag([k['R'], k['R'], k['y']])
    kW = k['W']   # np.diag([k['W'], k['W'], k['wy']])

    M = - kR @ eR \
        - kW @ eW \
        - k['I'] * eI \
        + hat(R.T @ Rd @ Wd) @ param['J'] @ R.T @ Rd @ Wd \
        + param['J'] @ R.T @ Rd @ Wd_dot                                        # (16)

    eI_dot = eW + k['c2'] * eR

    return M, eI_dot, eR, eW

def attitude_control_decoupled_yaw(R, W, eI, b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, k, param):
    """[M, eI_dot, eb, ew, ey, ewy] = attitude_control_decoupled_yaw(R, W, eI, 
    b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, k, param)
    
    Decoupled-yaw attitude controller
    
      Caluclates control moments for a given set of desired attitude commands 
      using a decoupled-yaw controller. This function uses the controller
      defined in https://ieeexplore.ieee.org/document/8815189.
      
      Inputs:
       R: (3x3 matrix) current attitude in SO(3)
       W: (3x1 matrix) current angular velocity
       eI: (3x1 matrix) attitude integral error
       b3d: (3x1 matrix) desired direction of b3 axis
       b3d_dot: (3x1 matrix) desired direction of b3 axis
       b3d_ddot: (3x1 matrix) desired rotational rate of b3 axis
       b1c: (3x1 matrix) desired direction of b1 axis
       wc3: (3x1 matrix) desired yaw angular velocity
       wc3_dot: (3x1 matrix) desired yaw angular acceleration
       k: (struct) control gains
       param: (struct) parameters such as m, g, J in a struct
    
      Outputs:
        M: (3x1 matrix) control moment required to reach desired conditions
        eI_dot: (3x1 matrix) attitude integral change rate
        eb: (3x1 matrix) roll/pitch angle error
        ew: (3x1 matrix) roll/pitch angular velocity error
        ey: (3x1 matrix) yaw angle error
        ewy: (3x1 matrix) yaw angular velocity error
    """
    ### Unpack other parameters
    J = param['J']
    c2 = param['c2']
    c3 = param['c3']

    ### Body axes
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    b1 = R @ e1
    b2 = R @ e2
    b3 = R @ e3

    ### Roll/pitch dynamics
    kb = k['R']
    kw = k['W']

    w = W[0] * b1 + W[1] * b2                      # (23)
    b3_dot = hat(w) @ b3                           # (22)

    wd = hat(b3d) @ b3d_dot
    wd_dot = hat(b3d) @ b3d_ddot

    eb = hat(b3d) @ b3                             # (27)
    ew = w + hat(b3)@hat(b3) @ wd                 # (28)
    tau = - kb * eb \
        - kw * ew \
        - J[0,0] * np.dot(b3, wd) * b3_dot \
        - J[0,0] * hat(b3)@hat(b3) @ wd_dot \
        - k['I'] * eI[0] * b1 - k['I'] * eI[1] * b2                            # (31)

    tau1 = np.dot(b1, tau)             
    tau2 = np.dot(b2, tau)

    M1 = tau1 + J[2,2] * W[2] * W[1]               # (24)              
    M2 = tau2 - J[2,2] * W[2] * W[0]               # (24)

    ### Yaw dynamics
    ey = -np.dot(b2, b1c)                          # (49)
    ewy = W[2] - wc3                               # (50)

    M3 = - k['y'] * ey \
        - k['wy'] * ewy \
        - k['yI'] * eI[2] \
        + J[2,2] * wc3_dot                         # (52)

    eI_dot = np.array([
        np.dot(b1, (c2 * eb + ew)),
        np.dot(b2, (c2 * eb + ew)),
        c3 * ey + ewy
    ])

    return np.array([M1, M2, M3]), eI_dot, eb, ew, ey, ewy

def L1AdaptiveAugmentation(thrustMomentCmd, t, cur_velocity_ned, cur_omega, k, param, prevState):
    # state predictor
    dt = t - prevState.t
    if dt <= 1e-6:
        return np.zeros(4)  # return zero if time is not valid
    v_hat = np.zeros(3)          # state predictor value of translational speed
    omega_hat = np.zeros(3)      # state predictor value of rotational speed
    e3 = np.array([0, 0, 1])     # unit vector

    # load translational velocity
    v_now = deepcopy(cur_velocity_ned)
    # coefficient for As
    As_v = k['Asv']
    As_omega = k['Asomega']

    # load rotational velocity
    omega_now = deepcopy(cur_omega)

    massInverse = 1.0 / param['m']

    # compute prediction error (on previous step)
    vpred_error_prev = prevState.v_hat - prevState.v
    omegapred_error_prev = prevState.omega_hat - prevState.omega

    # NOTE: per the definition in vector3.h, vector * scalar must have vector first then multiply the scalar later
    v_hat = (prevState.v_hat + 
            (e3 * param['g'] - 
                prevState.R[:, 2] * (prevState.u_b[0] + prevState.u_ad[0] + prevState.sigma_m_hat[0]) * massInverse + 
                prevState.R[:, 0] * prevState.sigma_um_hat[0] * massInverse + 
                prevState.R[:, 1] * prevState.sigma_um_hat[1] * massInverse + 
                vpred_error_prev * As_v) * dt)

    # temp vector: thrustMomentCmd[1--3] + prevState.u_ad[1--3] + sigma_m_hat[1--3]
    tempVec = prevState.u_b[1:4] + prevState.u_ad[1:4] + prevState.sigma_m_hat[1:4]
    
    omega_hat = prevState.omega_hat + \
                (-param['Jinv'] @ (np.cross(prevState.omega, param['J'] @ prevState.omega)) + \
                    param['Jinv'] @ tempVec + \
                    omegapred_error_prev * As_omega) * dt

    # update the state prediction storage
    prevState.v_hat = deepcopy(v_hat)
    prevState.omega_hat = deepcopy(omega_hat)

    # compute prediction error (for this step)
    vpred_error = v_hat - v_now
    omegapred_error = omega_hat - omega_now

    # exponential coefficients for As
    exp_As_v_dt = exp(As_v * dt)
    exp_As_omega_dt = exp(As_omega * dt)

    # later part of uncertainty estimation (piecewise constant)
    PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
    PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

    sigma_m_hat = np.zeros(4)  # estimated matched uncertainty
    sigma_m_hat_2to4 = np.zeros(3)  # second to fourth element of the estimated matched uncertainty
    sigma_um_hat = np.zeros(2)  # estimated unmatched uncertainty

    # use the rotation matrix in the current step
    # In Python, we'd need a proper quaternion to rotation matrix conversion
    # For now, we'll use the previous rotation matrix as a placeholder
    R = deepcopy(prevState.R)

    sigma_m_hat[0] = np.dot(R[:, 2], PhiInvmu_v) * param['m']
    sigma_m_hat_2to4 = -param['J'] @ PhiInvmu_omega
    sigma_m_hat[1:4] = sigma_m_hat_2to4

    sigma_um_hat[0] = -np.dot(R[:, 0], PhiInvmu_v) * param['m']
    sigma_um_hat[1] = -np.dot(R[:, 1], PhiInvmu_v) * param['m']

    # store uncertainty estimations
    prevState.sigma_m_hat = deepcopy(sigma_m_hat)
    prevState.sigma_um_hat = deepcopy(sigma_um_hat)

    # compute lpf1 coefficients
    lpf1_coefficientThrust1 = exp(-k['ctoffq1Thrust'] * 0.0025)
    lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

    lpf1_coefficientMoment1 = exp(-k['ctoffq1Moment'] * 0.0025)
    lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

    # update the adaptive control
    u_ad_int = np.zeros(4)
    u_ad = np.zeros(4)

    # low-pass filter 1 (negation is added to prevState.u_ad to filter the correct signal)
    u_ad_int = lpf1_coefficientThrust1 * prevState.lpf1 + lpf1_coefficientThrust2 * sigma_m_hat

    prevState.lpf1 = deepcopy(u_ad_int)  # store the current state

    lpf2_coefficientMoment1 = exp(-k['ctoffq2Moment'] * 0.0025)
    lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

    # low-pass filter 2 (optional)
    u_ad[0] = u_ad_int[0]  # only one filter on the thrust channel
    u_ad[1:4] = lpf2_coefficientMoment1 * prevState.lpf2[1:4] + lpf2_coefficientMoment2 * u_ad_int[1:4]

    prevState.lpf2 = deepcopy(u_ad)  # store the current state
    # negate
    # u_ad = -u_ad

    # store the values for next iteration
    prevState.u_ad = deepcopy(u_ad)

    prevState.v = deepcopy(v_now)
    prevState.omega = deepcopy(omega_now)
    prevState.R = deepcopy(R)
    prevState.u_b = deepcopy(thrustMomentCmd)
    prevState.t = deepcopy(t)
    if np.isnan(prevState.u_ad[0]) or np.isnan(prevState.u_ad[1]) or np.isnan(prevState.u_ad[2]) or np.isnan(prevState.u_ad[3]):
        print('nan')

    return deepcopy(prevState.u_ad)  # return the updated l1 control







def deriv_unit_vector(q, q_dot, q_ddot):
    nq = norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * np.dot(q, q_dot) / nq**3

    u_ddot = q_ddot / nq - q_dot / nq**3 * (2 * np.dot(q, q_dot))\
        - q / nq**3 * (np.dot(q_dot, q_dot) + np.dot(q, q_ddot))\
        + 3 * q / nq**5 * np.dot(q, q_dot)**2

    return u, u_dot, u_ddot

def hat(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def vee(S):
    return np.array([-S[1,2], S[0,2], -S[0,1]])

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
        if y[k] > sigma:
            z[k] = 0
        elif y[k] < -sigma:
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

def true_data(t):
    w = 1
    ship_x = 3 * np.sin(w*t)
    ship_y = 2 * np.cos(w*t)
    ship_z = 0

    uav_wrt_ship_x = 0.1 * np.cos(5*np.pi*t)
    uav_wrt_ship_y = 0.1 * np.sin(5*np.pi*t)
    uav_wrt_ship_z = 1.0 * np.sin(2*t)

    uav_x = ship_x + uav_wrt_ship_x
    uav_y = ship_y + uav_wrt_ship_y
    uav_z = ship_z + uav_wrt_ship_z

    x_ship = np.array([ship_x, ship_y, ship_z])
    x_uav = np.array([uav_x, uav_y, uav_z])
    x_us = np.array([uav_wrt_ship_x, uav_wrt_ship_y, uav_wrt_ship_z])

    return x_ship, x_uav, x_us

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

def plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10, fignum=1):
    plt.figure(fignum)
    fig, axs = plt.subplots(3, 1)
    for i in range(3):
        axs[i].plot(x, y[i,:], linetype, linewidth=linewidth)
        # axs[i].set(fontname='Times New Roman', fontsize=font_size)
    
    axs[0].set(title=title_)
    axs[2].set(xlabel=xlabel_)
    axs[1].set(ylabel=ylabel_)

def plot_3x3(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10, desired=False):
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            k = 3*i + j
            if desired:
                axs[i,j].plot(x, y[i,j,:], linetype, linewidth=linewidth, color='r')
            else:
                axs[i,j].plot(x, y[i,j,:], linetype, linewidth=linewidth)
            # axs[i,j].set(ylim=(-1,1), fontname='Times New Roman', fontsize=font_size)
    
    axs[0,1].set(title=title_)
    axs[2,1].set(xlabel=xlabel_)
    axs[1,0].set(ylabel=ylabel_)

def command_Lissajous(t):
    A = 1
    B = 1
    C = 0.2

    d = np.pi / 2 * 0

    a = 1
    b = 2
    c = 2
    alt = -1

    desired = {
        'x': np.array([A * np.sin(a * t + d), B * np.sin(b * t), alt + C * np.cos(c * t)]),
        'v': np.array([A * a * np.cos(a * t + d), B * b * np.cos(b * t), C * c * -np.sin(c * t)]),
        'x_2dot': np.array([A * a**2 * -np.sin(a * t + d), B * b**2 * -np.sin(b * t), C * c**2 * -np.cos(c * t)]),
        'x_3dot': np.array([A * a**3 * -np.cos(a * t + d), B * b**3 * -np.cos(b * t), C * c**3 * np.sin(c * t)]),
        'x_4dot': np.array([A * a**4 * np.sin(a * t + d), B * b**4 * np.sin(b * t), C * c**4 * np.cos(c * t)]),
    }

    w = 2 * np.pi / 10
    desired['b1'] = np.array([np.cos(w * t), np.sin(w * t), 0])
    desired['b1_dot'] = w * np.array([-np.sin(w * t), np.cos(w * t), 0])
    desired['b1_2dot'] = w**2 * np.array([-np.cos(w * t), -np.sin(w * t), 0])

    return desired

def command_line(t):
    height = 0

    desired = {
        'x': np.array([0.5 * t, 0, -height]),
        'v': np.array([0.5, 0, 0]),
        'x_2dot': np.zeros(3),
        'x_3dot': np.zeros(3),
        'x_4dot': np.zeros(3),
    }

    w = 2 * np.pi / 10
    desired['b1'] = np.array([np.cos(w * t), np.sin(w * t), 0])
    desired['b1_dot'] = w * np.array([-np.sin(w * t), np.cos(w * t), 0])
    desired['b1_2dot'] = w**2 * np.array([-np.cos(w * t), -np.sin(w * t), 0])

    return desired

def command_point(t):
    height = 0

    desired = {
        'x': np.array([0, 20, -40]),
        'v': np.array([0, 0, 0]),
        'x_2dot': np.zeros(3),
        'x_3dot': np.zeros(3),
        'x_4dot': np.zeros(3),
    }

    w = 2 * np.pi / 10
    desired['b1'] = np.array([1,0,0])
    desired['b1_dot'] = np.array([0,0,0])
    desired['b1_2dot'] = np.array([0,0,0])

    return desired

if __name__ == "__main__":
    test_controllerAllinOne()
    plt.show()
    i=1