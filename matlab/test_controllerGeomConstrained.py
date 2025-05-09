import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 31 August 2015
# driver for constrained attitude control

def coupled_constrained_control():
    # add path to utilities
    constants = {
        'scenario': 'single',  # or 'single'
        'avoid_switch': 'true',
        'dist_switch': 'true',
        'adaptive_switch': 'true',
        
        # ACC/IJCAS Simulation for Fig 2 is
        # constants.scenario = 'multiple'; % or 'single'
        # constants.avoid_switch = 'true';
        # constants.dist_switch = 'true';
        # constants.adaptive_switch = 'false';

        # constants for plotting/animations
        'animation_type': 'none',  # or 'movie' or 'none'
        'filename': 'multiple_avoid',

        # define constants/properties of rigid body
        'm_sc': 1
    }
    m_sc = constants['m_sc']

    ##########################################################################
    # INERTIA TENSOR
    ##########################################################################

    # constants.J = [1.059e-2 -5.156e-6 2.361e-5;...
    #                -5.156e-6 1.059e-2 -1.026e-5;
    #                2.361e-5 -1.026e-5 1.005e-2];
    # Chris's Hexrotor inertia matrix
    constants['J'] = np.array([
        [55710.50413e-7, 617.6577e-7, -250.2846e-7],
        [617.6577e-7, 55757.4605e-7, 100.6760e-7],
        [-250.2846e-7, 100.6760e-7, 105053.7595e-7]
    ])

    # % % from Farhad ASME paper
    # constants.J = [ 5.5711 0.0618 -0.0251; ...
    #                 0.06177 5.5757 0.0101;...
    #                 -0.02502 0.01007 1.05053] * 1e-2;

    # constants.J = diag([694 572 360]);

    J = constants['J']

    ##########################################################################
    # CONTROLLER
    ##########################################################################
    # controller parameters
    constants['G'] = np.diag([0.9, 1, 1.1])

    # con = -1+2.*rand(3,constants.num_con); % inertial frame vectors (3XN)
    # from [1] U. Lee and M. Mesbahi. Spacecraft Reorientation in Presence of Attitude Constraints via Logarithmic Barrier Potentials. In 2011 AMERICAN CONTROL CONFERENCE, Proceedings of the American Control Conference, pages 450?455, 345 E 47TH ST, NEW YORK, NY 10017 USA, 2011. Boeing; Bosch; Corning; Eaton; GE Global Res; Honeywell; Lockheed Martin; MathWorks; Natl Instruments; NT-MDT; United Technol, IEEE. American Control Conference (ACC), San Fransisco, CA, JUN 29-JUL 01, 2011.
    # con = [0.174    0   -0.853 -0.122;...
    #     -0.934   0.7071    0.436 -0.140;...
    #     -0.034   0.7071   -0.286 -0.983];
    # column vectors to define constraints
    # zeta = 0.7;
    # wn = 0.2;
    # constants.kp = wn^2;
    # constants.zeta = 2*zeta*wn;
    # constants.kp = 0.0424; % wn^2
    # constants.kp = 0.4;
    # constants.kv = 0.296; % 2*zeta*wn
    constants['kp'] = 0.4
    constants['kv'] = 0.296  # 2*zeta*wn

    ##########################################################################
    # CONSTRAINT
    ##########################################################################

    constants['sen'] = np.array([1, 0, 0])  # body fixed frame
    # define a number of constraints to avoids

    if constants['scenario'] == 'multiple':
        con = np.array([
            [0.174, 0.4, -0.853, -0.122],
            [-0.934, 0.7071, 0.436, -0.140],
            [-0.034, 0.7071, -0.286, -0.983]
        ])
        constants['con_angle'] = np.array([40, 40, 40, 20]) * np.pi/180
    elif constants['scenario'] == 'single':
        con = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        constants['con_angle'] = np.array([12]) * np.pi/180
    
    # normalize
    constants['con'] = con / np.sqrt(np.sum(con**2, axis=0)) if con.ndim > 1 else con / np.linalg.norm(con)
    constants['alpha'] = 15  # use the same alpha for each one
    constants['num_con'] = constants['con'].shape[1] if constants['con'].ndim > 1 else 1

    ##########################################################################
    # ADAPTIVE CONTROL FOR DISTURBANCE
    ##########################################################################
    # disturbance terms

    constants['W'] = np.eye(3)
    constants['delta'] = lambda t: 0.2 + 0.02 * np.array([
        np.sin(9*t),
        np.cos(9*t),
        0.5*(np.sin(9*t) + np.cos(9*t))
    ])
    constants['kd'] = 0.5  # adaptive controller gain term (rate of convergence)
    constants['c'] = 1  # input the bound on C here

    ##########################################################################
    # DESIRED/INITIAL CONDITION
    ##########################################################################

    # define the initial state of rigid body

    # R0 = ROT1(0*pi/180)*ROT2(45*pi/180)*ROT3(180*pi/180);
    constants['q0'] = np.array([-0.188, -0.735, -0.450, -0.471])
    # constants.qd = [-0.59 0.67 0.21 -0.38]; % from lee/meshbahi paper
    constants['qd'] = np.array([0, 0, 0, 1])

    # constants.R0 = quat2dcm(constants.q0)';
    # constants.Rd = quat2dcm(constants.qd)';

    if constants['scenario'] == 'multiple':
        constants['R0'] = ROT1(0*np.pi/180) @ ROT3(225*np.pi/180)  # avoid multiple constraints
        constants['Rd'] = np.eye(3)
    elif constants['scenario'] == 'single':
        constants['R0'] = ROT1(0*np.pi/180) @ ROT3(0*np.pi/180)  # avoid single constraint
        constants['Rd'] = ROT3(90*np.pi/180)

    R0 = constants['R0']
    w0 = np.zeros(3)
    delta_est0 = np.zeros(3)
    initial_state = np.concatenate([R0.flatten(), w0, delta_est0])

    # simulation timespan
    tspan = np.linspace(0, 20, 1000)

    # propagate a chief and deputy spacecraft (continuous time system)
    state = odeint(dynamics, initial_state, tspan, args=(constants,))
    
    # extract out the states
    R_b2i = np.zeros((len(tspan), 3, 3))
    u_f = np.zeros((3, len(tspan)))
    u_m = np.zeros((3, len(tspan)))
    R_des = np.zeros((len(tspan), 3, 3))
    ang_vel_des = np.zeros((3, len(tspan)))
    ang_vel_dot_des = np.zeros((3, len(tspan)))
    Psi = np.zeros(len(tspan))
    err_att = np.zeros((3, len(tspan)))
    err_vel = np.zeros((3, len(tspan)))

    ang_vel = state[:, 9:12]
    delta_est = state[:, 12:15]
    
    for ii in range(len(tspan)):
        R_b2i[ii] = state[ii, :9].reshape(3, 3)
        
        (u_f[:, ii], u_m[:, ii], R_des[ii], ang_vel_des[:, ii], 
         ang_vel_dot_des[:, ii], Psi[ii], err_att[:, ii], err_vel[:, ii]) = controller(tspan[ii], state[ii], constants)

    num_figs = 8
    fig_handle = [plt.figure(figsize=(10, 7.5)) for _ in range(num_figs)]

    fontsize = 18
    fontname = 'Times New Roman'
    fig_size = [680, 224, 800, 600]

    # plot the attitude error vector
    fig = fig_handle[0]
    fig.suptitle('Attitude error vector')
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax1.set_ylabel('$e_{R_1}$', fontsize=fontsize, fontname=fontname)
    ax1.grid(True)
    ax1.plot(tspan, err_att[0, :])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax2.set_ylabel('$e_{R_2}$', fontsize=fontsize, fontname=fontname)
    ax2.grid(True)
    ax2.plot(tspan, err_att[1, :])

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax3.set_ylabel('$e_{R_3}$', fontsize=fontsize, fontname=fontname)
    ax3.grid(True)
    ax3.plot(tspan, err_att[2, :])

    # plot the attitude error Psi
    fig = fig_handle[1]
    fig.suptitle('$\Psi$ error')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('$\Psi$', fontsize=fontsize, fontname=fontname)
    ax.grid(True)
    ax.plot(tspan, Psi)

    # plot the angular velocity error
    fig = fig_handle[2]
    fig.suptitle('Angular velocity error vector')
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax1.set_ylabel('$e_{\Omega_1}$', fontsize=fontsize, fontname=fontname)
    ax1.grid(True)
    ax1.plot(tspan, err_vel[0, :])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax2.set_ylabel('$e_{\Omega_2}$', fontsize=fontsize, fontname=fontname)
    ax2.grid(True)
    ax2.plot(tspan, err_vel[1, :])

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax3.set_ylabel('$e_{\Omega_3}$', fontsize=fontsize, fontname=fontname)
    ax3.grid(True)
    ax3.plot(tspan, err_vel[2, :])

    # plot the control input
    fig = fig_handle[3]
    fig.suptitle('Control Input')
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax1.set_ylabel('$u_{1}$', fontsize=fontsize, fontname=fontname)
    ax1.grid(True)
    ax1.plot(tspan, u_m[0, :])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax2.set_ylabel('$u_{2}$', fontsize=fontsize, fontname=fontname)
    ax2.grid(True)
    ax2.plot(tspan, u_m[1, :])

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax3.set_ylabel('$u_{3}$', fontsize=fontsize, fontname=fontname)
    ax3.grid(True)
    ax3.plot(tspan, u_m[2, :])

    # plot the desired and actual angular velocities
    fig = fig_handle[4]
    fig.suptitle('Angular Velocity')
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax1.set_ylabel('$\Omega_{1}$', fontsize=fontsize, fontname=fontname)
    ax1.grid(True)
    ax1.plot(tspan, ang_vel[:, 0], 'b')
    ax1.plot(tspan, ang_vel_des[0, :], 'r')
    ax1.legend(['Actual', 'Desired'])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax2.set_ylabel('$\Omega_{2}$', fontsize=fontsize, fontname=fontname)
    ax2.grid(True)
    ax2.plot(tspan, ang_vel[:, 1], 'b')
    ax2.plot(tspan, ang_vel_des[1, :], 'r')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax3.set_ylabel('$\Omega_{3}$', fontsize=fontsize, fontname=fontname)
    ax3.grid(True)
    ax3.plot(tspan, ang_vel[:, 2], 'b')
    ax3.plot(tspan, ang_vel_des[2, :], 'r')

    # plot the disturbance estimate
    fig = fig_handle[5]
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('$\Delta$', fontsize=fontsize, fontname=fontname)
    ax.grid(True)
    ax.plot(tspan, delta_est)

    # plot attitude on unit sphere
    fig = fig_handle[6]
    fig.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')

    # create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # plot the sphere
    ax.plot_surface(x, y, z, color=[0.8, 0.8, 0.8], alpha=0.3, linewidth=0)

    # loop over the number of constraints and draw them all
    for ii in range(constants['num_con']):
        H = np.cos(constants['con_angle'][ii])  # height
        R = np.sin(constants['con_angle'][ii])  # chord length
        N = 100  # number of points to define the circumference
        
        # create cylinder
        theta = np.linspace(0, 2*np.pi, N)
        x_cyl = np.zeros((2, N))
        y_cyl = np.array([np.zeros(N), np.ones(N)*R])
        z_cyl = np.array([np.zeros(N), np.ones(N)*H])
        
        # rotate cylinder to point along constraint vector
        con_vec = constants['con'][:, ii] if constants['con'].ndim > 1 else constants['con']
        
        if np.allclose(con_vec, [0, 0, 1]):
            dcm = np.eye(3)
        elif np.allclose(con_vec, [0, 0, -1]):
            dcm = ROT1(np.pi)
        else:
            k_hat = np.cross([0, 0, 1], con_vec)
            angle = np.arccos(np.dot([0, 0, 1], con_vec))
            k_hat_skew = skew_matrix(k_hat)
            dcm = np.eye(3) + k_hat_skew + k_hat_skew @ k_hat_skew * (1 - np.cos(angle)) / (np.sin(angle)**2)
        
        # rotate cylinder points
        cyl_points = dcm @ np.vstack([x_cyl[1], y_cyl[1], z_cyl[1]])
        x_cyl[1] = cyl_points[0]
        y_cyl[1] = cyl_points[1]
        z_cyl[1] = cyl_points[2]
        
        # plot cylinder
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='red', alpha=0.5, linewidth=0)

    # convert the body fixed vector to the inertial frame
    sen_inertial = np.zeros((len(tspan), 3))
    for ii in range(len(tspan)):
        sen_inertial[ii] = R_b2i[ii] @ constants['sen']
    
    sen_inertial_start = constants['R0'] @ constants['sen']
    sen_inertial_end = constants['Rd'] @ constants['sen']
    
    # plot path of body vector in inertial frame
    ax.plot([sen_inertial_start[0]], [sen_inertial_start[1]], [sen_inertial_start[2]], 'go', markersize=10, linewidth=2)
    ax.plot([sen_inertial_end[0]], [sen_inertial_end[1]], [sen_inertial_end[2]], 'gx', markersize=10, linewidth=2)
    ax.plot(sen_inertial[:, 0], sen_inertial[:, 1], sen_inertial[:, 2], 'b', linewidth=3)

    # plot inertial frame
    ax.plot([0, 1], [0, 0], [0, 0], 'k', linewidth=3)
    ax.text(1.05, 0, 0, 'X', color='k')
    ax.plot([0, 0], [0, 1], [0, 0], 'k', linewidth=3)
    ax.text(0, 1.05, 0, 'Y', color='k')
    ax.plot([0, 0], [0, 0], [0, 1], 'k', linewidth=3)
    ax.text(0, 0, 1.05, 'Z', color='k')

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=28, azim=58)

    # plot the angle to each constraint
    fig = fig_handle[7]
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    
    # calculate the angle to each constraint
    ang_con = np.zeros((len(tspan), constants['num_con']))
    for ii in range(len(tspan)):
        for jj in range(constants['num_con']):
            con_vec = constants['con'][:, jj] if constants['con'].ndim > 1 else constants['con']
            ang_con[ii, jj] = 180/np.pi * np.arccos(np.dot(sen_inertial[ii], con_vec))
    
    for jj in range(constants['num_con']):
        ax.plot(tspan, ang_con[:, jj])
    
    ax.set_xlabel('$t (sec)$', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('arc$\cos \,(r^T R^T v_i)$', fontsize=fontsize, fontname=fontname)

    plt.show()

def ROT1(angle):
    """Rotation matrix about first axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def ROT2(angle):
    """Rotation matrix about second axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def ROT3(angle):
    """Rotation matrix about third axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def skew_matrix(vec):
    """Skew symmetric matrix from vector"""
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

def vee_map(mat):
    """Vee map from skew symmetric matrix to vector"""
    return 0.5 * np.array([mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]])

def dynamics(state, t, constants):
    """ODE function for constrained attitude stabilization"""
    # constants
    m_sc = constants['m_sc']
    J = constants['J']
    kd = constants['kd']
    W = constants['W']

    # redefine the state vector
    R = state[:9].reshape(3, 3)  # rotation matrix from body to inertial frame
    ang_vel = state[9:12]
    delta_est = state[12:15]  # adaptive control term to estimate fixed disturbance

    # calculate external force and moment
    _, m = ext_force_moment(t, state, constants)

    _, u_m, _, _, _, _, err_att, err_vel = controller(t, state, constants)

    # differential equations of motion
    R_dot = R @ skew_matrix(ang_vel)
    ang_vel_dot = np.linalg.inv(J) @ (m + u_m - np.cross(ang_vel, J @ ang_vel))
    theta_est_dot = kd * W.T @ (err_vel + constants['c'] * err_att)

    # output the state derivative
    return np.concatenate([R_dot.flatten(), ang_vel_dot, theta_est_dot])

def ext_force_moment(t, state, constants):
    """External force and moment calculation"""
    # redefine the state vector
    R = state[:9].reshape(3, 3)  # rotation matrix from body to inertial frame
    ang_vel = state[9:12]

    # constants
    m_sc = constants['m_sc']
    J = constants['J']
    W = constants['W']
    delta = constants['delta']

    # calculate external moment and force
    f = np.zeros(3)

    # add a constant disturbance
    if constants['dist_switch'] == 'true':
        m = np.zeros(3) + W @ delta(t)
    else:
        m = np.zeros(3)
    
    return f, m

def controller(t, state, constants):
    """Controller implementation"""
    # redefine the state vector
    R = state[:9].reshape(3, 3)  # rotation matrix from body to inertial frame
    ang_vel = state[9:12]
    delta_est = state[12:15]

    # extract out constants
    J = constants['J']
    G = constants['G']
    kp = constants['kp']
    kv = constants['kv']
    sen = constants['sen']
    alpha = constants['alpha']
    con_angle = constants['con_angle']
    con = constants['con']
    W = constants['W']

    # desired attitude
    R_des, ang_vel_des, ang_vel_dot_des = des_attitude(t, constants)

    # attitude error function
    psi_attract = 0.5 * np.trace(G @ (np.eye(3) - R_des.T @ R))      # (14)
    dA = 0.5 * vee_map(G @ R_des.T @ R - R.T @ R_des @ G)            # (16)

    if constants['avoid_switch'] == 'true':  # add the avoidance term
        sen_inertial = R @ sen

        # loop over the constraints and form a bunch of repelling function
        psi_avoid = np.zeros(constants['num_con'])
        dB = np.zeros((3, constants['num_con']))
        
        for ii in range(constants['num_con']):
            con_vec = con[:, ii] if con.ndim > 1 else con
            # calculate error function
            psi_avoid[ii] = -1/alpha * np.log((np.cos(con_angle[ii]) - np.dot(sen_inertial, con_vec)) / (1 + np.cos(con_angle[ii])))
            dB[:, ii] = 1/alpha / (np.dot(sen_inertial, con_vec) - np.cos(con_angle[ii])) * skew_matrix(R.T @ con_vec) @ sen

        Psi = psi_attract * (np.sum(psi_avoid) + 1)
        err_att = dA * (np.sum(psi_avoid) + 1) + np.sum(dB * psi_attract, axis=1)
    else:
        err_att = dA
        Psi = psi_attract

    err_vel = ang_vel - R.T @ R_des @ ang_vel_des
    alpha_d = -skew_matrix(ang_vel) @ R.T @ R_des @ ang_vel_des + R.T @ R_des @ ang_vel_dot_des

    # compute the control input
    u_f = np.zeros(3)
    
    if constants['adaptive_switch'] == 'true':
        u_m = -kp * err_att - kv * err_vel + np.cross(ang_vel, J @ ang_vel) - W @ delta_est
    else:
        u_m = -kp * err_att - kv * err_vel + np.cross(ang_vel, J @ ang_vel)

    return u_f, u_m, R_des, ang_vel_des, ang_vel_dot_des, Psi, err_att, err_vel

def des_attitude(t, constants):
    """Desired attitude generation"""
    # use 3-2-1 euler angle sequence for the desired body to inertial attitude trajectory
    a = 2*np.pi/(20/10)
    b = np.pi/9

    phi = b * np.sin(a*t)  # third rotation
    theta = b * np.cos(a*t)  # second rotation
    psi = 0  # first rotation

    phi_d = b * a * np.cos(a*t)
    theta_d = -b * a * np.sin(a*t)
    psi_d = 0

    phi_dd = -b * a**2 * np.sin(a*t)
    theta_dd = -b * a**2 * np.cos(a*t)
    psi_dd = 0

    # euler 3-2-1 sequence
    R_des = constants['Rd']
    
    # convert the euler angle sequence to the desired angular velocity vector
    ang_vel_des = np.zeros(3)
    ang_vel_dot_des = np.zeros(3)

    return R_des, ang_vel_des, ang_vel_dot_des

def body1312dcm(theta):
    st1 = np.sin(theta[0])
    st2 = np.sin(theta[1])
    st3 = np.sin(theta[2])
    ct1 = np.cos(theta[0])
    ct2 = np.cos(theta[1])
    ct3 = np.cos(theta[2])

    dcm = np.zeros((3, 3))

    dcm[0, 0] = ct2
    dcm[0, 1] = -st2 * ct3
    dcm[0, 2] = st2 * st3
    dcm[1, 0] = ct1 * st2
    dcm[1, 1] = ct1 * ct2 * ct3 - st3 * st1
    dcm[1, 2] = -ct1 * ct2 * st3 - ct3 * st1
    dcm[2, 0] = st1 * st2
    dcm[2, 1] = st1 * ct2 * ct3 + st3 * ct1
    dcm[2, 2] = -st1 * ct2 * st3 + ct3 * ct1

    return dcm

def body131dot(theta, w):
    c2 = np.cos(theta[:, 1])
    c3 = np.cos(theta[:, 2])
    s2 = np.sin(theta[:, 1])
    s3 = np.sin(theta[:, 2])

    w1 = w[:, 0]
    w2 = w[:, 1]
    w3 = w[:, 2]

    theta_d = np.zeros_like(theta)
    theta_d[:, 0] = (-w2 * c3 + w3 * s3) / s2
    theta_d[:, 1] = w2 * s3 + w3 * c3
    theta_d[:, 2] = w1 + (w2 * c3 - w3 * s3) * c2 / s2

    return theta_d

def body313dot(theta, w):
    c2 = np.cos(theta[:, 1])
    c3 = np.cos(theta[:, 2])
    s2 = np.sin(theta[:, 1])
    s3 = np.sin(theta[:, 2])

    w1 = w[:, 0]
    w2 = w[:, 1]
    w3 = w[:, 2]

    theta_d = np.zeros_like(theta)
    theta_d[:, 0] = (w1 * s3 + w2 * c3) / s2
    theta_d[:, 1] = w1 * c3 - w2 * s3
    theta_d[:, 2] = w3 - (w1 * s3 + w2 * c3) * c2 / s2

    return theta_d

def dcm2body121(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 0], -dcm[2, 0])
    theta[1] = np.arccos(dcm[0, 0])
    theta[2] = np.arctan2(dcm[0, 1], dcm[0, 2])
    return theta

def dcm2body123(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[1, 2], dcm[2, 2])
    theta[1] = np.arcsin(dcm[0, 2])
    theta[2] = np.arctan2(-dcm[0, 1], dcm[0, 0])
    return theta

def dcm2body131(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 0], dcm[1, 0])
    theta[1] = np.arccos(dcm[0, 0])
    theta[2] = np.arctan2(dcm[0, 2], -dcm[0, 1])
    return theta

def dcm2body132(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 1], dcm[1, 1])
    theta[1] = np.arcsin(-dcm[0, 1])
    theta[2] = np.arctan2(dcm[0, 2], dcm[0, 0])
    return theta

def dcm2body212(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 1], dcm[2, 1])
    theta[1] = np.arccos(dcm[1, 1])
    theta[2] = np.arctan2(dcm[1, 0], -dcm[1, 2])
    return theta

def dcm2body213(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 2], dcm[2, 2])
    theta[1] = np.arcsin(-dcm[1, 2])
    theta[2] = np.arctan2(dcm[1, 0], dcm[1, 1])
    return theta

def dcm2body231(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[2, 0], dcm[0, 0])
    theta[1] = np.arcsin(dcm[1, 0])
    theta[2] = np.arctan2(-dcm[1, 2], dcm[1, 1])
    return theta

def dcm2body232(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 1], -dcm[0, 1])
    theta[1] = np.arccos(dcm[1, 1])
    theta[2] = np.arctan2(dcm[1, 2], dcm[1, 0])
    return theta

def dcm2body312(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[0, 1], dcm[1, 1])
    theta[1] = np.arcsin(dcm[2, 1])
    theta[2] = np.arctan2(-dcm[2, 0], dcm[2, 2])
    return theta

def dcm2body313(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 2], -dcm[1, 2])
    theta[1] = np.arccos(dcm[2, 2])
    theta[2] = np.arctan2(dcm[2, 0], dcm[2, 1])
    return theta

def dcm2body321(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 0], dcm[0, 0])
    theta[1] = np.arcsin(-dcm[2, 0])
    theta[2] = np.arctan2(dcm[2, 1], dcm[2, 2])
    return theta

def dcm2body323(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 2], dcm[0, 2])
    theta[1] = np.arccos(dcm[2, 2])
    theta[2] = np.arctan2(dcm[2, 1], -dcm[2, 0])
    return theta

def dcm2quat(C):
    tr = C[0, 0] + C[1, 1] + C[2, 2] + 1
    
    e = np.zeros(3)
    n = np.sqrt(tr) * 0.5
    
    e[0] = (C[1, 2] - C[2, 1]) / (4 * n)
    e[1] = (C[2, 0] - C[0, 2]) / (4 * n)
    e[2] = (C[0, 1] - C[1, 0]) / (4 * n)
    
    quat = np.concatenate([e, [n]])
    return quat

def dcm2space121(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 1], dcm[0, 2])
    theta[1] = np.arccos(dcm[0, 0])
    theta[2] = np.arctan2(dcm[1, 0], -dcm[2, 0])
    return theta

def dcm2space123(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 1], dcm[2, 2])
    theta[1] = np.arcsin(-dcm[2, 0])
    theta[2] = np.arctan2(dcm[1, 0], dcm[0, 0])
    return theta

def dcm2space131(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 2], -dcm[0, 1])
    theta[1] = np.arccos(dcm[0, 0])
    theta[2] = np.arctan2(dcm[2, 0], dcm[1, 0])
    return theta

def dcm2space132(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[1, 2], dcm[1, 1])
    theta[1] = np.arcsin(dcm[1, 0])
    theta[2] = np.arctan2(-dcm[2, 0], dcm[0, 0])
    return theta

def dcm2space212(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 0], -dcm[1, 2])
    theta[1] = np.arccos(dcm[1, 1])
    theta[2] = np.arctan2(dcm[0, 1], dcm[2, 1])
    return theta

def dcm2space213(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[2, 0], dcm[2, 2])
    theta[1] = np.arcsin(dcm[2, 1])
    theta[2] = np.arctan2(-dcm[0, 1], dcm[1, 1])
    return theta

def dcm2space231(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[0, 2], dcm[0, 0])
    theta[1] = np.arcsin(-dcm[0, 1])
    theta[2] = np.arctan2(dcm[2, 1], dcm[1, 1])
    return theta

def dcm2space232(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 2], dcm[1, 0])
    theta[1] = np.arccos(dcm[1, 1])
    theta[2] = np.arctan2(dcm[2, 1], -dcm[0, 1])
    return theta

def dcm2space312(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[1, 0], dcm[1, 1])
    theta[1] = np.arcsin(-dcm[1, 2])
    theta[2] = np.arctan2(dcm[0, 2], dcm[2, 2])
    return theta

def dcm2space313(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 0], dcm[2, 1])
    theta[1] = np.arccos(dcm[2, 2])
    theta[2] = np.arctan2(dcm[0, 2], -dcm[1, 2])
    return theta

def dcm2space321(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(-dcm[0, 1], dcm[0, 0])
    theta[1] = np.arcsin(dcm[0, 2])
    theta[2] = np.arctan2(-dcm[1, 2], dcm[2, 2])
    return theta

def dcm2space323(dcm):
    theta = np.zeros(3)
    theta[0] = np.arctan2(dcm[2, 1], -dcm[2, 0])
    theta[1] = np.arccos(dcm[2, 2])
    theta[2] = np.arctan2(dcm[1, 2], dcm[0, 2])
    return theta

def dcm2srt(C):
    C = C.T
    sigma = np.trace(C)
    theta = np.arccos(0.5 * (sigma - 1))
    
    sin_theta = np.sin(theta)
    lambda_ = np.zeros(3)
    
    if sigma == 3:
        print('Error: undefined')
    elif sigma == -1:
        print('error')
    else:
        lambda_[0] = 0.5 * (C[1, 2] - C[2, 1]) / sin_theta
        lambda_[1] = 0.5 * (C[2, 0] - C[0, 2]) / sin_theta
        lambda_[2] = 0.5 * (C[0, 1] - C[1, 0]) / sin_theta
    
    return lambda_, theta

def dcmdot(dcm, omega):
    return dcm @ skew_matrix(omega)

def orthodcm(DCM):
    delT = DCM @ DCM.T - np.eye(3)
    orthoDCM = DCM @ (np.eye(3) - 0.5*delT + 0.5*(3/4)*delT**2 - 0.5*(3/4)*(5/6)*delT**3 + 
                      0.5*(3/4)*(5/6)*(7/8)*delT**4 - 0.5*(3/4)*(5/6)*(7/8)*(9/10)*delT**5 +
                      0.5*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*delT**6 - 
                      0.5*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*delT**7 +
                      0.5*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*delT**8 -
                      0.5*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*(17/18)*delT**9 +
                      0.5*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*(17/18)*(19/20)*delT**10)
    return orthoDCM

def quat2body131(quat):
    dcm = quat2dcm(quat)
    theta = dcm2body131(dcm)
    return theta
    
import numpy as np
from numpy import sin, cos, arctan2, arcsin, arccos, sqrt, trace, dot, cross
from numpy.linalg import norm

def quat2dcm(quat):
    """
    Convert quaternion to direction cosine matrix.
    
    Parameters:
        quat : numpy.ndarray
            Nx4 array of quaternions [q1, q2, q3, q4] where q4 is the scalar part
            
    Returns:
        dcm : numpy.ndarray
            3x3xN array of direction cosine matrices
    """
    quat = np.atleast_2d(quat)
    N = quat.shape[0]
    
    # Normalize quaternions
    quat = quat / norm(quat, axis=1)[:, None]
    
    e1 = quat[:, 0]
    e2 = quat[:, 1]
    e3 = quat[:, 2]
    e4 = quat[:, 3]
    
    # Precompute products
    e1e1 = e1*e1
    e1e2 = e1*e2
    e1e3 = e1*e3
    e1e4 = e1*e4
    
    e2e2 = e2*e2
    e2e3 = e2*e3
    e2e4 = e2*e4
    
    e3e3 = e3*e3
    e3e4 = e3*e4
    
    e4e4 = e4*e4
    
    dcm = np.zeros((3, 3, N))
    
    dcm[0, 0, :] = 1 - 2*e2e2 - 2*e3e3
    dcm[0, 1, :] = 2*(e1e2 - e3e4)
    dcm[0, 2, :] = 2*(e1e3 + e2e4)
    
    dcm[1, 0, :] = 2*(e1e2 + e3e4)
    dcm[1, 1, :] = 1 - 2*e3e3 - 2*e1e1
    dcm[1, 2, :] = 2*(e2e3 - e1e4)
    
    dcm[2, 0, :] = 2*(e1e3 - e2e4)
    dcm[2, 1, :] = 2*(e2e3 + e1e4)
    dcm[2, 2, :] = 1 - 2*e1e1 - 2*e2e2
    
    if N == 1:
        return dcm[:, :, 0]
    return dcm

def quat2space131(quat):
    """
    Convert quaternion to Euler Space 1-3-1 angles.
    
    Parameters:
        quat : numpy.ndarray
            1x4 quaternion [q1, q2, q3, q4]
            
    Returns:
        theta : numpy.ndarray
            1x3 Euler angles [psi, theta, phi] in radians
    """
    dcm = quat2dcm(quat)
    return dcm2space131(dcm)

def quat2space213(quat):
    """
    Convert quaternion to Euler Space 2-1-3 angles.
    
    Parameters:
        quat : numpy.ndarray
            1x4 quaternion [q1, q2, q3, q4]
            
    Returns:
        theta : numpy.ndarray
            1x3 Euler angles [psi, theta, phi] in radians
    """
    dcm = quat2dcm(quat)
    return dcm2space213(dcm)

def quat2srt(quat):
    """
    Convert quaternion to Euler axis/angle representation.
    
    Parameters:
        quat : numpy.ndarray
            1x4 quaternion [q1, q2, q3, q4]
            
    Returns:
        lambda : numpy.ndarray
            Euler axis (unit vector)
        theta : float
            Euler angle in radians
    """
    e = quat[:3]
    n = quat[3]
    
    if norm(e) < 1e-10:
        lambda_ = np.array([1, 0, 0])
    else:
        lambda_ = e / norm(e)
    theta = 2 * arccos(n)
    
    return lambda_, theta

def quatdot(q, w):
    """
    Quaternion differential equation.
    
    Parameters:
        q : numpy.ndarray
            1x4 quaternion [q1, q2, q3, q4]
        w : numpy.ndarray
            1x3 angular velocity vector
            
    Returns:
        q_d : numpy.ndarray
            1x4 quaternion derivative
    """
    E = np.array([
        [q[3], -q[2], q[1], q[0]],
        [q[2], q[3], -q[0], q[1]],
        [-q[1], q[0], q[3], q[2]],
        [-q[0], -q[1], -q[2], q[3]]
    ])
    
    w = np.append(w, 0)
    q_d = 0.5 * w @ E.T
    
    return q_d

def quatdot2omega(q, qd):
    """
    Convert quaternion derivative to angular velocity.
    
    Parameters:
        q : numpy.ndarray
            1x4 quaternion
        qd : numpy.ndarray
            1x4 quaternion derivative
            
    Returns:
        w : numpy.ndarray
            1x3 angular velocity vector
    """
    E = np.array([
        [q[3], -q[2], q[1], q[0]],
        [q[2], q[3], -q[0], q[1]],
        [-q[1], q[0], q[3], q[2]],
        [-q[0], -q[1], -q[2], q[3]]
    ])
    
    w = 2 * qd @ E
    return w[:3]

def quatnorm(quat_in):
    """
    Normalize quaternion(s).
    
    Parameters:
        quat_in : numpy.ndarray
            Nx4 array of quaternions
            
    Returns:
        quat_out : numpy.ndarray
            Nx4 array of normalized quaternions
    """
    quat_mag = norm(quat_in, axis=1)
    quat_out = quat_in / quat_mag[:, None]
    return quat_out

def quat_rotvec(quat, a):
    """
    Rotate vector using quaternion.
    
    Parameters:
        quat : numpy.ndarray
            1x4 quaternion
        a : numpy.ndarray
            1x3 vector to rotate
            
    Returns:
        b : numpy.ndarray
            1x3 rotated vector
    """
    e = quat[:3]
    n = quat[3]
    
    a_e_cross = cross(a, e)
    b = a - 2*n*a_e_cross + 2*cross(e, -a_e_cross)
    return b

def quat_successive(quat_A2B1, quat_B12B):
    """
    Combine two successive quaternion rotations.
    
    Parameters:
        quat_A2B1 : numpy.ndarray
            Nx4 quaternion (frame A to B1)
        quat_B12B : numpy.ndarray
            Nx4 quaternion (frame B1 to B)
            
    Returns:
        quat_A2B : numpy.ndarray
            Nx4 quaternion (frame A to B)
    """
    e_A2B1 = quat_A2B1[:, :3]
    n_A2B1 = quat_A2B1[:, 3]
    
    e_B12B = quat_B12B[:, :3]
    n_B12B = quat_B12B[:, 3]
    
    e_A2B = e_A2B1 * n_B12B[:, None] + e_B12B * n_A2B1[:, None] + cross(e_B12B, e_A2B1)
    n_A2B = n_A2B1 * n_B12B - np.sum(e_A2B1 * e_B12B, axis=1)
    
    quat_A2B = np.column_stack((e_A2B, n_A2B))
    return quat_A2B

def ROT1(alpha):
    """
    Rotation matrix about first axis (X).
    
    Parameters:
        alpha : float
            Rotation angle in radians
            
    Returns:
        rot1 : numpy.ndarray
            3x3 rotation matrix
    """
    ca = cos(alpha)
    sa = sin(alpha)
    
    rot1 = np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])
    return rot1

def ROT2(beta):
    """
    Rotation matrix about second axis (Y).
    
    Parameters:
        beta : float
            Rotation angle in radians
            
    Returns:
        rot2 : numpy.ndarray
            3x3 rotation matrix
    """
    cb = cos(beta)
    sb = sin(beta)
    
    rot2 = np.array([
        [cb, 0, sb],
        [0, 1, 0],
        [-sb, 0, cb]
    ])
    return rot2

def ROT3(gamma):
    """
    Rotation matrix about third axis (Z).
    
    Parameters:
        gamma : float
            Rotation angle in radians
            
    Returns:
        rot3 : numpy.ndarray
            3x3 rotation matrix
    """
    cg = cos(gamma)
    sg = sin(gamma)
    
    rot3 = np.array([
        [cg, -sg, 0],
        [sg, cg, 0],
        [0, 0, 1]
    ])
    return rot3

def space131dot(theta, w):
    """
    Euler Space 1-3-1 differential equation.
    
    Parameters:
        theta : numpy.ndarray
            1x3 Euler angles [psi, theta, phi]
        w : numpy.ndarray
            1x3 angular velocity vector
            
    Returns:
        theta_d : numpy.ndarray
            1x3 Euler angle derivatives
    """
    c2 = cos(theta[1])
    c1 = cos(theta[0])
    s2 = sin(theta[1])
    s1 = sin(theta[0])
    
    theta_d = np.zeros(3)
    theta_d[0] = w[0] + (w[1]*c1 - w[2]*s1)*c2/s2
    theta_d[1] = w[1]*s1 + w[2]*c1
    theta_d[2] = (-w[1]*c1 + w[2]*s1)/s2
    
    return theta_d

def space213dot(theta, w):
    """
    Euler Space 2-1-3 differential equation.
    
    Parameters:
        theta : numpy.ndarray
            1x3 Euler angles [psi, theta, phi]
        w : numpy.ndarray
            1x3 angular velocity vector
            
    Returns:
        theta_d : numpy.ndarray
            1x3 Euler angle derivatives
    """
    c1 = cos(theta[0])
    c2 = cos(theta[1])
    s1 = sin(theta[0])
    s2 = sin(theta[1])
    
    theta_d = np.zeros(3)
    theta_d[0] = (w[0]*s1 - w[2]*c1)*s2/c2 + w[1]
    theta_d[1] = w[0]*c1 + w[2]*s1
    theta_d[2] = (-w[0]*s1 + w[2]*c1)/c2
    
    return theta_d

def space231dot(theta, w):
    """
    Euler Space 2-3-1 differential equation.
    
    Parameters:
        theta : numpy.ndarray
            1x3 Euler angles [psi, theta, phi]
        w : numpy.ndarray
            1x3 angular velocity vector
            
    Returns:
        theta_d : numpy.ndarray
            1x3 Euler angle derivatives
    """
    ct1 = cos(theta[0])
    st1 = sin(theta[0])
    st2 = sin(theta[1])
    ct2 = cos(theta[1])
    
    theta_d = np.zeros(3)
    theta_d[0] = (w[0]*ct1 + w[2]*st1)*st2/ct2 + w[1]
    theta_d[1] = -w[0]*st1 + w[2]*ct1
    theta_d[2] = (w[0]*ct1 + w[2]*st1)/ct2
    
    return theta_d

def srt2dcm(lambda_, theta):
    """
    Convert Euler axis/angle to DCM.
    
    Parameters:
        lambda_ : numpy.ndarray
            1x3 Euler axis (unit vector)
        theta : float
            Rotation angle in radians
            
    Returns:
        dcm : numpy.ndarray
            3x3 direction cosine matrix
    """
    lambda_ = lambda_.reshape(3, 1)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    lambda_skew = skew_matrix(lambda_)
    
    dcm = np.eye(3)*cos_theta - sin_theta*lambda_skew + (1-cos_theta)*(lambda_ @ lambda_.T)
    return dcm.T

def srt2quat(lambda_, theta):
    """
    Convert Euler axis/angle to quaternion.
    
    Parameters:
        lambda_ : numpy.ndarray
            1x3 Euler axis (unit vector)
        theta : float
            Rotation angle in radians
            
    Returns:
        quat : numpy.ndarray
            1x4 quaternion [e1, e2, e3, n]
    """
    e = lambda_ * sin(theta/2)
    n = cos(theta/2)
    return np.append(e, n)

def srt_rotvec(lambda_, theta, a):
    """
    Rotate vector using Euler axis/angle.
    
    Parameters:
        lambda_ : numpy.ndarray
            1x3 Euler axis (unit vector)
        theta : float
            Rotation angle in radians
        a : numpy.ndarray
            1x3 vector to rotate
            
    Returns:
        b : numpy.ndarray
            1x3 rotated vector
    """
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    
    b = a*cos_theta - cross(a, lambda_)*sin_theta + dot(a, lambda_)*lambda_*(1-cos_theta)
    return b

def skew_matrix(v):
    """
    Create skew-symmetric matrix from vector.
    
    Parameters:
        v : numpy.ndarray
            1x3 vector
            
    Returns:
        skew : numpy.ndarray
            3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])    
    
if __name__ == "__main__":
    coupled_constrained_control()