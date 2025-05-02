import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt

class Param:
    def __init__(self):
        self.m = 2
        self.g = 9.81 
        self.mass = 2
        self.gravity = 9.81
        self.d = 0.169
        self.c_tf = 0.0135
        self.J = np.diag([0.02, 0.02, 0.04])
        self.use_disturbances = False
        self.W_x = np.eye(3)
        self.theta_x = np.array([1, 0.8, -1])
        self.W_R = np.eye(3)
        self.theta_R = np.array([0.1, 0.1, -0.1])
        self.kp = 10
        self.kv = 4
        self.k1 = 6
        self.k2 = 2
        self.kW = 8
        self.kwy = 1
        self.dt = 0

        self.kX = np.diag([self.kp, self.kp, self.kp]) # Position gains
        self.kV = np.diag([self.kv, self.kv, self.kv]) # Velocity gains
        self.kW_ = np.diag([self.kW, self.kW, self.kwy])
        
def run_kooijman():
    # Simulation parameters
    t = np.arange(0, 10.01, 0.01)
    N = len(t)

    param = Param()

    # Initial conditions
    x0 = np.array([0, 0, 0])
    v0 = np.array([0, 0, 0])
    W0 = np.array([0, 0, 0])

    e3 = np.array([0, 0, 1])
    R0 = expm((np.pi - 0.01) * hat(e3))

    X0 = np.concatenate([x0, v0, R0.flatten(), W0])

    # Numerical integration
    X = odeint(eom_kooijman, X0, t, args=(param,), rtol=1e-6, atol=1e-6)


    # print every tenth row of the X array
    for i in range(0, 10, 1):
        print(i,X[i])
    # Output arrays
    e, d, R, f, M = generate_output_arrays(N)
    b1 = np.zeros((3, N))
    b3 = np.zeros((3, N))
    thr = np.zeros((4, N))

    avg_ex = 0
    avg_eR = 0
    avg_f = 0

    converge_t = 0
    is_converged = False
    converge_ex = 0.02

    for i in range(N):
        R[:,:,i] = X[i,6:15].reshape(3, 3)
        b1[:,i] = R[:,0,i]
        b3[:,i] = R[:,2,i]
        
        desired = command(t[i])
        f[i], M[:,i], err, calc = position_control_kooijman(X[i,:], desired, param)
        
        f[i], M[:,i] = saturate_fM(f[i], M[:,i], param)
        thr[:,i] = fM_to_thr(f[i], M[:,i], param)
        
        # Unpack errors
        e['x'][:,i] = -err['x']
        e['v'][:,i] = -err['v']
        
        # Unpack desired values
        d['x'][:,i] = desired['x']
        d['v'][:,i] = desired['v']
        
        # Find normalized errors
        norm_ex = norm(err['x'])
        norm_eR = norm(err['R'])
        
        avg_ex += norm_ex
        avg_eR += norm_eR
        
        norm_f = norm(thr[:,i])
        avg_f += norm_f
        
        if norm_ex < converge_ex:
            if not is_converged:
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

    plt.figure(1)
    plot_3x1(t, e['x'], '', xlabel_, 'e_x', linetype, linewidth)

    plt.figure(2)
    plot_3x1(t, e['R'], '', xlabel_, 'e_R', linetype, linewidth)

    plt.figure(3)
    plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)

    print('')

def eom_kooijman(X, t, param):
    e3 = np.array([0, 0, 1])
    m = param.m
    J = param.J

    x, v, R, W = split_to_states(X)

    desired = command(t)
    T_, tau_, error_, calculated_ = position_control_kooijman(X, desired, param)
    # T, tau, error, calculated = position_control_kooijman_original(X, desired, param)

    thr = fM_to_thr(T, tau, param)
    T, tau = saturate_fM(T, tau, param)

    x_dot = v
    v_dot = param.g*e3 - T*R@e3 / m
    if param.use_disturbances:
        v_dot += param['W_x']@param['theta_x']/m

    R_dot = R @ hat(W)

    if param.use_disturbances:
        W_dot = np.linalg.solve(J, -hat(J@W)@W + tau + param['W_R']@param['theta_R'])
    else:
        W_dot = np.linalg.solve(J, -hat(W)@J@W + tau)

    return np.concatenate([x_dot, v_dot, R_dot.flatten(), W_dot])

def position_control_kooijman(X, desired, param):
    x, v, R, W = split_to_states(X)
    
    currentBodyState=(x,v,np.zeros(3),W,R)
    desiredBodyState = [(desired['x'], desired['v'], desired['x_2dot'], desired['x_3dot'], desired['x_4dot']), 
                        (desired['b1'], desired['b1_dot'], desired['b1_2dot'])]
    
    params = Param()
    return getCommand(currentBodyState, desiredBodyState, pos_control=True, vel_control=True, param=params)
    
def position_control_kooijman_original(X, desired, param):
    x, v, R, W = split_to_states(X)
    R_dot = R@hat(W)

    m = param.m
    J = param.J
    g = param.g

    kp = param.kp
    kv = param.kv
    kW = param.kW
    kwy = param.kwy

    e3 = np.array([0, 0, 1])

    error = {
        'x': desired['x'] - x,
        'v': desired['v'] - v
    }

    r1 = R[:,0]
    r2 = R[:,1]
    r3 = R[:,2]

    r1_dot = R_dot[:,0]
    r2_dot = R_dot[:,1]
    r3_dot = R_dot[:,2]

    b3 = R@e3
    b3_dot = R_dot@e3

    b = -desired['x_2dot'] + g*e3

    T_bar = m*norm(b)
    T_msqrt3 = T_bar / (np.sqrt(3)*m)
    L_lower = np.array([-T_msqrt3, -T_msqrt3, g - T_msqrt3])
    L_upper = np.array([T_msqrt3, T_msqrt3, g + T_msqrt3])

    u_bar = desired['x_2dot']
    u = u_bar + kp*error['x'] + kv*error['v']

    a_bar_ref = g*e3 - u
    n_a_bar_ref = norm(a_bar_ref)

    T = m*n_a_bar_ref
    u_bar_dot = desired['x_3dot']
    v_dot = g * e3 - T / m * b3
    error['a'] = desired['x_2dot'] - v_dot
    u_dot = u_bar_dot + kp*error['v'] + kv*error['a']
    a_ref_dot = -u_dot

    n_a_ref_dot = a_bar_ref.T@a_ref_dot / n_a_bar_ref
    T_dot = m*n_a_ref_dot
    v_2dot = - T_dot / m * b3 - T / m * b3_dot
    error['a_dot'] = desired['x_3dot'] - v_2dot

    u_bar_2dot = desired['x_4dot']
    u_2dot = u_bar_2dot + kp*error['a'] + kv*error['a_dot']
    a_ref_2dot = -u_2dot

    r3_bar, r3_bar_dot, r3_bar_2dot = deriv_unit_vector(a_bar_ref, a_ref_dot, a_ref_2dot)

    phi_bar = desired['yaw']
    phi_bar_dot = desired['w']
    phi_bar_2dot = desired['w_dot']

    r_yaw = np.array([-np.sin(phi_bar), np.cos(phi_bar), 0])
    r_yaw_dot = np.array([
        -np.cos(phi_bar)*phi_bar_dot,
        -np.sin(phi_bar)*phi_bar_dot,
        0
    ])
    r_yaw_2dot = np.array([
        np.sin(phi_bar)*phi_bar_dot**2 + -np.cos(phi_bar)*phi_bar_2dot,
        -np.cos(phi_bar)*phi_bar_dot**2 - np.sin(phi_bar)*phi_bar_2dot,
        0
    ])

    num = hat(r_yaw)@r3_bar
    num_dot = hat(r_yaw_dot)@r3_bar + hat(r_yaw)@r3_bar_dot
    num_2dot = hat(r_yaw_2dot)@r3_bar + hat(r_yaw_dot)@r3_bar_dot + hat(r_yaw_dot)@r3_bar_dot + hat(r_yaw)@r3_bar_2dot

    den = s(r_yaw, r3_bar)
    den_dot = s_dot(r_yaw, r3_bar, r_yaw_dot, r3_bar_dot)
    den_2dot = s_2dot(r_yaw, r3_bar, r_yaw_dot, r3_bar_dot, r_yaw_2dot, r3_bar_2dot)

    r1_bar = num/den
    r1_bar_dot = diff_num_den(num, num_dot, den, den_dot)
    r1_bar_2dot = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    r2_bar = hat(r3_bar)@r1_bar

    u_v = calculate_u_v(r3, r3_bar, r3_bar_dot, r1, param)
    u_w = calculate_u_w(r1, r2, r3, r1_bar, r1_bar_dot, r3_bar, param)

    R_e, R_r = get_Re_Rr(r3, r3_bar)

    # r3_dot = (np.eye(3) - r3[:,None]@r3[None,:])@u_v
    R_r_dot = get_Rr_dot(r3, r3_dot, r3_bar, r3_bar_dot)
    w_r = vee(R_r.T@R_r_dot)

    R_e_dot = get_Re_dot(r3, r3_dot, r3_bar, r3_bar_dot)
    w_e = vee(R_e.T@R_e_dot)

    W_bar1 = -r2.T@u_v
    W_bar2 = r1.T@u_v

    if abs(r3.T@r3_bar) > 1e-3:
        w1 = r1.T@R_r@R_e.T@r1_bar
        w2 = r2.T@R_r@R_e.T@r1_bar
    else:
        w1 = r1.T@r1_bar
        w2 = r2.T@r2_bar

    beta1 = w2*r3.T@R_r@R_e.T@r1_bar - r1.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar
    beta2 = w1*r3.T@R_r@R_e.T@r1_bar + r2.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar

    if abs(w1) > abs(w2):
        w_r = beta2/w1
    else:
        w_r = beta1/w2

    W_bar = np.array([W_bar1, W_bar2, u_w + w_r])

    r3_dot = R_dot[:,2]
    u_v_dot = calculate_u_v_dot(r3, r3_dot, r3_bar, r3_bar_dot, r3_bar_2dot, r1_dot, param)

    u_w_dot = calculate_u_w_dot(r1, r1_dot, r2, r2_dot, r3, r3_dot, 
                              r1_bar, r1_bar_dot, r1_bar_2dot, r3_bar, r3_bar_dot, param)

    r3_2dot = (- r3_dot[:,None]@r3[None,:] - r3[:,None]@r3_dot[None,:])@u_v + (np.eye(3) - r3[:,None]@r3[None,:])@u_v_dot

    w1_dot = (r1_dot.T@R_r@R_e.T@r1_bar + r1.T@R_r_dot@R_e.T@r1_bar + 
             r1.T@R_r@R_e_dot.T@r1_bar + r1.T@R_r@R_e.T@r1_bar_dot)

    w2_dot = (r2_dot.T@R_r@R_e.T@r1_bar + r2.T@R_r_dot@R_e.T@r1_bar + 
             r2.T@R_r@R_e_dot.T@r1_bar + r2.T@R_r@R_e.T@r1_bar_dot)

    R_r_2dot = get_Rr_2dot(r3, r3_dot, r3_2dot, r3_bar, r3_bar_dot, r3_bar_2dot)
    R_e_2dot = get_Re_2dot(r3, r3_dot, r3_2dot, r3_bar, r3_bar_dot, r3_bar_2dot)

    w_r_dot = vee(R_r_dot.T@R_r_dot) + vee(R_r.T@R_r_2dot)
    w_e_dot = vee(R_e_dot.T@R_e_dot) + vee(R_e.T@R_e_2dot)

    beta1_dot = (w2_dot*r3.T@R_r@R_e.T@r1_bar + w2*r3_dot.T@R_r@R_e.T@r1_bar + 
                w2*r3.T@R_r_dot@R_e.T@r1_bar + w2*r3.T@R_r@R_e_dot.T@r1_bar + 
                w2*r3.T@R_r@R_e.T@r1_bar_dot - r1_dot.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar - 
                r1.T@R_r_dot@hat(w_r - w_e)@R_e.T@r1_bar - r1.T@R_r@hat(w_r_dot - w_e_dot)@R_e.T@r1_bar - 
                r1.T@R_r@hat(w_r - w_e)@R_e_dot.T@r1_bar - r1.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar_dot)

    beta2_dot = (w1_dot*r3.T@R_r@R_e.T@r1_bar + 
                 w1*r3_dot.T@R_r@R_e.T@r1_bar + 
                 w1*r3.T@R_r_dot@R_e.T@r1_bar + 
                 w1*r3.T@R_r@R_e_dot.T@r1_bar + 
                 w1*r3.T@R_r@R_e.T@r1_bar_dot + 
                 r2_dot.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar + 
                 r2.T@R_r_dot@hat(w_r - w_e)@R_e.T@r1_bar + 
                 r2.T@R_r@hat(w_r_dot - w_e_dot)@R_e.T@r1_bar + 
                 r2.T@R_r@hat(w_r - w_e)@R_e_dot.T@r1_bar + 
                 r2.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar_dot)

    if abs(w1) > abs(w2):
        w_r_dot = diff_num_den(beta2, beta2_dot, w1, w1_dot)
    else:
        w_r_dot = diff_num_den(beta1, beta1_dot, w2, w2_dot)

    W1_dot = - r2_dot.T@u_v - r2.T@u_v_dot
    W2_dot = r1_dot.T@u_v + r1.T@u_v_dot
    W3_dot = u_w_dot + w_r_dot
    W_bar_dot = np.array([W1_dot, W2_dot, W3_dot])

    Rd = np.column_stack([r1_bar, r2_bar, r3_bar])

    Wd = W_bar
    Wd_dot = W_bar_dot

    kW = np.diag([param.kW, param.kW, param.kwy])

    eW = W - Wd
    tau = -kW@eW + hat(W)@J@W + J@Wd_dot

    calculated = {
        'b3': r3_bar,
        'b3_dot': r3_bar_dot,
        'R': Rd
    }
    error['R'] = 0.5*vee(Rd.T@R - R.T@Rd)

    return T, tau, error, calculated





###############################################################################################################
def getCommand(currentBodyState, desiredBodyState, pos_control=True, vel_control=True, param=None):
    """Position controller to determine desired attitude and angular rates
    to achieve the deisred states.

    This uses the controller defined in "Control of Complex Maneuvers
    for a Quadrotor UAV using Geometric Methods on SE(3)"
    URL: https://arxiv.org/pdf/1003.2005.pdf
    """
    (pos_ned, vel_ned, accel_ned, omega_ned, quat_ned_bodyfrd) = currentBodyState
    (pos_ned, vel_ned, accel_ned, omega_ned, R) = currentBodyState
    W = omega_ned
    R = quat_ned_bodyfrd#.to_rotation_matrix()
    R_dot = R@hat(omega_ned)
    # (pos_des_ned, vel_des_ned, accel_des_ned, b1d_ned) = desiredState
    (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot) = desiredBodyState[0]
    (b1d, b1d_dot, b1d_2dot) = desiredBodyState[1] 

    e3 = np.array([0., 0., 1.]) #self.e3

    mass = param.mass
    kV = param.kV
    # kIV = param.kIV
    # kDV = param.kDV
    posFactor = 1 if pos_control else 0
    kX = param.kX*posFactor
    # kIX = param.kIX*posFactor

    # self.update_current_time()
    # dt = param.dt

    # Translational error functions
    
    eX = -(pos_ned - xd) if pos_control else np.zeros(3)   # position error - eq (11)
    eV = -(vel_ned - xd_dot)                            # velocity error - eq (12)

    # Position integral terms
    use_integralTerm = False

    # Force 'f' along negative b3-axis -                                 eq (14)
    # This term equals to R.e3

    r1 = R[:,0]
    r2 = R[:,1]
    r3 = R[:,2]

    r1_dot = R_dot[:,0]
    r2_dot = R_dot[:,1]
    r3_dot = R_dot[:,2]

    b3 = R@e3
    b3_dot = R_dot@e3

    b = -xd_2dot + param.gravity*e3

    T_bar = mass*norm(b)
    T_msqrt3 = T_bar / (np.sqrt(3)*mass)
    L_lower = np.array([-T_msqrt3, -T_msqrt3, param.gravity - T_msqrt3])
    L_upper = np.array([T_msqrt3, T_msqrt3, param.gravity + T_msqrt3])

    u_bar = xd_2dot
    u = u_bar + kX@eX + kV@eV

    a_bar_ref = param.gravity*e3 - u
    n_a_bar_ref = norm(a_bar_ref)

    T = mass*n_a_bar_ref
    u_bar_dot = xd_3dot
    v_dot = param.gravity * e3 - T / mass * b3
    ea = xd_2dot - v_dot
    u_dot = u_bar_dot + kX@eV + kV@ea
    a_ref_dot = -u_dot

    n_a_ref_dot = a_bar_ref.T@a_ref_dot / n_a_bar_ref
    T_dot = mass*n_a_ref_dot
    v_2dot = - T_dot / mass * b3 - T / mass * b3_dot
    ea_dot = xd_3dot - v_2dot

    u_bar_2dot = xd_4dot
    u_2dot = u_bar_2dot + kX@ea + kV@ea_dot
    a_ref_2dot = -u_2dot

    r3_bar, r3_bar_dot, r3_bar_2dot = deriv_unit_vector(a_bar_ref, a_ref_dot, a_ref_2dot)

    # phi_bar = desired['yaw']
    # phi_bar_dot = desired['w']
    # phi_bar_2dot = desired['w_dot']

    r_yaw = b1d #np.array([-np.sin(phi_bar), np.cos(phi_bar), 0])
    # r_yaw_dot = np.array([
    #     -np.cos(phi_bar)*phi_bar_dot,
    #     -np.sin(phi_bar)*phi_bar_dot,
    #     0
    # ])
    r_yaw_dot = b1d_dot
    # r_yaw_2dot = np.array([
    #     np.sin(phi_bar)*phi_bar_dot**2 + -np.cos(phi_bar)*phi_bar_2dot,
    #     -np.cos(phi_bar)*phi_bar_dot**2 - np.sin(phi_bar)*phi_bar_2dot,
    #     0
    # ])
    r_yaw_2dot = b1d_2dot
    
    num = hat(r_yaw)@r3_bar
    num_dot = hat(r_yaw_dot)@r3_bar + hat(r_yaw)@r3_bar_dot
    num_2dot = hat(r_yaw_2dot)@r3_bar + hat(r_yaw_dot)@r3_bar_dot + hat(r_yaw_dot)@r3_bar_dot + hat(r_yaw)@r3_bar_2dot

    den = s(r_yaw, r3_bar)
    den_dot = s_dot(r_yaw, r3_bar, r_yaw_dot, r3_bar_dot)
    den_2dot = s_2dot(r_yaw, r3_bar, r_yaw_dot, r3_bar_dot, r_yaw_2dot, r3_bar_2dot)

    r1_bar = num/den
    r1_bar_dot = diff_num_den(num, num_dot, den, den_dot)
    r1_bar_2dot = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    r2_bar = hat(r3_bar)@r1_bar

    u_v = calculate_u_v(r3, r3_bar, r3_bar_dot, r1, param)
    u_w = calculate_u_w(r1, r2, r3, r1_bar, r1_bar_dot, r3_bar, param)

    R_e, R_r = get_Re_Rr(r3, r3_bar)

    # r3_dot = (np.eye(3) - r3[:,None]@r3[None,:])@u_v
    R_r_dot = get_Rr_dot(r3, r3_dot, r3_bar, r3_bar_dot)
    w_r = vee(R_r.T@R_r_dot)

    R_e_dot = get_Re_dot(r3, r3_dot, r3_bar, r3_bar_dot)
    w_e = vee(R_e.T@R_e_dot)

    W_bar1 = -r2.T@u_v
    W_bar2 = r1.T@u_v

    if abs(r3.T@r3_bar) > 1e-3:
        w1 = r1.T@R_r@R_e.T@r1_bar
        w2 = r2.T@R_r@R_e.T@r1_bar
    else:
        w1 = r1.T@r1_bar
        w2 = r2.T@r2_bar

    beta1 = w2*r3.T@R_r@R_e.T@r1_bar - r1.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar
    beta2 = w1*r3.T@R_r@R_e.T@r1_bar + r2.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar

    if abs(w1) > abs(w2):
        w_r = beta2/w1
    else:
        w_r = beta1/w2

    W_bar = np.array([W_bar1, W_bar2, u_w + w_r])

    r3_dot = R_dot[:,2]
    u_v_dot = calculate_u_v_dot(r3, r3_dot, r3_bar, r3_bar_dot, r3_bar_2dot, r1_dot, param)

    u_w_dot = calculate_u_w_dot(r1, r1_dot, r2, r2_dot, r3, r3_dot, 
                            r1_bar, r1_bar_dot, r1_bar_2dot, r3_bar, r3_bar_dot, param)

    r3_2dot = (- r3_dot[:,None]@r3[None,:] - r3[:,None]@r3_dot[None,:])@u_v + (np.eye(3) - r3[:,None]@r3[None,:])@u_v_dot

    w1_dot = (r1_dot.T@R_r@R_e.T@r1_bar + r1.T@R_r_dot@R_e.T@r1_bar + 
            r1.T@R_r@R_e_dot.T@r1_bar + r1.T@R_r@R_e.T@r1_bar_dot)

    w2_dot = (r2_dot.T@R_r@R_e.T@r1_bar + r2.T@R_r_dot@R_e.T@r1_bar + 
            r2.T@R_r@R_e_dot.T@r1_bar + r2.T@R_r@R_e.T@r1_bar_dot)

    R_r_2dot = get_Rr_2dot(r3, r3_dot, r3_2dot, r3_bar, r3_bar_dot, r3_bar_2dot)
    R_e_2dot = get_Re_2dot(r3, r3_dot, r3_2dot, r3_bar, r3_bar_dot, r3_bar_2dot)

    w_r_dot = vee(R_r_dot.T@R_r_dot) + vee(R_r.T@R_r_2dot)
    w_e_dot = vee(R_e_dot.T@R_e_dot) + vee(R_e.T@R_e_2dot)

    beta1_dot = (w2_dot*r3.T@R_r@R_e.T@r1_bar + w2*r3_dot.T@R_r@R_e.T@r1_bar + 
                w2*r3.T@R_r_dot@R_e.T@r1_bar + w2*r3.T@R_r@R_e_dot.T@r1_bar + 
                w2*r3.T@R_r@R_e.T@r1_bar_dot - r1_dot.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar - 
                r1.T@R_r_dot@hat(w_r - w_e)@R_e.T@r1_bar - r1.T@R_r@hat(w_r_dot - w_e_dot)@R_e.T@r1_bar - 
                r1.T@R_r@hat(w_r - w_e)@R_e_dot.T@r1_bar - r1.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar_dot)

    beta2_dot = (w1_dot*r3.T@R_r@R_e.T@r1_bar + w1*r3_dot.T@R_r@R_e.T@r1_bar + 
                w1*r3.T@R_r_dot@R_e.T@r1_bar + w1*r3.T@R_r@R_e_dot.T@r1_bar + 
                w1*r3.T@R_r@R_e.T@r1_bar_dot + r2_dot.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar + 
                r2.T@R_r_dot@hat(w_r - w_e)@R_e.T@r1_bar + r2.T@R_r@hat(w_r_dot - w_e_dot)@R_e.T@r1_bar + 
                r2.T@R_r@hat(w_r - w_e)@R_e_dot.T@r1_bar + r2.T@R_r@hat(w_r - w_e)@R_e.T@r1_bar_dot)

    if abs(w1) > abs(w2):
        w_r_dot = diff_num_den(beta2, beta2_dot, w1, w1_dot)
    else:
        w_r_dot = diff_num_den(beta1, beta1_dot, w2, w2_dot)

    W1_dot = - r2_dot.T@u_v - r2.T@u_v_dot
    W2_dot = r1_dot.T@u_v + r1.T@u_v_dot
    W3_dot = u_w_dot + w_r_dot
    W_bar_dot = np.array([W1_dot, W2_dot, W3_dot])

    Rd = np.column_stack([r1_bar, r2_bar, r3_bar])

    Wd = W_bar
    Wd_dot = W_bar_dot

    eW = W - Wd
    tau = -param.kW_ @ eW + hat(W)@param.J@W + param.J@Wd_dot

    calculated = {
        'b3': r3_bar,
        'b3_dot': r3_bar_dot,
        'R': Rd
    }
    eR = 0.5*vee(Rd.T@R - R.T@Rd)

    error = {
        'x': eX,
        'v': eV,
        'a': ea,  # acceleration error
        'a_dot': ea_dot,  # acceleration error
        'R': eR,
        'W': eW
    }
    return T, tau, error, calculated
    # f_total = T
    # R_desired = Rd
    # Omega_desired = Wd
    # self.A = u
    # return f_total, R_desired, Omega_desired
#














def command(t):
    return command_line(t)

def calculate_u_v_dot(v, v_dot, v_bar, v_bar_dot, v_bar_2dot, r1_dot, param):
    k1 = param.k1#['k1']

    if v.T@v_bar >= 0:
        u_v_FB_dot = k1*v_bar_dot
    elif np.allclose(v, -v_bar):
        u_v_FB_dot = k1*r1_dot
    else:
        num = k1*v_bar
        num_dot = k1*v_bar_dot
        den = s(v, v_bar)
        den_dot = s_dot(v, v_bar, v_dot, v_bar_dot)
        u_v_FB_dot = diff_num_den(num, num_dot, den, den_dot)

    if np.allclose(v, v_bar):
        u_v_FF_dot = v_bar_2dot
    elif np.allclose(v, -v_bar):
        u_v_FF_dot = -v_bar_2dot
    else:
        vxvbar = np.cross(v, v_bar)
        vxvbar_dot = np.cross(v_dot, v_bar) + np.cross(v, v_bar_dot)

        num = vxvbar[:,None]@vxvbar[None,:] - (np.eye(3) - v[:,None]@v[None,:])@v_bar[:,None]@v[None,:]
        num_dot = (vxvbar_dot[:,None]@vxvbar[None,:] + vxvbar[:,None]@vxvbar_dot[None,:] - 
                  (- v_dot[:,None]@v[None,:] - v[:,None]@v_dot[None,:])@v_bar[:,None]@v[None,:] - 
                  (np.eye(3) - v[:,None]@v[None,:])@v_bar_dot[:,None]@v[None,:] - 
                  (np.eye(3) - v[:,None]@v[None,:])@v_bar[:,None]@v_dot[None,:])

        den = s(v, v_bar)**2
        den_dot = 2*s(v, v_bar)*s_dot(v, v_bar, v_dot, v_bar_dot)

        theta = num / den
        theta_dot = diff_num_den(num, num_dot, den, den_dot)

        u_v_FF_dot = theta_dot@v_bar_dot + theta@v_bar_2dot

    return u_v_FB_dot + u_v_FF_dot

def calculate_u_v(v, v_bar, v_bar_dot, r1, param):
    k1 = param.k1 #['k1']

    if v.T@v_bar >= 0:
        u_v_FB = k1*v_bar
    elif np.allclose(v, -v_bar):
        u_v_FB = k1*r1
    else:
        u_v_FB = k1*v_bar / s(v, v_bar)

    if np.allclose(v, v_bar):
        u_v_FF = v_bar_dot
    elif np.allclose(v, -v_bar):
        u_v_FF = -v_bar_dot
    else:
        vxvbar = np.cross(v, v_bar)
        theta = 1 / s(v, v_bar)**2 * (vxvbar[:,None]@vxvbar[None,:] - (np.eye(3) - v[:,None]@v[None,:])@v_bar[:,None]@v[None,:])
        u_v_FF = theta@v_bar_dot

    return u_v_FB + u_v_FF

def calculate_u_w_dot(r1, r1_dot, r2, r2_dot, r3, r3_dot, r1_bar, r1_bar_dot, r1_bar_2dot, r3_bar, r3_bar_dot, param):
    k2 = param.k2 #['k2']

    R_e, R_r = get_Re_Rr(r3, r3_bar)
    R_r_dot = get_Rr_dot(r3, r3_dot, r3_bar, r3_bar_dot)
    R_e_dot = get_Re_dot(r3, r3_dot, r3_bar, r3_bar_dot)

    w1 = r1.T@R_r@R_e.T@r1_bar
    w1_dot = (r1_dot.T@R_r@R_e.T@r1_bar + r1.T@R_r_dot@R_e.T@r1_bar + 
             r1.T@R_r@R_e_dot.T@r1_bar + r1.T@R_r@R_e.T@r1_bar_dot)

    w2 = r2.T@R_r@R_e.T@r1_bar
    w2_dot = (r2_dot.T@R_r@R_e.T@r1_bar + r2.T@R_r_dot@R_e.T@r1_bar + 
             r2.T@R_r@R_e_dot.T@r1_bar + r2.T@R_r@R_e.T@r1_bar_dot)

    if abs(w1) > abs(w2):
        theta2 = r2.T@R_r@R_e.T@r1_bar_dot
        theta2_dot = (r2_dot.T@R_r@R_e.T@r1_bar_dot + r2.T@R_r_dot@R_e.T@r1_bar_dot + 
                     r2.T@R_r@R_e_dot.T@r1_bar_dot + r2.T@R_r@R_e.T@r1_bar_2dot)
        
        num = theta2
        num_dot = theta2_dot

        den = w1
        den_dot = w1_dot

        u_w_FF_dot = diff_num_den(num, num_dot, den, den_dot)
    else:
        theta1 = -r1.T@R_r@R_e.T@r1_bar_dot
        theta1_dot = (-r1_dot.T@R_r@R_e.T@r1_bar_dot - r1.T@R_r_dot@R_e.T@r1_bar_dot - 
                     r1.T@R_r@R_e_dot.T@r1_bar_dot - r1.T@R_r@R_e.T@r1_bar_2dot)

        num = theta1
        num_dot = theta1_dot

        den = w2
        den_dot = w2_dot

        u_w_FF_dot = diff_num_den(num, num_dot, den, den_dot)

    if w1 >= 0:
        u_w_FB_dot = k2*w2_dot
    elif w1 < 0 and w2 < 0:
        u_w_FB_dot = -k2
    else:
        u_w_FB_dot = k2

    return u_w_FB_dot + u_w_FF_dot

def calculate_u_w(r1, r2, r3, r1_bar, r1_bar_dot, r3_bar, param):
    k2 = param.k2#['k2']

    R_e, R_r = get_Re_Rr(r3, r3_bar)

    w1 = r1.T@R_r@R_e.T@r1_bar
    w2 = r2.T@R_r@R_e.T@r1_bar

    if abs(w1) > abs(w2):
        theta2 = r2.T@R_r@R_e.T@r1_bar_dot
        u_w_FF = theta2 / w1
    else:
        theta1 = -r1.T@R_r@R_e.T@r1_bar_dot
        u_w_FF = theta1 / w2

    if w1 >= 0:
        u_w_FB = k2*w2
    elif w1 < 0 and w2 < 0:
        u_w_FB = -k2
    else:
        u_w_FB = k2

    return u_w_FB + u_w_FF

def diff_num_den(num, num_dot, den, den_dot):
    return (den*num_dot - num*den_dot) / den**2

def diff2_num_den(num, num_den, num_2dot, den, den_dot, den_2dot):
    numerator = den**2*(den*num_2dot - num*den_2dot) - (den*num_den - num*den_dot)*2*den*den_dot
    denominator = den**4
    return numerator / denominator

def fM_to_thr(f, M, param):
    d = param.d
    ctf = param.c_tf

    f_to_fM = np.array([
        [1, 1, 1, 1],
        [0, -d, 0, d],
        [d, 0, -d, 0],
        [-ctf, ctf, -ctf, ctf]
    ])

    return np.linalg.solve(f_to_fM, np.concatenate([[f], M]))

def get_Re_2dot(v, v_dot, v_2dot, v_bar, v_bar_dot, v_bar_2dot):
    den = s(v_bar, v)
    den_dot = s_dot(v, v_bar, v_dot, v_bar_dot)
    den_2dot = s_2dot(v, v_bar, v_dot, v_bar_dot, v_2dot, v_bar_2dot)

    num = hat(v_bar)@v
    num_dot = hat(v_bar_dot)@v + hat(v_bar)@v_dot
    num_2dot = hat(v_bar_2dot)@v + hat(v_bar_dot)@v_dot + hat(v_bar_dot)@v_dot + hat(v_bar)@v_2dot
    Rrd1 = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    num1 = (np.eye(3) - v_bar[:,None]@v_bar[None,:])
    num1_dot = -(v_bar_dot[:,None]@v_bar[None,:] + v_bar[:,None]@v_bar_dot[None,:])
    num1_2dot = -(v_bar_2dot[:,None]@v_bar[None,:] + v_bar_dot[:,None]@v_bar_dot[None,:] + 
                 v_bar_dot[:,None]@v_bar_dot[None,:] + v_bar[:,None]@v_bar_2dot[None,:])

    num = -num1@v
    num_dot = -num1_dot@v - num1@v_dot
    num_2dot = - num1_2dot@v - 2*(num1_dot@v_dot) - num1@v_2dot

    Rrd2 = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    Rrd3 = v_bar_2dot

    return np.column_stack([Rrd1, Rrd2, Rrd3])

def get_Re_dot(v, v_dot, v_bar, v_bar_dot):
    den = s(v_bar, v)
    den_dot = s_dot(v, v_bar, v_dot, v_bar_dot)

    num = hat(v_bar)@v
    num_dot = hat(v_bar_dot)@v + hat(v_bar)@v_dot
    Rrd1 = diff_num_den(num, num_dot, den, den_dot)
    if norm(den) < 1e-3:
        Rrd1 *= 0

    num = -(np.eye(3) - v_bar[:,None]@v_bar[None,:])@v
    num_dot = (v_bar_dot[:,None]@v_bar[None,:] + v_bar[:,None]@v_bar_dot[None,:])@v - (np.eye(3) - v_bar[:,None]@v_bar[None,:])@v_dot
    Rrd2 = diff_num_den(num, num_dot, den, den_dot)
    if norm(den) < 1e-3:
        Rrd2 *= 0

    Rrd3 = v_bar_dot

    return np.column_stack([Rrd1, Rrd2, Rrd3])

def get_Re_Rr(v, v_bar):
    v_barTv = s(v_bar, v)

    R_e = np.column_stack([
        hat(v_bar)@v / v_barTv,
        -(np.eye(3) - v_bar[:,None]@v_bar[None,:])@v / v_barTv,
        v_bar
    ])

    R_r = np.column_stack([
        hat(v_bar)@v / v_barTv,
        (np.eye(3) - v[:,None]@v[None,:])@v_bar / v_barTv,
        v
    ])

    return R_e, R_r

def get_Rr_2dot(v, v_dot, v_2dot, v_bar, v_bar_dot, v_bar_2dot):
    den = s(v_bar, v)
    den_dot = s_dot(v, v_bar, v_dot, v_bar_dot)
    den_2dot = s_2dot(v, v_bar, v_dot, v_bar_dot, v_2dot, v_bar_2dot)

    num = hat(v_bar)@v
    num_dot = hat(v_bar_dot)@v + hat(v_bar)@v_dot
    num_2dot = hat(v_bar_2dot)@v + hat(v_bar_dot)@v_dot + hat(v_bar_dot)@v_dot + hat(v_bar)@v_2dot

    Rrd1 = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    num = (np.eye(3) - v[:,None]@v[None,:])@v_bar

    num1 = v_dot[:,None]@v[None,:] + v[:,None]@v_dot[None,:]
    num1_dot = v_2dot[:,None]@v[None,:] + 2*(v_dot[:,None]@v_dot[None,:]) + v[:,None]@v_2dot[None,:]

    num_dot = -num1@v_bar + (np.eye(3) - v[:,None]@v[None,:])@v_bar_dot
    num_2dot = (-num1_dot@v_bar - num1@v_bar_dot + 
               (- v_dot[:,None]@v[None,:] - v[:,None]@v_dot[None,:])@v_bar_dot + 
               (np.eye(3) - v[:,None]@v[None,:])@v_bar_2dot)
    Rrd2 = diff2_num_den(num, num_dot, num_2dot, den, den_dot, den_2dot)

    Rrd3 = v_2dot

    return np.column_stack([Rrd1, Rrd2, Rrd3])

def s_2dot(a, b, a_dot, b_dot, a_2dot, b_2dot):
    num = -a.T@b*(a_dot.T@b + a.T@b_dot)
    num_dot = (-a_dot.T@b*(a_dot.T@b + a.T@b_dot) - 
              a.T@b_dot*(a_dot.T@b + a.T@b_dot) - 
              a.T@b*(a_2dot.T@b + 2*a_dot.T@b_dot + a.T@b_2dot))

    den = s(a, b)
    den_dot = s_dot(a, b, a_dot, b_dot)

    return diff_num_den(num, num_dot, den, den_dot)

def get_Rr_dot(v, v_dot, v_bar, v_bar_dot):
    den = s(v_bar, v)
    den_dot = s_dot(v, v_bar, v_dot, v_bar_dot)

    num = hat(v_bar)@v
    num_dot = hat(v_bar_dot)@v + hat(v_bar)@v_dot
    Rrd1 = diff_num_den(num, num_dot, den, den_dot)
    if norm(den) < 1e-3:
        Rrd1 *= 0

    num = (np.eye(3) - v[:,None]@v[None,:])@v_bar
    num_dot = -(v_dot[:,None]@v[None,:] + v[:,None]@v_dot[None,:])@v_bar + (np.eye(3) - v[:,None]@v[None,:])@v_bar_dot
    Rrd2 = diff_num_den(num, num_dot, den, den_dot)
    if norm(den) < 1e-3:
        Rrd2 *= 0

    Rrd3 = v_dot

    return np.column_stack([Rrd1, Rrd2, Rrd3])

def s_dot(a, b, a_dot, b_dot):
    return -a.T@b*(a_dot.T@b + a.T@b_dot) / s(a, b)

def s(a, b):
    return np.sqrt(1.0 - (a.T@b)**2)

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

def saturate_u(u, a, b):
    u_sat = np.zeros(3)

    for i in range(3):
        ui = u[i]
        ai = a[i]
        bi = b[i]
        
        e = 0.01
        e_upper = (bi - ai) / 2
        if e > e_upper:
            e = e_upper
        
        if ai + e < ui and ui < bi - e:
            u_sat[i] = u[i]
        elif ui <= ai - e:
            u_sat[i] = ai
        elif bi + e <= ui:
            u_sat[i] = bi
        elif ai - e < ui and ui <= ai + e:
            u_sat[i] = ui + 1 / (4*e) * (ui - (ai + e))**2
        elif bi - e <= ui and ui < bi + e:
            u_sat[i] = ui - 1 / (4*e) * (ui - (bi - e))**2
    
    return u_sat

def split_to_states(X):
    x = X[0:3]
    v = X[3:6]
    R = X[6:15].reshape(3, 3)
    W = X[15:18]
    return x, v, R, W

def thr_to_fM(thr, param):
    d = param.d
    ctf = param.c_tf

    f_to_fM = np.array([
        [1, 1, 1, 1],
        [0, -d, 0, d],
        [d, 0, -d, 0],
        [-ctf, ctf, -ctf, ctf]
    ])

    fM = f_to_fM @ thr
    return fM[0], fM[1:4]

def hat(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def command_line(t):
    height = 5
    v = 4
    w = 0

    desired = {
        'x': np.array([v*t, 0, -height]),
        'v': np.array([v, 0, 0]),
        'x_2dot': np.array([0, 0, 0]),
        'x_3dot': np.array([0, 0, 0]),
        'x_4dot': np.array([0, 0, 0]),
        'w': w,
        'w_dot': 0,
        'yaw': w*t,
        'b1': np.array([np.cos(w * t), np.sin(w * t), 0]),
        'b1_dot': w * np.array([-np.sin(w * t), np.cos(w * t), 0]),
        'b1_2dot': w**2 * np.array([-np.cos(w * t), -np.sin(w * t), 0])
    }
    return desired

def deriv_unit_vector(q, q_dot, q_ddot):
    nq = norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * (q.T@q_dot) / nq**3

    u_ddot = (q_ddot / nq - q_dot / nq**3 * (2 * q.T@q_dot) - 
             q / nq**3 * (q_dot.T@q_dot + q.T@q_ddot) + 
             3 * q / nq**5 * (q.T@q_dot)**2)

    return u, u_dot, u_ddot

def vee(S):
    return np.array([-S[1,2], S[0,2], -S[0,1]])

def generate_output_arrays(N):
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
        'R': np.zeros((3, 3, N))
    }

    R = np.zeros((3, 3, N))
    f = np.zeros(N)
    M = np.zeros((3, N))

    return error, desired, R, f, M

def plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(x, y[i,:], linetype, linewidth=linewidth)
    plt.xlabel(xlabel_)
    plt.suptitle(title_)
    plt.subplot(3, 1, 2)
    plt.ylabel(ylabel_)

def plot_4x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, font_size=10):
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(x, y[i,:], linetype, linewidth=linewidth)
    plt.xlabel(xlabel_)
    plt.suptitle(title_)
    plt.subplot(4, 1, 2)
    plt.ylabel(ylabel_)

if __name__ == "__main__":
    run_kooijman()
    plt.show()