function test_kooijman
% addpath('../Common');
%% Simulation parameters
t = 0:0.01:10;
N = length(t);

param.m = 2;
param.g = 9.81;
param.d = 0.169;
param.c_tf = 0.0135;

J1 = 0.02;
J2 = 0.02;
J3 = 0.04;
param.J = diag([J1, J2, J3]);

%% Fixed disturbance
% Uncomment to use disturbances.
% param.use_disturbances = true;

% Uncomment to remove disturbances.
param.use_disturbances = false;

param.W_x = eye(3);
param.theta_x = [1, 0.8, -1]';

param.W_R = eye(3);
param.theta_R = [0.1, 0.1, -0.1]';

%% Controller gains
param.kp = 10;
param.kv = 4;
param.k1 = 6;
param.k2 = 2;

param.kW = 8;
param.kwy = 1;
%% Initial conditions
% x0 = [1, -1, 0]';
x0 = [0, 0, 0]';
v0 = [0, 0, 0]';
W0 = [0, 0, 0]';

e3 = [0, 0, 1]';
R0 = expm((pi - 0.01) * hat(e3));

X0 = [x0; v0; reshape(R0,9,1); W0];

%% Numerical integration

[t, X] = ode45(@(t, XR) eom_kooijman(t, XR, param), t, X0, ...
    odeset('RelTol', 1e-6, 'AbsTol', 1e-6, 'MaxStep', 0.1, 'InitialStep', 0.1));

    # print every tenth row of the X array
for i=1:1:10
  str=sprintf("%d ",i);
  str = [str, num2str(X(i,:))];
  disp(str);
end
%% Output arrays
% Create empty arrays to save data
[e, d, R, f, M] = generate_output_arrays(N);
%% Post processing
x = X(:, 1:3)';
v = X(:, 4:6)';
W = X(:, 16:18)';

thr = zeros(4, N);

avg_ex = 0;
avg_eR = 0;
avg_f = 0;

converge_t = 0;
is_converged = false;
converge_ex = 0.02;

for i = 1:N
    R(:,:,i) = reshape(X(i,7:15), 3, 3);
    b1(:,i) = R(:,1,i);
    b3(:,i) = R(:,3,i);

    desired = command(t(i));
    [f(i), M(:,i), err, calc] = position_control_kooijman(X(i,:)', desired, param);

    [f(i), M(:,i)] = saturate_fM(f(i), M(:,i), param);
    thr(:,i) = fM_to_thr(f(i), M(:,i), param);

    % Unpack errors
    e.x(:,i) = -err.x;
    e.v(:,i) = -err.v;

    % Unpack desired values
    d.x(:,i) = desired.x;
    d.v(:,i) = desired.v;

    % Find normalized errors
    norm_ex = norm(err.x);
    norm_eR = norm(err.R);

    avg_ex = avg_ex + norm_ex;
    avg_eR = avg_eR + norm_eR;

    norm_f = norm(thr(:,i));
    avg_f = avg_f + norm_f;

    if norm_ex < converge_ex
        if ~is_converged
            converge_t = t(i);
            is_converged = true;
        end
    end
end
avg_ex = avg_ex / N
avg_eR = avg_eR / N
avg_f = avg_f / N
converge_t

%% Plots

linetype = 'k';
linewidth = 1;
xlabel_ = 'time (s)';

figure(1);
plot_3x1(t, e.x, '', xlabel_, 'e_x', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(2);
plot_3x1(t, e.R, '', xlabel_, 'e_R', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(3);
plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

disp ('')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xdot = eom_kooijman(t, X, param)

e3 = [0, 0, 1]';
m = param.m;
J = param.J;

[~, v, R, W] = split_to_states(X);

desired = command(t);
[T, tau, error, calculated] = position_control_kooijman(X, desired, param);

thr = fM_to_thr(T, tau, param)';
[T, tau] = saturate_fM(T, tau, param);

x_dot = v;
v_dot = param.g*e3 - T*R*e3 / m;
if param.use_disturbances
    v_dot = v_dot + param.W_x*param.theta_x/m;
end

R_dot = R * hat(W);

if param.use_disturbances
    W_dot = J \ (-hat(J*W)*W + tau + param.W_R*param.theta_R);
else
    W_dot = J \ (-hat(W)*J*W + tau);
end

Xdot = [x_dot; v_dot; reshape(R_dot,9,1); W_dot];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [T, tau, error, calculated] = position_control_kooijman(X, desired, param)

[x, v, R, W] = split_to_states(X);

R_dot = R*hat(W);

m = param.m;
J = param.J;
g = param.g;

kp = param.kp;
kv = param.kv;
kW = param.kW;
kwy = param.kwy;

e3 = [0, 0, 1]';

%%
error.x = desired.x - x;
error.v = desired.v - v;

%%
r1 = R(:,1);
r2 = R(:,2);
r3 = R(:,3);

r1_dot = R_dot(:,1);
r2_dot = R_dot(:,2);
r3_dot = R_dot(:,3);

b3 = R*e3;
b3_dot = R_dot*e3;
%%
b = -desired.x_2dot + g*e3;

T_bar = m*norm(b);
T_msqrt3 = T_bar / (sqrt(3)*m);
L_lower = [-T_msqrt3, -T_msqrt3, g - T_msqrt3]';
L_upper = [T_msqrt3, T_msqrt3, g + T_msqrt3]';

u_bar = desired.x_2dot;
% u = saturate_u(u_bar + kp*error.x + kv*error.v, L_lower, L_upper);
u = u_bar + kp*error.x + kv*error.v;

a_bar_ref = g*e3 - u;
n_a_bar_ref = norm(a_bar_ref);

T = m*n_a_bar_ref;
u_bar_dot = desired.x_3dot;
v_dot = g * e3 ...
    - T / m * b3;
error.a = desired.x_2dot - v_dot;
%u_dot = saturate_u(u_bar_dot + kp*error.v + kv*error.a, L_lower, L_upper);
u_dot = u_bar_dot + kp*error.v + kv*error.a;
a_ref_dot = -u_dot;

n_a_ref_dot = a_bar_ref'*a_ref_dot / n_a_bar_ref;
T_dot = m*n_a_ref_dot;
v_2dot = - T_dot / m * b3 ...
    - T / m * b3_dot;
error.a_dot = desired.x_3dot - v_2dot;

u_bar_2dot = desired.x_4dot;
% u_2dot = saturate_u(u_bar_2dot + kp*error.a + kv*error.a_dot, ...
%     L_lower, L_upper);
u_2dot = u_bar_2dot + kp*error.a + kv*error.a_dot;
a_ref_2dot = -u_2dot;

[r3_bar, r3_bar_dot, r3_bar_2dot] = deriv_unit_vector(a_bar_ref, ...
    a_ref_dot, a_ref_2dot);

% phi_bar = atan2(desired.b1(2), desired.b1(1));
phi_bar = desired.yaw;
phi_bar_dot = desired.w;
phi_bar_2dot = desired.w_dot;

r_yaw = [-sin(phi_bar), cos(phi_bar), 0]';
r_yaw_dot = [-cos(phi_bar)*phi_bar_dot;
    -sin(phi_bar)*phi_bar_dot;
    0];
r_yaw_2dot = [sin(phi_bar)*phi_bar_dot^2 + -cos(phi_bar)*phi_bar_2dot;
    -cos(phi_bar)*phi_bar_dot^2 - sin(phi_bar)*phi_bar_2dot;
    0];

num = hat(r_yaw)*r3_bar;
num_dot = hat(r_yaw_dot)*r3_bar + hat(r_yaw)*r3_bar_dot;
num_2dot = hat(r_yaw_2dot)*r3_bar ...
    + hat(r_yaw_dot)*r3_bar_dot ...
    + hat(r_yaw_dot)*r3_bar_dot ...
    + hat(r_yaw)*r3_bar_2dot;

den = s(r_yaw, r3_bar);
den_dot = s_dot(r_yaw, r3_bar, r_yaw_dot, r3_bar_dot);
den_2dot = s_2dot(r_yaw, r3_bar, ...
    r_yaw_dot, r3_bar_dot, ...
    r_yaw_2dot, r3_bar_2dot);

r1_bar = num/den;
r1_bar_dot = diff_num_den(num, num_dot, den, den_dot);
r1_bar_2dot = diff2_num_den(num, num_dot, num_2dot, ...
    den, den_dot, den_2dot);

r2_bar = hat(r3_bar)*r1_bar;

u_v = calculate_u_v(r3, r3_bar, r3_bar_dot, r1, param);
u_w = calculate_u_w(r1, r2, r3, r1_bar, r1_bar_dot, r3_bar, param);

[R_e, R_r] = get_Re_Rr(r3, r3_bar);

% r3_dot = (eye(3) - r3*r3')*u_v;
R_r_dot = get_Rr_dot(r3, r3_dot, r3_bar, r3_bar_dot);
w_r = vee(R_r'*R_r_dot);

R_e_dot = get_Re_dot(r3, r3_dot, r3_bar, r3_bar_dot);
w_e = vee(R_e'*R_e_dot);

W_bar1 = -r2'*u_v;
W_bar2 = r1'*u_v;

if abs(r3'*r3_bar) > 1e-3
    w1 = r1'*R_r*R_e'*r1_bar;
    w2 = r2'*R_r*R_e'*r1_bar;
else
    w1 = r1'*r1_bar;
    w2 = r2'*r2_bar;
end

beta1 = w2*r3'*R_r*R_e'*r1_bar - r1'*R_r*hat(w_r - w_e)*R_e'*r1_bar;
beta2 = w1*r3'*R_r*R_e'*r1_bar + r2'*R_r*hat(w_r - w_e)*R_e'*r1_bar;

if abs(w1) > abs(w2)
    w_r = beta2/w1;
else
    w_r = beta1/w2;
end

W_bar = [W_bar1, W_bar2, u_w + w_r]';
%%
r3_dot = R_dot(:,3);
u_v_dot = calculate_u_v_dot(r3, r3_dot, ...
    r3_bar, r3_bar_dot, r3_bar_2dot, ...
    r1_dot, param);

u_w_dot = calculate_u_w_dot(r1, r1_dot, ...
    r2, r2_dot, ...
    r3, r3_dot, ...
    r1_bar, r1_bar_dot, r1_bar_2dot, ...
    r3_bar, r3_bar_dot, ...
    param);
%%
r3_2dot = (- r3_dot*r3' - r3*r3_dot')*u_v ...
    + (eye(3) - r3*r3')*u_v_dot;
%%
w1_dot = r1_dot'*R_r*R_e'*r1_bar ...
    + r1'*R_r_dot*R_e'*r1_bar ...
    + r1'*R_r*R_e_dot'*r1_bar ...
    + r1'*R_r*R_e'*r1_bar_dot;

w2_dot = r2_dot'*R_r*R_e'*r1_bar ...
    + r2'*R_r_dot*R_e'*r1_bar ...
    + r2'*R_r*R_e_dot'*r1_bar ...
    + r2'*R_r*R_e'*r1_bar_dot;

R_r_2dot = get_Rr_2dot(r3, r3_dot, r3_2dot, ...
    r3_bar, r3_bar_dot, r3_bar_2dot);
R_e_2dot = get_Re_2dot(r3, r3_dot, r3_2dot, ...
    r3_bar, r3_bar_dot, r3_bar_2dot);

w_r_dot = vee(R_r_dot'*R_r_dot) + vee(R_r'*R_r_2dot);
w_e_dot = vee(R_e_dot'*R_e_dot) + vee(R_e'*R_e_2dot);

beta1_dot = w2_dot*r3'*R_r*R_e'*r1_bar ...
    + w2*r3_dot'*R_r*R_e'*r1_bar ...
    + w2*r3'*R_r_dot*R_e'*r1_bar ...
    + w2*r3'*R_r*R_e_dot'*r1_bar ...
    + w2*r3'*R_r*R_e'*r1_bar_dot ...
    - r1_dot'*R_r*hat(w_r - w_e)*R_e'*r1_bar ...
    - r1'*R_r_dot*hat(w_r - w_e)*R_e'*r1_bar ...
    - r1'*R_r*hat(w_r_dot - w_e_dot)*R_e'*r1_bar ...
    - r1'*R_r*hat(w_r - w_e)*R_e_dot'*r1_bar ...
    - r1'*R_r*hat(w_r - w_e)*R_e'*r1_bar_dot;

beta2_dot = w1_dot*r3'*R_r*R_e'*r1_bar ...
    + w1*r3_dot'*R_r*R_e'*r1_bar ...
    + w1*r3'*R_r_dot*R_e'*r1_bar ...
    + w1*r3'*R_r*R_e_dot'*r1_bar ...
    + w1*r3'*R_r*R_e'*r1_bar_dot ...
    + r2_dot'*R_r*hat(w_r - w_e)*R_e'*r1_bar ...
    + r2'*R_r_dot*hat(w_r - w_e)*R_e'*r1_bar ...
    + r2'*R_r*hat(w_r_dot - w_e_dot)*R_e'*r1_bar ...
    + r2'*R_r*hat(w_r - w_e)*R_e_dot'*r1_bar ...
    + r2'*R_r*hat(w_r - w_e)*R_e'*r1_bar_dot;

if abs(w1) > abs(w2)
    w_r_dot = diff_num_den(beta2, beta2_dot, w1, w1_dot);
else
    w_r_dot = diff_num_den(beta1, beta1_dot, w2, w2_dot);
end

W1_dot = - r2_dot'*u_v - r2'*u_v_dot;
W2_dot = r1_dot'*u_v + r1'*u_v_dot;
W3_dot = u_w_dot + w_r_dot;
W_bar_dot = [W1_dot, W2_dot, W3_dot]';
%%
Rd = [r1_bar, r2_bar, r3_bar];

Wd = W_bar;
Wd_dot = W_bar_dot;

kW = diag([kW, kW, kwy]);

eW = W - Wd;
tau = -kW*eW + hat(W)*J*W + J*Wd_dot;

%% Saving data
calculated.b3 = r3_bar;
calculated.b3_dot = r3_bar_dot;
calculated.R = Rd;
error.R = 0.5*vee(Rd'*R - R'*Rd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command(t)

desired = command_line(t);
% desired = command_circle(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u_v_dot = calculate_u_v_dot(v, v_dot, ...
    v_bar, v_bar_dot, v_bar_2dot, ...
    r1_dot, param)

k1 = param.k1;

if (v'*v_bar >= 0)
    u_v_FB_dot = k1*v_bar_dot;
elseif (v == -v_bar)
    u_v_FB_dot = k1*r1_dot;
else
    num = k1*v_bar;
    num_dot = k1*v_bar_dot;
    den = s(v, v_bar);
    den_dot = s_dot(v, v_bar, v_dot, v_bar_dot);
    u_v_FB_dot = diff_num_den(num, num_dot, den, den_dot);
end

if (v == v_bar)
    u_v_FF_dot = v_bar_2dot;
elseif (v == -v_bar)
    u_v_FF_dot = -v_bar_2dot;
else
    vxvbar = cross(v, v_bar);
    vxvbar_dot = cross(v_dot, v_bar) + cross(v, v_bar_dot);

    num = vxvbar*vxvbar' - (eye(3) - v*v')*v_bar*v';
    num_dot = vxvbar_dot*vxvbar' ...
        + vxvbar*vxvbar_dot' ...
        - (- v_dot*v' - v*v_dot')*v_bar*v' ...
        - (eye(3) - v*v')*v_bar_dot*v' ...
        - (eye(3) - v*v')*v_bar*v_dot';

    den = s(v, v_bar)^2;
    den_dot = 2*s(v, v_bar)*s_dot(v, v_bar, v_dot, v_bar_dot);

    theta = num / den;
    theta_dot = diff_num_den(num, num_dot, den, den_dot);

    u_v_FF_dot = theta_dot*v_bar_dot + theta*v_bar_2dot;
end

u_v_dot = u_v_FB_dot + u_v_FF_dot;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u_v = calculate_u_v(v, v_bar, v_bar_dot, r1, param)

k1 = param.k1;

if (v'*v_bar >= 0)
    u_v_FB = k1*v_bar;
elseif (v == -v_bar)
    u_v_FB = k1*r1;
else
    u_v_FB = k1*v_bar / s(v, v_bar);
end

if (v == v_bar)
    u_v_FF = v_bar_dot;
elseif (v == -v_bar)
    u_v_FF = -v_bar_dot;
else
    vxvbar = cross(v, v_bar);
    theta = 1 / s(v, v_bar)^2 ...
        * (vxvbar*vxvbar' - (eye(3) - v*v')*v_bar*v');
    u_v_FF = theta*v_bar_dot;
end

u_v = u_v_FB + u_v_FF;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u_w_dot = calculate_u_w_dot(r1, r1_dot, ...
    r2, r2_dot, ...
    r3, r3_dot, ...
    r1_bar, r1_bar_dot, r1_bar_2dot, ...
    r3_bar, r3_bar_dot, ...
    param)

k2 = param.k2;

[R_e, R_r] = get_Re_Rr(r3, r3_bar);
R_r_dot = get_Rr_dot(r3, r3_dot, r3_bar, r3_bar_dot);
R_e_dot = get_Re_dot(r3, r3_dot, r3_bar, r3_bar_dot);

w1 = r1'*R_r*R_e'*r1_bar;
w1_dot = r1_dot'*R_r*R_e'*r1_bar ...
    + r1'*R_r_dot*R_e'*r1_bar ...
    + r1'*R_r*R_e_dot'*r1_bar ...
    + r1'*R_r*R_e'*r1_bar_dot;

w2 = r2'*R_r*R_e'*r1_bar;
w2_dot = r2_dot'*R_r*R_e'*r1_bar ...
    + r2'*R_r_dot*R_e'*r1_bar ...
    + r2'*R_r*R_e_dot'*r1_bar ...
    + r2'*R_r*R_e'*r1_bar_dot;

if abs(w1) > abs(w2)
    theta2 = r2'*R_r*R_e'*r1_bar_dot;
    theta2_dot = r2_dot'*R_r*R_e'*r1_bar_dot ...
        + r2'*R_r_dot*R_e'*r1_bar_dot ...
        + r2'*R_r*R_e_dot'*r1_bar_dot ...
        + r2'*R_r*R_e'*r1_bar_2dot;

    num = theta2;
    num_dot = theta2_dot;

    den = w1;
    den_dot = w1_dot;

    u_w_FF_dot = diff_num_den(num, num_dot, den, den_dot);
else
    theta1 = -r1'*R_r*R_e'*r1_bar_dot;
    theta1_dot = -r1_dot'*R_r*R_e'*r1_bar_dot ...
        - r1'*R_r_dot*R_e'*r1_bar_dot ...
        - r1'*R_r*R_e_dot'*r1_bar_dot ...
        - r1'*R_r*R_e'*r1_bar_2dot;

    num = theta1;
    num_dot = theta1_dot;

    den = w2;
    den_dot = w2_dot;

    u_w_FF_dot = diff_num_den(num, num_dot, den, den_dot);
end

if w1 >= 0
    u_w_FB_dot = k2*w2_dot;
elseif w1 < 0 && w2 < 0
    u_w_FB_dot = -k2;
else
    u_w_FB_dot = k2;
end

u_w_dot = u_w_FB_dot + u_w_FF_dot;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u_w = calculate_u_w(r1, r2, r3, r1_bar, r1_bar_dot, ...
    r3_bar, param)

k2 = param.k2;

[R_e, R_r] = get_Re_Rr(r3, r3_bar);

w1 = r1'*R_r*R_e'*r1_bar;
w2 = r2'*R_r*R_e'*r1_bar;

if abs(w1) > abs(w2)
    theta2 = r2'*R_r*R_e'*r1_bar_dot;
    u_w_FF = theta2 / w1;
else
    theta1 = -r1'*R_r*R_e'*r1_bar_dot;
    u_w_FF = theta1 / w2;
end

if w1 >= 0
    u_w_FB = k2*w2;
elseif w1 < 0 && w2 < 0
    u_w_FB = -k2;
else
    u_w_FB = k2;
end

u_w = u_w_FB + u_w_FF;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x_dot = diff_num_den(num, num_dot, den, den_dot)
x_dot = (den*num_dot - num*den_dot) / den^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function num_den = diff2_num_den(num, num_den, num_2dot, ...
    den, den_dot, den_2dot)

num = den^2*(den*num_2dot - num*den_2dot) ...
    - (den*num_den - num*den_dot)*2*den*den_dot;
den = den^4;

num_den = num / den;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function thr = fM_to_thr(f, M, param)

d = param.d;
ctf = param.c_tf;

f_to_fM = [1, 1, 1, 1;
           0, -d, 0, d;
           d, 0, -d, 0;
           -ctf, ctf, -ctf, ctf];

thr = f_to_fM \ [f; M];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Re_2dot = get_Re_2dot(v, v_dot, v_2dot, ...
    v_bar, v_bar_dot, v_bar_2dot)

den = s(v_bar, v);
den_dot = s_dot(v, v_bar, v_dot, v_bar_dot);
den_2dot = s_2dot(v, v_bar, v_dot, v_bar_dot, v_2dot, v_bar_2dot);
%%
num = hat(v_bar)*v;
num_dot = hat(v_bar_dot)*v + hat(v_bar)*v_dot;
num_2dot = hat(v_bar_2dot)*v + hat(v_bar_dot)*v_dot ...
    + hat(v_bar_dot)*v_dot + hat(v_bar)*v_2dot;
Rrd1 = diff2_num_den(num, num_dot, num_2dot, ...
    den, den_dot, den_2dot);
%%
num1 = (eye(3) - v_bar*v_bar');
num1_dot = -(v_bar_dot*v_bar' + v_bar*v_bar_dot');
num1_2dot = -(v_bar_2dot*v_bar' + v_bar_dot*v_bar_dot' ...
    + v_bar_dot*v_bar_dot' + v_bar*v_bar_2dot');

num = -num1*v;
num_dot = -num1_dot*v - num1*v_dot;
num_2dot = - num1_2dot*v - 2*(num1_dot*v_dot) - num1*v_2dot;

Rrd2 = diff2_num_den(num, num_dot, num_2dot, ...
    den, den_dot, den_2dot);
%%
Rrd3 = v_bar_2dot;
%%
Re_2dot = [Rrd1, Rrd2, Rrd3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Re_dot = get_Re_dot(v, v_dot, v_bar, v_bar_dot)

den = s(v_bar, v);
den_dot = s_dot(v, v_bar, v_dot, v_bar_dot);

num = hat(v_bar)*v;
num_dot = hat(v_bar_dot)*v + hat(v_bar)*v_dot;
Rrd1 = diff_num_den(num, num_dot, den, den_dot);
if norm(den) < 1e-3
    Rrd1 = Rrd1*0;
end

num = -(eye(3) - v_bar*v_bar')*v;
num_dot = (v_bar_dot*v_bar' + v_bar*v_bar_dot')*v ...
    - (eye(3) - v_bar*v_bar')*v_dot;
Rrd2 = diff_num_den(num, num_dot, den, den_dot);
if norm(den) < 1e-3
    Rrd2 = Rrd2*0;
end

Rrd3 = v_bar_dot;

Re_dot = [Rrd1, Rrd2, Rrd3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R_e, R_r] = get_Re_Rr(v, v_bar)

v_barTv = s(v_bar, v);

R_e = [hat(v_bar)*v / v_barTv, ...
    -(eye(3) - v_bar*v_bar')*v / v_barTv, ...
    v_bar];

R_r = [hat(v_bar)*v / v_barTv, ...
    (eye(3) - v*v')*v_bar / v_barTv, ...
    v];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Rr_2dot = get_Rr_2dot(v, v_dot, v_2dot, ...
    v_bar, v_bar_dot, v_bar_2dot)

den = s(v_bar, v);
den_dot = s_dot(v, v_bar, v_dot, v_bar_dot);
den_2dot = s_2dot(v, v_bar, v_dot, v_bar_dot, v_2dot, v_bar_2dot);
%%
num = hat(v_bar)*v;
num_dot = hat(v_bar_dot)*v + hat(v_bar)*v_dot;
num_2dot = hat(v_bar_2dot)*v + hat(v_bar_dot)*v_dot ...
    + hat(v_bar_dot)*v_dot + hat(v_bar)*v_2dot;

Rrd1 = diff2_num_den(num, num_dot, num_2dot, ...
    den, den_dot, den_2dot);
%%
num = (eye(3) - v*v')*v_bar;

num1 = v_dot*v' + v*v_dot';
num1_dot = v_2dot*v' + 2*(v_dot*v_dot') + v*v_2dot';

num_dot = -num1*v_bar + (eye(3) - v*v')*v_bar_dot;
num_2dot = -num1_dot*v_bar ...
    - num1*v_bar_dot ...
    + (- v_dot*v' - v*v_dot')*v_bar_dot ...
    + (eye(3) - v*v')*v_bar_2dot;
Rrd2 = diff2_num_den(num, num_dot, num_2dot, ...
    den, den_dot, den_2dot);
%%
Rrd3 = v_2dot;
%%
Rr_2dot = [Rrd1, Rrd2, Rrd3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = s_2dot(a, b, a_dot, b_dot, a_2dot, b_2dot)

num = -a'*b*(a_dot'*b + a'*b_dot);
num_dot = -a_dot'*b*(a_dot'*b + a'*b_dot) ...
    - a'*b_dot*(a_dot'*b + a'*b_dot) ...
    - a'*b*(a_2dot'*b + 2*a_dot'*b_dot + a'*b_2dot);

den = s(a, b);
den_dot = s_dot(a, b, a_dot, b_dot);

x = diff_num_den(num, num_dot, den, den_dot);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Rr_dot = get_Rr_dot(v, v_dot, v_bar, v_bar_dot)

den = s(v_bar, v);
den_dot = s_dot(v, v_bar, v_dot, v_bar_dot);

num = hat(v_bar)*v;
num_dot = hat(v_bar_dot)*v + hat(v_bar)*v_dot;
Rrd1 = diff_num_den(num, num_dot, den, den_dot);
if norm(den) < 1e-3
    Rrd1 = Rrd1*0;
end

num = (eye(3) - v*v')*v_bar;
num_dot = -(v_dot*v' + v*v_dot')*v_bar + (eye(3) - v*v')*v_bar_dot;
Rrd2 = diff_num_den(num, num_dot, den, den_dot);
if norm(den) < 1e-3
    Rrd2 = Rrd2*0;
end

Rrd3 = v_dot;

Rr_dot = [Rrd1, Rrd2, Rrd3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = s_dot(a, b, a_dot, b_dot)

x = -a'*b*(a_dot'*b + a'*b_dot) / s(a, b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = s(a, b)

x = sqrt(1.0 - (transpose(a)*b)^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f_sat, M_sat] = saturate_fM(f, M, param)
thr = fM_to_thr(f, M, param);

max_f = 8;
min_f = 0.1;

for i = 1:4
    if thr(i) > max_f
        thr(i) = max_f;
    elseif thr(i) < min_f
        thr(i) = min_f;
    end
end

[f_sat, M_sat] = thr_to_fM(thr, param);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u_sat = saturate_u(u, a, b)

u_sat = zeros(3, 1);

for i = 1:3
    ui = u(i);
    ai = a(i);
    bi = b(i);

    e = 0.01;
    e_upper = (bi - ai) / 2;
    if e > e_upper
        e = e_upper;
    end

    if ai + e < ui && ui < bi - e
        u_sat(i) = u(i);
    elseif ui <= ai - e
        u_sat(i) = ai;
    elseif bi + e <= ui
        u_sat(i) = bi;
    elseif ai - e < ui && ui <= ai + e
        u_sat(i) = ui + 1 / (4*e) * (ui - (ai + e))^2;
    elseif bi - e <= ui && ui < bi + e
        u_sat(i) = ui - 1 / (4*e) * (ui - (bi - e))^2;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, v, R, W] = split_to_states(X)

x = X(1:3);
v = X(4:6);
R = reshape(X(7:15), 3, 3);
W = X(16:18);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, M] = thr_to_fM(thr, param)
d = param.d;
ctf = param.c_tf;

f_to_fM = [1, 1, 1, 1;
           0, -d, 0, d;
           d, 0, -d, 0;
           -ctf, ctf, -ctf, ctf];

fM = f_to_fM * thr;

f = fM(1);
M = fM(2:4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hat_x = hat(x)

hat_x = [0 -x(3) x(2);
    x(3) 0 -x(1);
    -x(2) x(1) 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command_line(t)

height = 5;

v = 4;
desired.x = [v*t, 0, -height]';
desired.v = [v, 0, 0]';
desired.x_2dot = [0, 0, 0]';
desired.x_3dot = [0, 0, 0]';
desired.x_4dot = [0, 0, 0]';

w = 0;
desired.w = w;
desired.w_dot = 0;

desired.yaw = w*t;

desired.b1 = [cos(w * t), sin(w * t), 0]';
desired.b1_dot = w * [-sin(w * t), cos(w * t), 0]';
desired.b1_2dot = w^2 * [-cos(w * t), -sin(w * t), 0]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u, u_dot, u_ddot] = deriv_unit_vector(q, q_dot, q_ddot)

nq = norm(q);
u = q / nq;
u_dot = q_dot / nq - q * dot(q, q_dot) / nq^3;

u_ddot = q_ddot / nq - q_dot / nq^3 * (2 * dot(q, q_dot))...
    - q / nq^3 * (dot(q_dot, q_dot) + dot(q, q_ddot))...
    + 3 * q / nq^5 * dot(q, q_dot)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = vee(S)
s = [-S(2,3); S(1,3); -S(1,2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [error, desired, R, f, M] = generate_output_arrays(N)

error.x = zeros(3, N);
error.v = zeros(3, N);
error.R = zeros(3, N);
error.W = zeros(3, N);
error.y = zeros(1, N);
error.Wy = zeros(1, N);

desired.x = zeros(3, N);
desired.b1 = zeros(3, N);
desired.R = zeros(3, 3, N);

R = zeros(3, 3, N);
f = zeros(1, N);
M = zeros(3, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, ...
    font_size)

if nargin < 8
    font_size = 10;
end

for i = 1:3
    subplot(3, 1, i);
    plot(x, y(i,:), linetype, 'LineWidth', linewidth);
    % set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
    hold on;
end
xlabel(xlabel_, 'interpreter', 'latex');
title(title_);

subplot(3, 1, 2);
ylabel(['$' ylabel_ '$'], 'interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_4x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, ...
    font_size)

if nargin < 8
    font_size = 10;
end

for i = 1:4
    subplot(4, 1, i);
    plot(x, y(i,:), linetype, 'LineWidth', linewidth);
    % set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
    hold on;
end
xlabel(xlabel_, 'interpreter', 'latex');
title(title_);

subplot(4, 1, 2);
ylabel(['$' ylabel_ '$'], 'interpreter', 'latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
