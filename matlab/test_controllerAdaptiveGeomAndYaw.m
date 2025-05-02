function adaptiveGeometricController
%% Simulation mode
% Uncomment to use geometric adaptive decoupled-yaw controller.
param.use_decoupled_controller = true;

% Uncomment to use geometric adaptive coupled-yaw controller in
% Geometric adaptive tracking control of a quadrotor unmanned aerial 
% vehicle on {SE(3)}, F. Goodarzi, D. Lee, and T. Lee.
% param.use_decoupled_controller = false;
%% Disturbances
% Uncomment to use disturbances.
param.use_disturbances = true;

% Uncomment to remove disturbances.
% param.use_disturbances = false;
%% Simulation parameters
t = 0:0.01:10;
N = length(t);

% Quadrotor
J1 = 0.02;
J2 = 0.02;
J3 = 0.04;
param.J = diag([J1, J2, J3]);
param.m = 2;
param.g = 9.81;

param.d = 0.169;
param.c_tf = 0.0135;

% Fixed disturbance
if param.use_disturbances
    param.W_x = eye(3);
    param.theta_x = [1, 0.8, -1]';

    param.W_R = eye(3);
    param.theta_R = [0.1, 0.1, -0.1]';
else
    param.W_x = zeros(3);
    param.theta_x = [0, 0, 0]';

    param.W_R = eye(3);
    param.theta_R = [0, 0, 0]';
end
%% Controller gains
k.x = 12;
k.v = 8;
k.R = 6;
k.W = 2;
k.y = 2;
k.wy = 0.8;
param.gamma_x = 2;
param.gamma_R = 10;
param.B_theta_x = 10;
%%
param.c1 = min(sqrt(k.x / param.m), ...
    4 * k.x * k.v / (k.v ^2 + 4 * param.m * k.x));

B2 = 1;
J1 = param.J(1,1);
param.c2 = min(sqrt(k.R / J1), ...
    4 * k.R * k.W / ((k.W + J1 * B2)^2 + 4 * J1 * k.R));

J3 = param.J(3,3);
param.c3 = min(sqrt(k.y / J3), ...
    4 * k.y * k.wy / (k.wy^2 + 4 * J1 * k.R));
%% Initial conditions
x0 = [1, -1, 0]'; % for circle trajectory
% x0 = [0, 0, 0]'; % for line trajectory

v0 = [0, 0, 0]';
e3 = [0, 0, 1]';
R0 = expm((pi - 0.01) * hat(e3));
W0 = [0, 0, 0]';

X0 = [x0; v0; W0; reshape(R0,9,1); zeros(6,1)];
%% Numerical integration
[t, X] = ode45(@(t, XR) eom(t, XR, k, param), t, X0, ...
    odeset('RelTol', 1e-6, 'AbsTol', 1e-6));
%% Output arrays
% Create empty arrays to save data
[e, d, R, f, M] = generate_output_arrays(N);
%% Post processing
x = X(:, 1:3)';
v = X(:, 4:6)';
W = X(:, 7:9)';
theta_x = X(:, 19:21)';
theta_R = X(:, 22:24)';

b1 = zeros(3, N);
b1c = zeros(3, N);

thr = zeros(4, N);

avg_ex = 0;
avg_eR = 0;
avg_f = 0;

converge_t = 0;
is_converged = false;
converge_ex = 0.02;

for i = 1:N
    R(:,:,i) = reshape(X(i,10:18), 3, 3);
    
    des = command(t(i));
    [f(i), M(:,i), ~, ~, err, calc] = position_control(X(i,:)', des, ...
        k, param);
    
    % Unpack errors
    e.x(:,i) = err.x;
    e.v(:,i) = err.v;
    e.R(:,i) = err.R;
    e.W(:,i) = err.W;
    
    if param.use_decoupled_controller
        e.y(i) = err.y;
        e.Wy(i) = err.Wy;
    end
    
    [f(i), M(:,i)] = saturate_fM(f(i), M(:,i), param);
    thr(:,i) = fM_to_thr(f(i), M(:,i), param);
    
    % Unpack desired values
    d.x(:,i) = des.x;
    d.v(:,i) = des.v;
    d.b1(:,i) = des.b1;
    d.R(:,:,i) = calc.R;
    b1(:,i) = R(:,:,i) * [1, 0, 0]';
    b1c(:,i) = calc.b1;
    
    norm_ex = norm(err.x);
    norm_eR = norm(err.R);
    
    % Find normalized errors
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
plot_3x1(t, e.R, '', xlabel_, 'e_R', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(2);
plot_3x1(t, e.x, '', xlabel_, 'e_x', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(3);
plot_3x1(t, x, '', xlabel_, 'x', linetype, linewidth)
plot_3x1(t, d.x, '', xlabel_, 'x', 'r:', linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(4);
plot_3x1(t, theta_x - param.theta_x, '', xlabel_, ...
    '\tilde\theta_x', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(5);
plot_3x1(t, theta_R - param.theta_R, '', xlabel_, ...
    '\tilde\theta_R', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

figure(6);
plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)
% set(gca, 'FontName', 'Times New Roman');

%% Save data
if param.use_decoupled_controller
    save('decoupled.mat');
else
    save('coupled.mat');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M, dot_theta_R, eR, eW] = attitude_control_coupled(R, W, ...
    theta_R, Rd, Wd, Wddot, k, param)
# Based on: 2010, Taeyoung Lee, Melvin Leok, and N. Harris McClamroch
# "Geometric tracking control of a quadrotor UAV on SE(3)"
#  https://ieeexplore.ieee.org/abstract/document/5717652
#
# The control law is based on the attitude control law for rigid bodies
# with a coupling term that depends on the desired angular velocity
# and its derivative.
#
eR = 1 / 2 * vee(Rd' * R - R' * Rd);                    # (10)
eW = W - R' * Rd * Wd;                                  # (11)
M = - k.R * eR ...                                      # (16)
    - k.W * eW ...
    - param.W_R * theta_R ...
    + hat(R' * Rd * Wd) * param.J * R' * Rd * Wd ...
    + param.J * R' * Rd * Wddot;

dot_theta_R = param.gamma_R * param.W_R' * (eW + param.c2 * eR);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M, dot_theta_R, eb, ew, ey, ewy] = attitude_control(...
    R, W, bar_theta_R, ...  % states
    b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, ...  % desired values
    k, param ...  % gains and parameters
)
%% Unpack other parameters
J = param.J;
c2 = param.c2;
c3 = param.c3;

gamma_R = param.gamma_R;
W_R = param.W_R;

W_R_1 = W_R(1,:);
W_R_2 = W_R(2,:);
W_R_3 = W_R(3,:);
%% Body axes
e1 = [1, 0, 0]';
e2 = [0, 1 ,0]';
e3 = [0, 0, 1]';

b1 = R * e1;
b2 = R * e2;
b3 = R * e3;
%% Roll/pitch dynamics
kb = k.R;
kw = k.W;

w = W(1) * b1 + W(2) * b2;                                          # (23)
b3_dot = hat(w) * b3;                                               # (22)

wd = hat(b3d) * b3d_dot;
wd_dot = hat(b3d) * b3d_ddot;

eb = hat(b3d) * b3;                                                 # (27)
ew = w + hat(b3)^2 * wd;                                            # (28)
tau = - kb * eb ...                                                 # (31)
    - kw * ew ...
    - J(1,1) * dot(b3, wd) * b3_dot ...
    - J(1,1) * hat(b3)^2 * wd_dot ...
    - W_R_1 * bar_theta_R * b1 - W_R_2 * bar_theta_R * b2;

tau1 = dot(b1, tau);
tau2 = dot(b2, tau);

M1 = tau1 + J(3,3) * W(3) * W(2);                                   # (24)
M2 = tau2 - J(3,3) * W(3) * W(1);                                   # (24)
%% Yaw dynamics
ey = -dot(b2, b1c);                                                 # (49)
ewy = W(3) - wc3;                                                   # (50)

M3 = - k.y * ey ...                                                 # (52)
    - k.wy * ewy ...
    - W_R_3 * bar_theta_R ...
    + J(3,3) * wc3_dot;
%% Attitude adaptive term
ew_c2eb = ew + c2 * eb;
dot_theta_R = gamma_R * W_R_1' * ew_c2eb' * b1 ...
    + gamma_R * W_R_2' * ew_c2eb' * b2 ...
    + gamma_R * W_R_3' * (ewy + c3 * ey);

M = [M1, M2, M3]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, M, bar_theta_x_dot, theta_R_dot, error, calculated] = ...
    position_control(X, desired, k, param)

[x, v, R, W, bar_theta_x, bar_theta_R] = split_to_states(X);

c1 = param.c1;
m = param.m;
g = param.g;

W_x = param.W_x;
W_x_dot = zeros(3);
W_x_2dot = zeros(3);

e3 = [0, 0, 1]';

%%
error.x = x - desired.x;
error.v = v - desired.v;
A = - k.x * error.x ...
    - k.v * error.v ...
    - m * g * e3 ...
    + m * desired.x_2dot ...
    - W_x * bar_theta_x;
%%
gamma_x = param.gamma_x;
c1 = param.c1;
ev_c1ex = error.v + c1 * error.x;

norm_theta_x = norm(bar_theta_x);
if norm_theta_x < param.B_theta_x || ...
    (norm_theta_x == param.B_theta_x && bar_theta_x' * W_x' * ev_c1ex <= 0)
    
    bar_theta_x_dot = gamma_x * W_x' * ev_c1ex;
else
    I_theta = eye(3) ...
        - bar_theta_x * bar_theta_x' / (bar_theta_x' * bar_theta_x);
    bar_theta_x_dot = gamma_x * I_theta * W_x' * ev_c1ex;
end
%%
b3 = R * e3;
f = -dot(A, b3);
ev_dot = g * e3 ...
    - f / m * b3 ...
    - desired.x_2dot ...
    + W_x * bar_theta_x / m;
A_dot = - k.x * error.v ...
    - k.v * ev_dot ...
    + m * desired.x_3dot ...
    - W_x_dot * bar_theta_x ...
    - W_x * bar_theta_x_dot;
%%
norm_theta_x = norm(bar_theta_x);
if norm_theta_x < param.B_theta_x || ...
    (norm_theta_x == param.B_theta_x && bar_theta_x' * W_x' * ev_c1ex <= 0)
    
    bar_theta_x_2dot = gamma_x * W_x_dot' * ev_c1ex ...
        + gamma_x * W_x' * (ev_dot + c1 * error.v);
else
    I_theta = eye(3) ...
        - bar_theta_x * bar_theta_x' / (bar_theta_x' * bar_theta_x);
    
    num = norm_theta_x * (bar_theta_x_dot * bar_theta_x' ...
        + bar_theta_x * bar_theta_x_dot') ...
        - 2 * (bar_theta_x * bar_theta_x') * bar_theta_x_dot;
    I_theta_dot = - num / norm_theta_x^3;
    bar_theta_x_2dot = gamma_x * I_theta_dot * W_x' * ev_c1ex ...
        + gamma_x * I_theta * W_x_dot' * ev_c1ex ...
        + gamma_x * I_theta * W_x' * (ev_dot + c1 * error.v);
end
%%
b3_dot = R * hat(W) * e3;
f_dot = -dot(A_dot, b3) - dot(A, b3_dot);
ev_2dot = - f_dot / m * b3 - f / m * b3_dot - desired.x_3dot ...
    + W_x_dot * bar_theta_x / m + W_x * bar_theta_x_dot / m;
A_ddot = - k.x * ev_dot ...
    - k.v * ev_2dot ...
    + m * desired.x_4dot ...
    - W_x_2dot * bar_theta_x ...
    - 2 * W_x_dot * bar_theta_x_dot ...
    - W_x * bar_theta_x_2dot;
%%
[b3c, b3c_dot, b3c_ddot] = deriv_unit_vector(-A, -A_dot, -A_ddot);

A2 = -hat(desired.b1) * b3c;
A2_dot = -hat(desired.b1_dot) * b3c - hat(desired.b1) * b3c_dot;
A2_ddot = - hat(desired.b1_2dot) * b3c ...
    - 2 * hat(desired.b1_dot) * b3c_dot ...
    - hat(desired.b1) * b3c_ddot;

[b2c, b2c_dot, b2c_ddot] = deriv_unit_vector(A2, A2_dot, A2_ddot);

b1c = hat(b2c) * b3c;
b1c_dot = hat(b2c_dot) * b3c + hat(b2c)*b3c_dot;
b1c_ddot = hat(b2c_ddot) * b3c ...
    + 2 * hat(b2c_dot) * b3c_dot ...
    + hat(b2c) * b2c_ddot;
%%
Rc = [b1c, b2c, b3c];
Rc_dot = [b1c_dot, b2c_dot, b3c_dot];
Rc_ddot = [b1c_ddot, b2c_ddot, b3c_ddot];
%%
Wc = vee(Rc' * Rc_dot);
Wc_dot = vee(Rc' * Rc_ddot - hat(Wc)^2);
%%
W3 = dot(R * e3, Rc * Wc);
W3_dot = dot(R * e3, Rc * Wc_dot) ...
    + dot(R * hat(W) * e3, Rc * Wc);
%% Run attitude controller
if param.use_decoupled_controller
    [M, theta_R_dot, error.R, error.W, error.y, error.Wy] ...
        = attitude_control( ...
        R, W, bar_theta_R, ...
        b3c, b3c_dot, b3c_ddot, b1c, W3, W3_dot, ...
        k, param);
    
    % For comparison with non-decoupled controller
    error.R = 1 / 2 * vee(Rc' * R - R' * Rc);
else
    [M, theta_R_dot, error.R, error.W] = attitude_control_coupled(...
        R, W, bar_theta_R, Rc, Wc, Wc_dot, k, param);
end
%% Saving data
calculated.b3 = b3c;
calculated.b3_dot = b3c_dot;
calculated.b3_ddot = b3c_ddot;
calculated.b1 = b1c;
calculated.R = Rc;
calculated.W = Wc;
calculated.W_dot = Wc_dot;
calculated.W3 = dot(R * e3, Rc * Wc);
calculated.W3_dot = dot(R * e3, Rc * Wc_dot) ...
    + dot(R * hat(W) * e3, Rc * Wc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command(t)

% desired = command_line(t);
desired = command_circle(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xdot = eom(t, X, k, param)

e3 = [0, 0, 1]';
m = param.m;
J = param.J;

W_x = param.W_x;
W_R = param.W_R;

[~, v, R, W, ~, ~] = split_to_states(X);

desired = command(t);
[f, M, bar_theta_x_dot, bar_theta_R_dot, ~, ~] = position_control(X, ...
    desired, k, param);

[f, M] = saturate_fM(f, M, param);

xdot = v;
vdot = param.g * e3 ...
    - f / m * R * e3 + W_x * param.theta_x / m;
Wdot = J \ (-hat(W) * J * W + M + W_R * param.theta_R);
Rdot = R * hat(W);

if ~param.use_disturbances
    bar_theta_x_dot = 0*bar_theta_x_dot;
    bar_theta_R_dot = 0*bar_theta_R_dot;
end

Xdot=[xdot; vdot; Wdot; reshape(Rdot,9,1); ...
    bar_theta_x_dot; bar_theta_R_dot];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function thr = fM_to_thr(f, M, param)
d = param.d;
ctf = param.c_tf;

f_to_fM = [1, 1, 1, 1;
           0, -d, 0, d;
           d, 0, -d, 0;
           -ctf, ctf, -ctf, ctf];

thr = f_to_fM \ [f; M];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, v, R, W, bar_theta_x, bar_theta_R] = split_to_states(X)

x = X(1:3);
v = X(4:6);
W = X(7:9);
R = reshape(X(10:18), 3, 3);
bar_theta_x = X(19:21);
bar_theta_R = X(22:24);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command_circle(t)
rad = 1;
w = 2*pi / 10;
height = 1;

desired.x = rad*[cos(w*t) - 1, sin(w*t), - height]';
desired.v = w*rad*[-sin(w*t), cos(w*t), 0]';
desired.x_2dot = w^2*rad*[-cos(w*t), -sin(w*t), 0]';
desired.x_3dot = w^3*rad*[sin(w*t), -cos(w*t), 0]';
desired.x_4dot = w^4*rad*[cos(w*t), sin(w*t), 0]';

w = 2*pi / 40;
desired.b1 = [cos(w*t), sin(w*t), 0]';
desired.b1_dot = w*[-sin(w*t), cos(w*t), 0]';
desired.b1_2dot = w^2*[-cos(w*t), -sin(w*t), 0]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u, u_dot, u_ddot] = deriv_unit_vector(q, q_dot, q_ddot)

nq = norm(q);
u = q / nq;
u_dot = q_dot / nq - q * dot(q, q_dot) / nq^3;

u_ddot = q_ddot / nq - q_dot / nq^3 * (2 * dot(q, q_dot))...
    - q / nq^3 * (dot(q_dot, q_dot) + dot(q, q_ddot))...
    + 3 * q / nq^5 * dot(q, q_dot)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hat_x = hat(x)

hat_x = [0 -x(3) x(2);
    x(3) 0 -x(1);
    -x(2) x(1) 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = vee(S)
s = [-S(2,3); S(1,3); -S(1,2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_3x3(x, y, title_, xlabel_, ylabel_, linetype, linewidth, ...
    font_size, desired)

if nargin < 8
    font_size = 10;
    desired = false;
elseif nargin < 9
    desired = false;
end

for i = 1:3
    for j = 1:3
        k = 3*(i - 1) + j;
        subplot(3, 3, k)
        
        if desired
            plot(x, squeeze(y(i,j,:)), linetype, ...
                'LineWidth', linewidth, 'Color', [1, 0, 0]);
        else
            plot(x, squeeze(y(i,j,:)), linetype, 'LineWidth', linewidth);
        end  
        % set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
        ylim([-1 1]);
        hold on;
    end
end

title(title_);

subplot(3, 3, 8);
xlabel(xlabel_, 'interpreter', 'latex');

subplot(3, 3, 4);
ylabel(['$' ylabel_ '$'], 'interpreter', 'latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = sat(sigma, y)

for k = 1:length(y)
    if y > sigma
        z = sigma;
    elseif y < -sigma
        z = -sigma;
    else
        z = y;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = satdot(sigma, y, ydot)

for k = 1:length(y)
    if y > sigma
        z = 0;
    elseif y < -sigma
        z = 0;
    else
        z = ydot;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x_sat = saturate(x, x_min, x_max)

if x > x_max
    x_sat = x_max;
elseif x < x_min
    x_sat = x_min;
else
    x_sat = x;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





