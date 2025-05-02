function run_brescianini

% addpath('../Common');
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

%% Fixed disturbance

% Uncomment to use disturbances.
% param.use_disturbances = true;

% Uncomment to remove disturbances.
param.use_disturbances = false;

param.W_x = eye(3);
param.theta_x = [1, 0.8, -1]';

param.W_R = eye(3);
param.theta_R = [0.1, 0.1, -0.1]';

%% Controller gains for Brescianini
% Position
kx = 24;
kv = 14;
k.x = diag([kx, kx, 12]);
k.v = diag([kv, kv, 8]);

% Attitude
param.kp_xy = 24;
param.kd_xy = 0.8;
param.kp_z = 0.7;
param.kd_z = 0.3;

%% Initial conditions
x0 = [0, 0, 0]';  % for line
% x0 = [1, -1, 0]';  % for circle
v0 = [0, 0, 0]';
R0 = expm((pi - 0.01) * hat([0, 0, 1]'));
W0 = [0, 0, 0]';
X0 = [x0; v0; W0; reshape(R0,9,1)];

%% Numerical integration
[t, X] = ode45(@(t, XR) eom_brescianini(t, XR, k, param), t, X0, ...
    odeset('RelTol', 1e-6, 'AbsTol', 1e-6));

%% Output arrays
% Create empty arrays to save data
[e, d, R, f, M] = generate_output_arrays(N);
thr = zeros(4, N);

%% Post processing
x = X(:, 1:3)';
v = X(:, 4:6)';
W = X(:, 7:9)';

avg_ex = 0;
avg_eR = 0;
avg_f = 0;

converge_t = 0;
is_converged = false;
converge_ex = 0.02;

for i = 1:N
    R(:,:,i) = reshape(X(i,10:18), 3, 3);
    
    des = command(t(i));
    [f(i), M(:,i), err, calc] = position_control(X(i,:)', des, ...
        k, param);
    
    [f(i), M(:,i)] = saturate_fM(f(i), M(:,i), param);
    thr(:,i) = fM_to_thr(f(i), M(:,i), param);
    
    % Unpack errors
    e.x(:,i) = err.x;
    e.v(:,i) = err.v;
    e.R(:,i) = err.R;
    e.W(:,i) = W(:,i) - calc.W;

    % Unpack desired values
    d.x(:,i) = des.x;
    d.v(:,i) = des.v;
    d.b1(:,i) = des.b1;
    d.R(:,:,i) = calc.R;
    d.W(:,i) = calc.W;
    
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
plot_3x1(t, e.R, '', xlabel_, 'e_R', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

figure(2);
plot_3x1(t, e.x, '', xlabel_, 'e_x', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

figure(3);
plot_4x1(t, thr, '', xlabel_, 'f', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

save('brescianini.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xdot = eom_brescianini(t, X, k, param)

e3 = [0, 0, 1]';
m = param.m;
J = param.J;

[~, v, R, W] = split_to_states(X);

desired = command(t);
[f, M, ~, ~] = position_control(X, desired, k, param);

[f, M] = saturate_fM(f, M, param);

x_dot = v;
v_dot = param.g * e3 ...
    - f / m * R * e3;
if param.use_disturbances
     v_dot = v_dot + param.W_x * param.theta_x / m;
end

if param.use_disturbances
    W_dot = J \ (-hat(W) * J * W + M + param.W_R * param.theta_R);
else
    W_dot = J \ (-hat(W) * J * W + M);
end
R_dot = R * hat(W);

Xdot = [x_dot; v_dot; W_dot; reshape(R_dot,9,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f_hat, tau_hat] = saturate_fM(f, tau, param)

l = param.d;
c_tf = param.c_tf;

u_max = 8;
u_min = 0.1;

tau_hat = [0, 0, 0]';
tau_max_xy = (u_max - u_min)*l;
for i = 1:2
    tau_hat(i) = saturate(tau(i), -tau_max_xy, tau_max_xy);
end

tau_hat_x = tau_hat(1);
tau_hat_y = tau_hat(2);
f_min = 4*u_min + abs(tau_hat_x)/l + abs(tau_hat_y)/l;
f_max = 4*u_max - abs(tau_hat_x)/l - abs(tau_hat_y)/l;
f_hat = saturate(f, f_min, f_max);

tau_min_z_list = c_tf*[4*u_min - f_hat + 2*abs(tau_hat_x)/l;
    -4*u_max + f_hat + 2*abs(tau_hat_y)/l];
tau_min_z = max(tau_min_z_list);

tau_max_z_list = c_tf*[4*u_max - f_hat - 2*abs(tau_hat_x)/l;
    -4*u_min + f_hat - 2*abs(tau_hat_y)/l];
tau_max_z = min(tau_max_z_list);

tau_hat(3) = saturate(tau(3), tau_min_z, tau_max_z);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, v, R, W] = split_to_states(X)

x = X(1:3);
v = X(4:6);
W = X(7:9);
R = reshape(X(10:18), 3, 3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, M, error, calculated] = position_control(X, desired, k, param)
% Unpack states
[x, v, R, W] = split_to_states(X);

m = param.m;
g = param.g;
e3 = [0, 0, 1]';

error.x = x - desired.x;
error.v = v - desired.v;
A = - k.x * error.x ...
    - k.v * error.v ...
    - m * g * e3 ...
    + m * desired.x_2dot;

b3 = R * e3;
f = -dot(A, b3);
ea = g * e3 ...
    - f / m * b3 ...
    - desired.x_2dot;
A_dot = - k.x * error.v ...
    - k.v * ea ...
    + m * desired.x_3dot;

b3_dot = R * hat(W) * e3;
f_dot = -dot(A_dot, b3) - dot(A, b3_dot);
eb = - f_dot / m * b3 - f / m * b3_dot - desired.x_3dot;
A_ddot = - k.x * ea ...
    - k.v * eb ...
    + m * desired.x_4dot;

[b3c, b3c_dot, b3c_ddot] = deriv_unit_vector(-A, -A_dot, -A_ddot);

A2 = -hat(desired.b1) * b3c;
A2_dot = -hat(desired.b1_dot) * b3c - hat(desired.b1) * b3c_dot;
A2_ddot = - hat(desired.b1_2dot) * b3c ...
    - 2 * hat(desired.b1_dot) * b3c_dot ...
    - hat(desired.b1) * b3c_ddot;

[b2c, b2c_dot, b2c_ddot] = deriv_unit_vector(A2, A2_dot, A2_ddot);

b1c = hat(b2c) * b3c;
b1c_dot = hat(b2c_dot) * b3c+hat(b2c)*b3c_dot;
b1c_ddot = hat(b2c_ddot) * b3c ...
    + 2 * hat(b2c_dot) * b3c_dot ...
    + hat(b2c) * b2c_ddot;

Rc = [b1c, b2c, b3c];
Rc_dot = [b1c_dot, b2c_dot, b3c_dot];
Rc_ddot = [b1c_ddot, b2c_ddot, b3c_ddot];

Wc = vee(Rc' * Rc_dot);
Wc_dot = vee(Rc' * Rc_ddot - hat(Wc)^2);

%% Run attitude controller
[M, ~] = attitude_control_brescianini(...
    R, W, ...  % states
    Rc, Wc, Wc_dot, ...  % desired values
    param ...  % gains and parameters
);
error.R = 0.5*vee(Rc'*R - R'*Rc);
%% Saving data
calculated.b3 = b3c;
calculated.b3_dot = b3c_dot;
calculated.b3_ddot = b3c_ddot;
calculated.b1 = b1c;
calculated.R = Rc;
calculated.W = Wc;
calculated.W_dot = Wc_dot;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function thr = fM_to_thr(f, M, param)
d = param.d;
ctf = param.c_tf;

f_to_fM = [1, 1, 1, 1;
           0, -d, 0, d;
           d, 0, -d, 0;
           -ctf, ctf, -ctf, ctf];

thr = f_to_fM \ [f; M];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command(t)

desired = command_line(t);
% desired = command_circle(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [tau, error] = attitude_control_brescianini(...
    R, w, ...  % states
    Rd, wd, wd_dot, ...  % desired values
    param ...  % gains and parameters
)
%% Unpack parameters
J = param.J;
kp_xy = param.kp_xy;
kp_z = param.kp_z;
kd_xy = param.kd_xy;
kd_z = param.kd_z;
%%
q = quaternion(R, 'rotmat', 'frame');
qd = quaternion(Rd, 'rotmat', 'frame');

qe = qd*conj(q);

wd_bar = rotmat(qe, 'frame')'*wd;
we = wd_bar - w;

wd_bar_dot = hat(we)*wd_bar + rotmat(qe, 'frame')'*wd_dot;

qe = conj(qe);
[q0, q1, q2, q3] = parts(qe);

q0q3 = sqrt(q0^2 + q3^2);
B = [q0^2 + q3^2;
    q0*q1 - q2*q3;
    q0*q2 + q1*q3;
    0];
qe_red = B / q0q3;
qe_yaw = [q0; 0; 0; q3] / q0q3;

tilde_qe_red = qe_red(2:4);
tilde_qe_yaw = qe_yaw(2:4);

tau_ff = J*wd_bar_dot - hat(J*w)*w;

Kd = diag([kd_xy, kd_xy, kd_z]);
tau = kp_xy*tilde_qe_red ...
    + kp_z*sign(q0)*tilde_qe_yaw ...
    + Kd*we ...
    + tau_ff;
%%
error.qe = qe;
error.we = we;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = vee(S)
s = [-S(2,3); S(1,3); -S(1,2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x_sat = saturate(x, x_min, x_max)

if x > x_max
    x_sat = x_max;
elseif x < x_min
    x_sat = x_min;
else
    x_sat = x;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hat_x = hat(x)

hat_x = [0 -x(3) x(2);
    x(3) 0 -x(1);
    -x(2) x(1) 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u, u_dot, u_ddot] = deriv_unit_vector(q, q_dot, q_ddot)

nq = norm(q);
u = q / nq;
u_dot = q_dot / nq - q * dot(q, q_dot) / nq^3;

u_ddot = q_ddot / nq - q_dot / nq^3 * (2 * dot(q, q_dot))...
    - q / nq^3 * (dot(q_dot, q_dot) + dot(q, q_ddot))...
    + 3 * q / nq^5 * dot(q, q_dot)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%