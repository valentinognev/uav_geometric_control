%
% Copyright (c) 2020 Flight Dynamics and Control Lab
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the
% following conditions:
%
% The above copyright notice and this permission notice shall be included
%  in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%%
function test_controllerAllinOne
% addpath('aux_functions');
% addpath('test_functions');

%% Simulation parameters
t = 0:0.01:10;
N = length(t);

% Quadrotor
J1 = 0.02;
J2 = 0.02;
J3 = 0.04;
param.J = diag([J1, J2, J3]);

param.m = 2;

param.d = 0.169;
param.ctf = 0.0135;

% Fixed disturbance
param.x_delta = [0.5, 0.8, -1]';
param.R_delta = [0.2, 1.0, -0.1]';

% Other parameters
param.g = 9.81;

%% Controller gains
k.x = 10;
k.v = 8;
k.i = 10;
param.c1 = 1.5;
param.sigma = 10;

% Attitude
k.R = 1.5;
k.W = 0.35;
k.I = 10;
param.c2 = 2;

% Yaw
k.y = 0.8;
k.wy = 0.15;
k.yI = 2;
param.c3 = 2;

%% Initial conditions
x0 = [0, 0, 0]';
v0 = [0, 0, 0]';
R0 = expm(pi/2 * hat([0, 0, 1]'));
W0 = [0, 0, 0]';
X0 = [x0; v0; W0; reshape(R0,9,1); zeros(6,1)];

%% Numerical integration
[t, X] = ode45(@(t, XR) eom(t, XR, k, param), t, X0, odeset('RelTol', 1e-6, 'AbsTol', 1e-6));

%% Post processing

% Create empty arrays to save data
[e, d, R, f, M] = generate_output_arrays(N);

% Unpack the outputs of ode45 function
x = X(:, 1:3)';
v = X(:, 4:6)';
W = X(:, 7:9)';
ei = X(:, 19:21)';
eI = X(:, 22:24)';

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
    e.y(i) = err.y;
    e.Wy(i) = err.Wy;

    % Unpack desired values
    d.x(:,i) = des.x;
    d.v(:,i) = des.v;
    d.b1(:,i) = des.b1;
    d.R(:,:,i) = calc.R;
end

% Plot data
linetype = 'k';
linewidth = 1;
xlabel_ = 'time (s)';

figure;
plot_3x1(t, e.R, '', xlabel_, 'e_R', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot_3x1(t, e.x, '', xlabel_, 'e_x', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot_3x1(t, e.v, '', xlabel_, 'e_v', linetype, linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot_3x1(t, eI .* [k.I, k.I, k.yI]', '', xlabel_, 'e', linetype, linewidth)
plot_3x1(t, param.R_delta .* ones(3, N), ...
    '', xlabel_, 'e_I', 'r', linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot_3x1(t, ei * k.i, '', xlabel_, 'e_i', linetype, linewidth)
plot_3x1(t, param.x_delta .* ones(3, N), ...
    '', xlabel_, 'e_i', 'r', linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot_3x1(t, x, '', xlabel_, 'x', linetype, linewidth)
plot_3x1(t, d.x, '', xlabel_, 'x', 'r', linewidth)
set(gca, 'FontName', 'Times New Roman');

figure;
plot3(x(1,:), x(2,:), x(3,:), 'k');
hold on;
plot3(d.x(1,:), d.x(2,:), d.x(3,:), 'r');
set(gca, 'YDir', 'reverse', 'ZDir', 'reverse');
axis equal;
xlabel('$x_1$', 'interpreter', 'latex');
ylabel('$x_2$', 'interpreter', 'latex');
zlabel('$x_3$', 'interpreter', 'latex');
set(gca, 'Box', 'on');
grid on;
set(gca, 'FontName', 'Times New Roman');

disp ('')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xdot = eom(t, X, k, param)

e3 = [0, 0, 1]';
m = param.m;
J = param.J;

[~, v, R, W, ~, ~] = split_to_states(X);

desired = command(t);
[f, M, ei_dot, eI_dot, ~, ~] = position_control(X, desired, k, param);

xdot = v;
vdot = param.g * e3 - f / m * R * e3 + param.x_delta / m;
Wdot = J \ (-hat(W) * J * W + M + param.R_delta);
Rdot = R * hat(W);

Xdot=[xdot; vdot; Wdot; reshape(Rdot,9,1); ei_dot; eI_dot];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command(t)

% desired = command_line(t);
desired = command_Lissajous(t);
% desired = command_point(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, M, ei_dot, eI_dot, error_, calculated] = position_control(X, desired, k, param)
% [f, M, ei_dot, eI_dot, error, calculated] = position_control(X, desired, 
% k, param)
%
% Position controller that uses decoupled-yaw controller as the attitude
% controller
% 
%   Caluclates the force and moments required for a UAV to reach a given 
%   set of desired position commands using a decoupled-yaw controller
%   defined in https://ieeexplore.ieee.org/document/8815189.
%   
%   Inputs:
%    X: (24x1 matrix) states of the system (x, v, R, W, ei, eI)
%    desired: (struct) desired states
%    k: (struct) control gains
%    param: (struct) parameters such as m, g, J in a struct
%
%  Outputs:
%    f: (scalar) required motor force
%    M: (3x1 matrix) control moment required to reach desired conditions
%    ei_dot: (3x1 matrix) position integral change rate
%    eI_dot: (3x1 matrix) attitude integral change rate
%    error: (struct) errors for attitude and position control (for data
%    logging)
%    calculated: (struct) calculated desired commands (for data logging)

% Use this flag to enable or disable the decoupled-yaw attitude controller.
use_decoupled = false;

% Unpack states
[x, v, R, W, ei, eI] = split_to_states(X);

sigma = param.sigma;
c1 = param.c1;
m = param.m;
g = param.g;
e3 = [0, 0, 1]';

error_.x = x - desired.x;                                                % (11)
error_.v = v - desired.v;                                                % (12)
A = - k.x * error_.x ...                                                 % (14)
    - k.v * error_.v ...
    - m * g * e3 ...
    + m * desired.x_2dot ...
    - k.i * sat(sigma, ei);

ei_dot = error_.v + c1 * error_.x;                                        % (13)
b3 = R * e3;
f = -dot(A, b3);
ea = g * e3 ...
    - f / m * b3 ...
    - desired.x_2dot ...
    + param.x_delta / m;
A_dot = - k.x * error_.v ...                                ,            % (14)
    - k.v * ea ...
    + m * desired.x_3dot ...
    - k.i * satdot(sigma, ei, ei_dot);

ei_ddot = ea + c1 * error_.v;
b3_dot = R * hat(W) * e3;                                               % (22)
f_dot = -dot(A_dot, b3) - dot(A, b3_dot);
eb = - f_dot / m * b3 - f / m * b3_dot - desired.x_3dot;                % (27)
A_ddot = - k.x * ea ...
    - k.v * eb ...
    + m * desired.x_4dot ...
    - k.i * satdot(sigma, ei, ei_ddot);

[b3c, b3c_dot, b3c_ddot] = deriv_unit_vector(-A, -A_dot, -A_ddot);

A2 = -hat(desired.b1) * b3c;
A2_dot = -hat(desired.b1_dot) * b3c - hat(desired.b1) * b3c_dot;
A2_ddot = - hat(desired.b1_2dot) * b3c ...
    - 2 * hat(desired.b1_dot) * b3c_dot ...
    - hat(desired.b1) * b3c_ddot;

[b2c, b2c_dot, b2c_ddot] = deriv_unit_vector(A2, A2_dot, A2_ddot);

b1c = hat(b2c) * b3c;
b1c_dot = hat(b2c_dot) * b3c + hat(b2c) * b3c_dot;
b1c_ddot = hat(b2c_ddot) * b3c ...
    + 2 * hat(b2c_dot) * b3c_dot ...
    + hat(b2c) * b3c_ddot;

Rc = [b1c, b2c, b3c];
Rc_dot = [b1c_dot, b2c_dot, b3c_dot];
Rc_ddot = [b1c_ddot, b2c_ddot, b3c_ddot];

Wc = vee(Rc' * Rc_dot);
Wc_dot = vee(Rc' * Rc_ddot - hat(Wc)^2);

W3 = dot(R * e3, Rc * Wc);
W3_dot = dot(R * e3, Rc * Wc_dot) + dot(R * hat(W) * e3, Rc * Wc);

%% Run attitude controller
if use_decoupled
    [M, eI_dot, error_.b, error_.W, error_.y, error_.Wy] ...
        = attitude_control_decoupled_yaw( ...
        R, W, eI, ...
        b3c, b3c_dot, b3c_ddot, b1c, W3, W3_dot, ...
        k, param);
    
    % Only used for comparison between two controllers
    error_.R = 1 / 2 * vee(Rc' * R - R' * Rc);
else
    [M, eI_dot, error_.R, error_.W] = attitude_control( ...
        R, W, eI, ...
        Rc, Wc, Wc_dot, ...
        k, param);
    error_.y = 0;
    error_.Wy = 0;
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
calculated.W3_dot = dot(R * e3, Rc * Wc_dot) + dot(R * hat(W) * e3, Rc * Wc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M, eI_dot, eR, eW] = attitude_control( ...
    R, W, eI, ...  % states
    Rd, Wd, Wd_dot, ...  % desired values
    k, param ...  % gains and parameters
)
% [M, eI_dot, eR, eW] = attitude_control(R, W, eI, Rd, Wd, Wddot, k, param)
%
% Attitude controller
% 
%   Caluclates control moments for a given set of desired attitude commands 
%   using a the controller defined in 
%   https://ieeexplore.ieee.org/abstract/document/5717652
%   
%   Inputs:
%    R: (3x3 matrix) current attitude in SO(3)
%    W: (3x1 matrix) current angular velocity
%    eI: (3x1 matrix) attitude integral error
%    Rd: (3x3 matrix) desired attitude in SO(3)
%    Wd: (3x1 matrix) desired body angular velocity
%    Wd_dot: (3x1 matrix) desired body angular acceleration
%    k: (struct) control gains
%    param: (struct) parameters such as m, g, J in a struct
%
%  Outputs:
%    M: (3x1 matrix) control moment required to reach desired conditions
%    eI_dot: (3x1 matrix) attitude integral change rate
%    eR: (3x1 matrix) attitude error
%    eW: (3x1 matrix) angular velocity error

eR = 1 / 2 * vee(Rd' * R - R' * Rd);                    # (10)
eW = W - R' * Rd * Wd;                                  # (11)

kR = diag([k.R, k.R, k.y]);
kW = diag([k.W, k.W, k.wy]);

M = - kR * eR ...                                       # (16)
    - kW * eW ...
    - k.I * eI ...
    + hat(R' * Rd * Wd) * param.J * R' * Rd * Wd ...
    + param.J * R' * Rd * Wd_dot;

eI_dot = eW + param.c2 * eR;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M, eI_dot, eb, ew, ey, ewy] = attitude_control_decoupled_yaw(...
    R, W, eI, ...  % states
    b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, ...  % desired values
    k, param ...  % gains and parameters
)
% [M, eI_dot, eb, ew, ey, ewy] = attitude_control_decoupled_yaw(R, W, eI, 
% b3d, b3d_dot, b3d_ddot, b1c, wc3, wc3_dot, k, param)
%
% Decoupled-yaw attitude controller
% 
%   Caluclates control moments for a given set of desired attitude commands 
%   using a decoupled-yaw controller. This function uses the controller
%   defined in https://ieeexplore.ieee.org/document/8815189.
%   
%   Inputs:
%    R: (3x3 matrix) current attitude in SO(3)
%    W: (3x1 matrix) current angular velocity
%    eI: (3x1 matrix) attitude integral error
%    b3d: (3x1 matrix) desired direction of b3 axis
%    b3d_dot: (3x1 matrix) desired direction of b3 axis
%    b3d_ddot: (3x1 matrix) desired rotational rate of b3 axis
%    b1c: (3x1 matrix) desired direction of b1 axis
%    wc3: (3x1 matrix) desired yaw angular velocity
%    wc3_dot: (3x1 matrix) desired yaw angular acceleration
%    k: (struct) control gains
%    param: (struct) parameters such as m, g, J in a struct
%
%  Outputs:
%    M: (3x1 matrix) control moment required to reach desired conditions
%    eI_dot: (3x1 matrix) attitude integral change rate
%    eb: (3x1 matrix) roll/pitch angle error
%    ew: (3x1 matrix) roll/pitch angular velocity error
%    ey: (3x1 matrix) yaw angle error
%    ewy: (3x1 matrix) yaw angular velocity error

%% Unpack other parameters
J = param.J;
c2 = param.c2;
c3 = param.c3;

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

w = W(1) * b1 + W(2) * b2;                      # (23)
b3_dot = hat(w) * b3;                           # (22)

wd = hat(b3d) * b3d_dot;
wd_dot = hat(b3d) * b3d_ddot;

eb = hat(b3d) * b3;                             # (27)
ew = w + hat(b3)^2 * wd;                        # (28)
tau = - kb * eb ...                             # (31)
    - kw * ew ...
    - J(1,1) * dot(b3, wd) * b3_dot ...
    - J(1,1) * hat(b3)^2 * wd_dot ...
    - k.I * eI(1) * b1 - k.I * eI(2) * b2;

tau1 = dot(b1, tau);             
tau2 = dot(b2, tau);

M1 = tau1 + J(3,3) * W(3) * W(2);               # (24)              
M2 = tau2 - J(3,3) * W(3) * W(1);               # (24)

%% Yaw dynamics
ey = -dot(b2, b1c);                             # (49)
ewy = W(3) - wc3;                               # (50)

M3 = - k.y * ey ...                             # (52)
    - k.wy * ewy ...
    - k.yI * eI(3) ...
    + J(3,3) * wc3_dot;

eI_dot = [b1' * (c2 * eb + ew);
    b2' * (c2 * eb + ew);
    c3 * ey + ewy];

M=[M1, M2, M3]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u, u_dot, u_ddot] = deriv_unit_vector(q, q_dot, q_ddot)

nq = norm(q);
u = q / nq;
u_dot = q_dot / nq - q * dot(q, q_dot) / nq^3;

u_ddot = q_ddot / nq - q_dot / nq^3 * (2 * dot(q, q_dot))...
    - q / nq^3 * (dot(q_dot, q_dot) + dot(q, q_ddot))...
    + 3 * q / nq^5 * dot(q, q_dot)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hat_x = hat(x)

hat_x = [0 -x(3) x(2);
    x(3) 0 -x(1);
    -x(2) x(1) 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = vee(S)
s = [-S(2,3); S(1,3); -S(1,2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = sat(sigma, y)

for k=1:length(y)
    if y > sigma
        z = sigma;
    elseif y < -sigma
        z = -sigma;
    else
        z = y;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z=satdot(sigma, y, ydot)

for k = 1:length(y)
    if y > sigma
        z = 0;
    elseif y < -sigma
        z = 0;
    else
        z = ydot;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, v, R, W, ei, eI] = split_to_states(X)

x = X(1:3);
v = X(4:6);
W = X(7:9);
R = reshape(X(10:18), 3, 3);
ei = X(19:21);
eI = X(22:24);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_ship, x_uav, x_us] = true_data(t)

w = 1;
ship_x = 3 * sin(w*t);
ship_y = 2 * cos(w*t);
ship_z = 0;

uav_wrt_ship_x = 0.1 * cos(5*pi*t);
uav_wrt_ship_y = 0.1 * sin(5*pi*t);
uav_wrt_ship_z = 1.0 * sin(2*t);

uav_x = ship_x + uav_wrt_ship_x;
uav_y = ship_y + uav_wrt_ship_y;
uav_z = ship_z + uav_wrt_ship_z;

x_ship = [ship_x, ship_y, ship_z];
x_uav = [uav_x, uav_y, uav_z];
x_us = [uav_wrt_ship_x, uav_wrt_ship_y, uav_wrt_ship_z];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [error_, desired, R, f, M] = generate_output_arrays(N)

error_.x = zeros(3, N);
error_.v = zeros(3, N);
error_.R = zeros(3, N);
error_.W = zeros(3, N);
error_.y = zeros(1, N);
error_.Wy = zeros(1, N);

desired.x = zeros(3, N);
desired.b1 = zeros(3, N);
desired.R = zeros(3, 3, N);

R = zeros(3, 3, N);
f = zeros(1, N);
M = zeros(3, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_3x1(x, y, title_, xlabel_, ylabel_, linetype, linewidth, ...
    font_size)

if nargin < 8
    font_size = 10;
end

for i = 1:3
    subplot(3, 1, i);
    plot(x, y(i,:), linetype, 'LineWidth', linewidth);
    %ylabel(['$' ylabel_ '_' num2str(i) '}$'], 'interpreter', 'latex')    
    set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
    hold on;
end
xlabel(xlabel_, 'interpreter', 'latex');
title(title_);

subplot(3, 1, 2);
ylabel(['$' ylabel_ '$'], 'interpreter', 'latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        %ylabel(['$' ylabel_ '_' num2str(i) '}$'], 'interpreter', 'latex')    
        set(gca, 'FontName', 'Times New Roman', 'FontSize', font_size);
        ylim([-1 1]);
        hold on;
    end
end

title(title_);

subplot(3, 3, 8);
xlabel(xlabel_, 'interpreter', 'latex');

subplot(3, 3, 4);
ylabel(['$' ylabel_ '$'], 'interpreter', 'latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command_Lissajous(t)

A = 1;
B = 1;
C = 0.2;

d = pi / 2 * 0;

a = 2;
b = 3;
c = 2;
alt = -1;

% t = linspace(0, 2*pi, 2*pi*100+1);
% x = A * sin(a * t + d);
% y = B * sin(b * t);
% z = alt + C * cos(2 * t);
% plot3(x, y, z);

desired.x = [A * sin(a * t + d), B * sin(b * t), alt + C * cos(c * t)]';
desired.v = [A * a * cos(a * t + d), B * b * cos(b * t), C * c * -sin(c * t)]';
desired.x_2dot = [A * a^2 * -sin(a * t + d), B * b^2 * -sin(b * t), C * c^2 * -cos(c * t)]';
desired.x_3dot = [A * a^3 * -cos(a * t + d), B * b^3 * -cos(b * t), C * c^3 * sin(c * t)]';
desired.x_4dot = [A * a^4 * sin(a * t + d), B * b^4 * sin(b * t), C * c^4 * cos(c * t)]';

w = 2 * pi / 10;
desired.b1 = [cos(w * t), sin(w * t), 0]';
desired.b1_dot = w * [-sin(w * t), cos(w * t), 0]';
desired.b1_2dot = w^2 * [-cos(w * t), -sin(w * t), 0]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command_line(t)

height = 0;

desired.x = [0.5 * t, 0, -height]';
desired.v = [0.5 * 1, 0, 0]';
desired.x_2dot = [0, 0, 0]';
desired.x_3dot = [0, 0, 0]';
desired.x_4dot = [0, 0, 0]';

w = 2 * pi / 10;
desired.b1 = [cos(w * t), sin(w * t), 0]';
desired.b1_dot = w * [-sin(w * t), cos(w * t), 0]';
desired.b1_2dot = w^2 * [-cos(w * t), -sin(w * t), 0]';

