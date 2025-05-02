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
function test_controller
addpath('aux_functions');
addpath('test_functions');

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
[t, X] = ode45(@(t, XR) eom(t, XR, k, param), t, X0, ...
    odeset('RelTol', 1e-6, 'AbsTol', 1e-6));

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


A = 1;
B = 1;
C = 0.2;

d = pi / 2 * 0;

a = 2;
b = 3;
c = 2;
alt = -1;

t = linspace(0, 2*pi, 2*pi*100+1);
x = A * sin(a * t + d);
y = B * sin(b * t);
z = alt + C * cos(2 * t);
plot3(x, y, z);

% desired = command_line(t);
desired = command_lissajou(t);
% desired = command_point(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function desired = command_lissajou(t)

A = 1;
B = 1;
C = 0.2;

d = pi / 2 * 0;

a = 2;
b = 3;
c = 2;
alt = -1;

t = linspace(0, 2*pi, 2*pi*100+1);
x = A * sin(a * t + d);
y = B * sin(b * t);
z = alt + C * cos(2 * t);
plot3(x, y, z);

desired.x = [A * sin(a * t + d), B * sin(b * t), alt + C * cos(c * t)]';
desired.v = [A * a * cos(a * t + d), B * b * cos(b * t), C * c * -sin(c * t)]';
desired.x_2dot = [A * a^2 * -sin(a * t + d), B * b^2 * -sin(b * t), C * c^2 * -cos(c * t)]';
desired.x_3dot = [A * a^3 * -cos(a * t + d), B * b^3 * -cos(b * t), C * c^3 * sin(c * t)]';
desired.x_4dot = [A * a^4 * sin(a * t + d), B * b^4 * sin(b * t), C * c^4 * cos(c * t)]';

w = 2 * pi / 10;
desired.b1 = [cos(w * t), sin(w * t), 0]';
desired.b1_dot = w * [-sin(w * t), cos(w * t), 0]';
desired.b1_2dot = w^2 * [-cos(w * t), -sin(w * t), 0]';



