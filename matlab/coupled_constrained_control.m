% 31 August 2015
% driver for constrained attitude control

function coupled_constrained_control

% add path to utilities
constants.scenario = 'single'; % or 'single'
constants.avoid_switch = 'true';
constants.dist_switch = 'true';
constants.adaptive_switch = 'true';

% ACC/IJCAS Simulation for Fig 2 is
% constants.scenario = 'multiple'; % or 'single'
% constants.avoid_switch = 'true';
% constants.dist_switch = 'true';
% constants.adaptive_switch = 'false';


% constants for plotting/animations
constants.animation_type = 'none'; % or 'movie' or 'none'
constants.filename = 'multiple_avoid';

% define constants/properties of rigid body
constants.m_sc = 1;
m_sc = constants.m_sc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INERTIA TENSOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% constants.J = [1.059e-2 -5.156e-6 2.361e-5;...
%                 -5.156e-6 1.059e-2 -1.026e-5;
%                 2.361e-5 -1.026e-5 1.005e-2];
% Chris's Hexrotor inertia matrix
constants.J = [55710.50413e-7 ,  617.6577e-7   , -250.2846e-7 ;...
               617.6577e-7    ,  55757.4605e-7 , 100.6760e-7 ;...
               -250.2846e-7  ,  100.6760e-7   , 105053.7595e-7];

% % % from Farhad ASME paper
% constants.J = [ 5.5711 0.0618 -0.0251; ...
%                 0.06177 5.5757 0.0101;...
%                 -0.02502 0.01007 1.05053] * 1e-2;

% constants.J = diag([694 572 360]);

J = constants.J;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONTROLLER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% controller parameters
constants.G = diag([0.9 1 1.1]);

% con = -1+2.*rand(3,constants.num_con); % inertial frame vectors (3XN)
% from [1] U. Lee and M. Mesbahi. Spacecraft Reorientation in Presence of Attitude Constraints via Logarithmic Barrier Potentials. In 2011 AMERICAN CONTROL CONFERENCE, Proceedings of the American Control Conference, pages 450?455, 345 E 47TH ST, NEW YORK, NY 10017 USA, 2011. Boeing; Bosch; Corning; Eaton; GE Global Res; Honeywell; Lockheed Martin; MathWorks; Natl Instruments; NT-MDT; United Technol, IEEE. American Control Conference (ACC), San Fransisco, CA, JUN 29-JUL 01, 2011.
% con = [0.174    0   -0.853 -0.122;...
%     -0.934   0.7071    0.436 -0.140;...
%     -0.034   0.7071   -0.286 -0.983];
% column vectors to define constraints
% zeta = 0.7;
% wn = 0.2;
% constants.kp = wn^2;
% constants.zeta = 2*zeta*wn;
% constants.kp = 0.0424; % wn^2
% constants.kp = 0.4;
% constants.kv = 0.296; % 2*zeta*wn
constants.kp = 0.4;
constants.kv = 0.296; % 2*zeta*wn


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constants.sen = [1;0;0]; % body fixed frame
% define a number of constraints to avoids

switch constants.scenario
    case 'multiple'
        con = [0.174    0.4   -0.853 -0.122;...
            -0.934   0.7071    0.436 -0.140;...
            -0.034   0.7071   -0.286 -0.983];
        constants.con_angle = [40;40;40;20]*pi/180;
    case 'single'
        con = [1/sqrt(2);1/sqrt(2);0];
        constants.con_angle = 12*pi/180;
end
constants.con = con./repmat(sqrt(sum(con.^2,1)),3,1); % normalize

constants.alpha = 15; % use the same alpha for each one
constants.num_con = size(constants.con,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADAPTIVE CONTROL FOR DISTURBANCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disturbance terms

constants.W = eye(3,3);
constants.delta = @(t) 0.2 + 0.02*[sin(9*t);cos(9*t);1/2*(sin(9*t)+cos(9*t))];
constants.kd = 0.5; % adaptive controller gain term (rate of convergence)
constants.c = 1; % input the bound on C here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DESIRED/INITIAL CONDITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the initial state of rigid body

% R0 = ROT1(0*pi/180)*ROT2(45*pi/180)*ROT3(180*pi/180);
constants.q0 = [-0.188 -0.735 -0.450 -0.471];
% constants.qd = [-0.59 0.67 0.21 -0.38]; % from lee/meshbahi paper
constants.qd = [0 0 0 1];

% constants.R0 = quat2dcm(constants.q0)';
% constants.Rd = quat2dcm(constants.qd)';

switch constants.scenario
    case 'multiple'
        constants.R0 = ROT1(0*pi/180)*ROT3(225*pi/180); % avoid multiple constraints
        constants.Rd = eye(3,3);
    case 'single'
        constants.R0 = ROT1(0*pi/180)*ROT3(0*pi/180); % avoid single constraint
        constants.Rd = ROT3(90*pi/180);
end

R0 = constants.R0;
w0 = zeros(3,1);
delta_est0 = zeros(3,1);
initial_state = [constants.R0(:);w0; delta_est0];

% simulation timespan
tspan = linspace(0,20,1000);

% propogate a chief and deputy spacecraft (continuous time system)
[t, state] = ode45(@(t,state)dynamics(t,state,constants),tspan, initial_state);
% calculate the relative position and attitude of deputy wrt chief

% extract out the states

% loop to save the Body to Inertial rotation matrix into a big array
% calculate the desired trajectory
R_b2i = zeros(3,3,length(tspan));
u_f = zeros(3,length(tspan));
u_m = zeros(3,length(tspan));
R_des = zeros(3,3,length(tspan));
ang_vel_des = zeros(3,length(tspan));
ang_vel_dot_des = zeros(3,length(tspan));
Psi = zeros(length(tspan),1);
err_att = zeros(3,length(tspan));
err_vel = zeros(3,length(tspan));

ang_vel = state(:,10:12);
delta_est = state(:,13:15);
for ii = 1:length(tspan)
   R_b2i(:,:,ii) = reshape(state(ii,1:9),3,3);

   [u_f(:,ii), u_m(:,ii), R_des(:,:,ii), ang_vel_des(:,ii), ang_vel_dot_des(:,ii), Psi(ii), err_att(:,ii), err_vel(:,ii)] ...
    = controller(t(ii),state(ii,:)', constants);
end


num_figs = 8;
fig_handle = zeros(num_figs,1);

fontsize = 18;
fontname = 'Times';

fontsize = 18;
fontname = 'Times';
figx = 680;
figy = 224;
figw = 800;
figh = 600;

fig_size = [figx,figy,figw,figh];

for ii = 1:num_figs
    fig_handle(ii) = figure('Position',fig_size);
end

% % plot the position
% figure
%
% subplot(3,1,1)
% title('Position of SC in inertial frame','interpreter','latex')
% xlabel('$t (sec)$','interpreter','latex')
% ylabel('$x (km)$','interpreter','latex')
% grid on;hold on
% plot(t,pos(:,1));
%
% subplot(3,1,2)
% xlabel('$t (sec)$','interpreter','latex')
% ylabel('$y (km)$','interpreter','latex')
% grid on;hold on
% plot(t,pos(:,2));
%
% subplot(3,1,3)
% xlabel('$t (sec)$','interpreter','latex')
% ylabel('$z (km)$','interpreter','latex')
% grid on;hold on
% plot(t,pos(:,3));

% plot the attitude error vector
set(0, 'CurrentFigure', fig_handle(1)) % attitude error vector
subplot(3,1,1)
% title('Attitude error vector','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{R_1}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_att(1,:));


subplot(3,1,2)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{R_2}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_att(2,:));
##

subplot(3,1,3)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{R_3}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_att(3,:));

##

% plot the attitude error \Psi
set(0, 'CurrentFigure', fig_handle(2)) % \Psi
% title('$\Psi$ error ','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$\Psi$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,Psi);



% plot the angular velocity error
set(0, 'CurrentFigure', fig_handle(3))
subplot(3,1,1)
% title('Angular velocity error vector','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{\Omega_1}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_vel(1,:));
##

subplot(3,1,2)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{\Omega_2}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_vel(2,:));
##

subplot(3,1,3)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$e_{\Omega_3}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,err_vel(3,:));

##

% plot the control input
set(0, 'CurrentFigure', fig_handle(4))
subplot(3,1,1)
% title('Control Input','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$u_{1}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,u_m(1,:));
##

subplot(3,1,2)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$u_{2}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,u_m(2,:));
##

subplot(3,1,3)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$u_{3}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,u_m(3,:));

##

% plot the desired adn actual angular velocities
set(0, 'CurrentFigure', fig_handle(5))

subplot(3,1,1)
title('Angular Velocity','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$\Omega_{1}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,ang_vel(:,1),'b');
plot(t,ang_vel_des(1,:),'r');
l=legend('Actual','Desired');
set(l,'interpreter','latex')
##

subplot(3,1,2)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$\Omega_{2}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,ang_vel(:,2),'b');
plot(t,ang_vel_des(2,:),'r');
##

subplot(3,1,3)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$\Omega_{3}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
grid on;hold on
plot(t,ang_vel(:,3),'b');
plot(t,ang_vel_des(3,:),'r');

##

% plot the disturbance estimate
set(0, 'CurrentFigure', fig_handle(6))
hold all
grid on
% title('$\bar{\Delta}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('$\bar{\Delta}$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
plot(t,delta_est);

##

% plot attitude on unit sphere
% create a sphere
set(0, 'CurrentFigure', fig_handle(7))
set(fig_handle(7),'Color','w');

hold all

[sph.x, sph.y, sph.z]=sphere(100);
% radius of cone at unit length
% loop over the number of constraints and draw them all
h_cyl = zeros(constants.num_con,1);
for ii = 1:constants.num_con

    H = cos(constants.con_angle(ii)); %// height
    R = sin(constants.con_angle(ii)); %// chord length
    N = 100; %// number of points to define the circumference
    [cyl.x, cyl.y, cyl.z] = cylinder([0 R], N);
    cyl.z = cyl.z*H;
    % calculate the rotation matrix to go from pointing in e3 direction to
    % along the con unit vector
    if sum(constants.con(:,ii) == [0;0;1]) == 3
        dcm = eye(3,3);
    elseif sum(constants.con(:,ii) == [0;0;-1]) == 3
        dcm = ROT1(pi);
    else
    k_hat = cross([0;0;1],constants.con(:,ii));
    angle = acos(dot([0;0;1],constants.con(:,ii)));
    dcm = eye(3,3) + hat_map(k_hat) + hat_map(k_hat)*hat_map(k_hat)*(1-cos(angle))/sin(angle)^2;
    end

    cyl_open = dcm*[cyl.x(2,:);cyl.y(2,:);cyl.z(2,:)];
    cyl.x(2,:) = cyl_open(1,:);
    cyl.y(2,:) = cyl_open(2,:);
    cyl.z(2,:) = cyl_open(3,:);


    h_cyl(ii) = surf(cyl.x,cyl.y,cyl.z);
end
h_sph=surf(sph.x,sph.y,sph.z);

set(h_sph,'LineStyle','none','FaceColor',0.8*[1 1 1],...
    'FaceLighting','gouraud','AmbientStrength',0.5,...
    'Facealpha',0.3,'Facecolor',[0.8 0.8 0.8]);
set(h_cyl,'Linestyle','none',...
    'FaceLighting','gouraud','AmbientStrength',0.5,...
    'Facealpha',0.5,'Facecolor','red');

light('Position',[0 0 100],'Style','infinite');
material dull;
axis equal;
axis off
xlabel('x')
ylabel('y')
zlabel('z')
% convert the body fixed vector to the inertial frame
sen_inertial = zeros(length(tspan),3);

for ii = 1:length(tspan)
   sen_inertial(ii,:) = (R_b2i(:,:,ii)*constants.sen)';
end
sen_inertial_start = constants.R0*constants.sen;
sen_inertial_end = constants.Rd*constants.sen;
% plot path of body vector in inertial frame
plot3(sen_inertial_start(1),sen_inertial_start(2),sen_inertial_start(3),'go','markersize',10,'linewidth',2)
plot3(sen_inertial_end(1),sen_inertial_end(2),sen_inertial_end(3),'gx','markersize',10,'linewidth',2)
plot3(sen_inertial(:,1),sen_inertial(:,2),sen_inertial(:,3),'b','linewidth',3)

% plot inertial frame
line([0 1],[0 0],[0 0],'color','k','linewidth',3);
line([0 0],[0 1],[0 0],'color','k','linewidth',3);
line([0 0],[0 0],[0 1],'color','k','linewidth',3);
view(3)

% plot the angle to each constraint

% calculate the angle to each constraint
ang_con = zeros(length(tspan),constants.num_con);

for ii = 1:length(tspan)
    for jj = 1:constants.num_con
       ang_con(ii,jj) = 180/pi*acos(dot(sen_inertial(ii,:),constants.con(:,jj)));
    end
end
set(0, 'CurrentFigure', fig_handle(8))
grid on
hold all
for ii = 1:constants.num_con
    plot(t,ang_con(:,ii))
end

xlabel('$t (sec)$','interpreter','latex','FontName',fontname,'FontSize',fontsize)
ylabel('arc$$\cos \,(r^T R^T v_i)$$','interpreter','latex','FontName',fontname,'FontSize',fontsize)

##

% draw_cad

#####################################################################################################################
function [Rd Wd Wd_dot]=tracking_command(t)
% Tracking Command Generation

phi=20*pi/180*sin(pi*t);
phidot=20*pi/180*pi*cos(pi*t);
phiddot=20*pi/180*pi*pi*-sin(pi*t);

theta=20*pi/180*cos(pi*t);
thetadot=20*pi/180*pi*-sin(pi*t);
thetaddot=20*pi/180*pi*pi*-cos(pi*t);

psi=0;
psidot=0;
psiddot=0;

Rd=[cos(theta)*cos(psi) sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi) cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);
    cos(theta)*sin(psi) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi);
    -sin(theta) sin(phi)*cos(theta) cos(phi)*cos(theta)];
Wd=[1 0 -sin(theta);
    0 cos(phi) sin(phi)*cos(theta);
    0 -sin(phi) cos(phi)*cos(theta)]*[phidot; thetadot; psidot];
Wd_dot=[ -psidot*thetadot*cos(theta);
    - phidot*(thetadot*sin(phi) - psidot*cos(phi)*cos(theta)) - psidot*thetadot*sin(phi)*sin(theta);
    - phidot*(thetadot*cos(phi) + psidot*cos(theta)*sin(phi)) - psidot*thetadot*cos(phi)*sin(theta)]+...
    [1 0 -sin(theta);
    0 cos(phi) sin(phi)*cos(theta);
    0 -sin(phi) cos(phi)*cos(theta)]*[phiddot; thetaddot; psiddot];

#####################################################################################################################
% ODE function for constrained attitude stabilization

function [state_dot] = dynamics(t, state, constants)

% constants
m_sc = constants.m_sc;
J = constants.J;
kd = constants.kd;
W = constants.W;

% redefine the state vector
R = reshape(state(1:9),3,3); % rotation matrix from body to inertial frame
ang_vel = state(10:12);
delta_est = state(13:15); % adaptive control term to estimate fixed disturbance

% calculate external force and moment
[~, m] = ext_force_moment(t,state,constants);

[~, u_m, ~, ~, ~, ~, err_att, err_vel] ...
    = controller(t,state,constants);

% differential equations of motion

R_dot = R*hat_map(ang_vel);
ang_vel_dot =J \ ( m + u_m - cross(ang_vel,J*ang_vel));
% theta_est_dot =  gam/2 * W' *(err_vel+ constants.kp/constants.kv * err_att);
% theta_est_dot =  gam/2 * W' *(err_vel+ (constants.c + constants.kp/constants.kv)* err_att);
theta_est_dot = kd * W' *(err_vel + constants.c*err_att);
% output the state derivative
state_dot = [R_dot(:);ang_vel_dot; theta_est_dot];



function [f, m] = ext_force_moment(t,state, constants)

% redefine the state vector
R = reshape(state(1:9),3,3); % rotation matrix from body to inertial frame
ang_vel = state(10:12);

% constants
m_sc = constants.m_sc;
J = constants.J;
W = constants.W;
delta = constants.delta;

% calculate external moment and force
f = zeros(3,1);

% add a constant disturbance
% m = 3*mu/norm(pos)^3 * cross(R_body2lvlh'*a1_hat,J*R_body2lvlh'*a1_hat);
switch constants.dist_switch
    case 'true'
        m = zeros(3,1) + W*delta(t);
    case 'false'
        m = zeros(3,1);
end
#####################################################################################################################
% vee map function to take a skew symmetric matrix and map it to a 3 vector

function [vec] = vee_map(mat)

x1 = mat(3,2)-mat(2,3);
x2 = mat(1,3) - mat(3,1);
x3 = mat(2,1)-mat(1,2);

vec = 1/2*[x1;x2;x3];
#####################################################################################################################
% 8 June 15
% skew symmetric operator

function mat = hat_map(vec)
% maps a 3-vec to a skew symmetric matrix
mat = zeros(3,3);

mat(1,2) = -vec(3);
mat(1,3) = vec(2);
mat(2,1) = vec(3);
mat(2,3) = -vec(1);
mat(3,1) = -vec(2);
mat(3,2) = vec(1);
#####################################################################################################################
% 11 June 15
% controller

function [u_f, u_m, R_des, ang_vel_des, ang_vel_dot_des, Psi, err_att, err_vel] = controller(t,state, constants)

% redefine the state vector
R = reshape(state(1:9),3,3); % rotation matrix from body to inertial frame
ang_vel = state(10:12);
delta_est = state(13:15);

% extract out constants
J = constants.J;
G = constants.G;
kp = constants.kp;
kv = constants.kv;
sen = constants.sen;
alpha = constants.alpha;
con_angle = constants.con_angle;
con = constants.con;
W = constants.W;

% desired attitude
[R_des, ang_vel_des, ang_vel_dot_des] = des_attitude(t,constants);

% attitude error function
% Psi = 1/2*trace(G*(eye(3,3)-R_des'*R));
% rotate body vector to inertial frame
psi_attract = 1/2*trace(G*(eye(3,3)-R_des'*R));
dA = 1/2*vee_map(G*R_des'*R - R'*R_des*G);

switch constants.avoid_switch
    case 'true' % add the avoidance term
        sen_inertial = R * sen;

        % loop over the constraints and form a bunch of repelling function
        psi_avoid = zeros(constants.num_con,1);
        dB = zeros(3,constants.num_con);
        for ii = 1:constants.num_con

            % calculate error function
            psi_avoid(ii) = -1/alpha*log((cos(con_angle(ii))-dot(sen_inertial,con(:,ii)))/(1+cos(con_angle(ii))));

            dB(:,ii) = 1/alpha/(dot(sen_inertial,con(:,ii))-cos(con_angle(ii)))*hat_map(R'*con(:,ii))*sen;
        end

        Psi = psi_attract*(sum(psi_avoid)+1);

        err_att = dA*(sum(psi_avoid)+1) + sum(dB.*psi_attract,2);

    case 'false'
        err_att = dA;
        Psi = psi_attract;
end

err_vel = ang_vel - R'*R_des*ang_vel_des;

alpha_d = -hat_map(ang_vel)*R'*R_des*ang_vel_des + R'*R_des*ang_vel_dot_des;

% compute the control input
u_f = zeros(3,1);
% u_m = -kp*err_att - kv*err_vel + cross(ang_vel,J*ang_vel) + J*alpha_d - W * theta_est;
switch constants.adaptive_switch
    case 'true'
        u_m = -kp*err_att - kv*err_vel + cross(ang_vel,J*ang_vel) -W * delta_est;
    case 'false'
        u_m = -kp*err_att - kv*err_vel + cross(ang_vel,J*ang_vel);
end



function [R_des, ang_vel_des, ang_vel_dot_des] = des_attitude(t,constants)

% use 3-2-1 euler angle sequence for the desired body to inertial attitude
% trajectory
a = 2*pi/(20/10);
% a = pi;
b = pi/9;

phi = b*sin(a*t); % third rotation
theta = b*cos(a*t); % second rotation
psi = 0; % first rotation

phi_d = b*a*cos(a*t);
theta_d = -b*a*sin(a*t);
psi_d = 0;

phi_dd = -b*a^2*sin(a*t);
theta_dd = -b*a^2*cos(a*t);
psi_dd = 0;

% euler 3-2-1 sequence
% R_des = ROT1(phi)*ROT2(theta)*ROT3(psi);
R_des = constants.Rd;
% R_des = R_des'; % Dr. Lee is using the transpose of the attitude matrix

% convert the euler angle sequence to the desired angular velocity vector
ang_vel_des = zeros(3,1);

% ang_vel_des(1) = -psi_d*sin(theta) + phi_d;
% ang_vel_des(2) = psi_d*cos(theta)*sin(phi) + theta_d*cos(phi);
% ang_vel_des(3) = psi_d*cos(theta)*cos(phi) - theta_d*sin(phi);

ang_vel_dot_des = zeros(3,1);
%
% ang_vel_dot_des(1) = -psi_dd*sin(theta) - theta_d*psi_d*cos(theta) + phi_dd;
% ang_vel_dot_des(2) = psi_dd*cos(theta)*sin(phi) - theta_d*psi_d*sin(theta)*sin(phi) ...
%                     + psi_d*phi_d*cos(theta)*cos(phi) + theta_dd*cos(phi) ...
%                     -theta_d*phi_d*sin(phi);
% ang_vel_dot_des(3) = psi_dd*cos(theta)*cos(phi) - theta_d*psi_d*sin(theta)*cos(phi) ...
%                     - phi_d*psi_d*cos(theta)*sin(phi) - theta_dd*sin(phi) ...
%                     - theta_d*phi_d*cos(phi);

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Body Euler 1-3-1 Angles to direction cosine matrix
%
%   Purpose:
%       - Converts the Body Euler 1-3-1 Angles representing a rotation into the equivalent
%        row vector format direction cosine matrix
%
%   dcm = body1312dcm(theta)
%
%   Inputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Outputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 4 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dcm = body1312dcm(theta)

st1 = sin(theta(1));
st2 = sin(theta(2));
st3 = sin(theta(3));
ct1 = cos(theta(1));
ct2 = cos(theta(2));
ct3 = cos(theta(3));

dcm = zeros(3,3);

dcm(1,1) = ct2;
dcm(1,2) = -st2*ct3;
dcm(1,3) = st2*st3;
dcm(2,1) = ct1*st2;
dcm(2,2) = ct1*ct2*ct3-st3*st1;
dcm(2,3) = -ct1*ct2*st3-ct3*st1;
dcm(3,1) = st1*st2;
dcm(3,2) = st1*ct2*ct3+st3*ct1;
dcm(3,3) = -st1*ct2*st3+ct3*ct1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Body 1-3-1 Differential Equation
%
%   Purpose:
%       - Finds the rate of change of the Euler Body 1-3-1 angles
%
%   theta_d = body131dot(theta,w)
%
%   Inputs:
%       - theta - Nx3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%       - w - Nx3 angular velocity vector in teh body frame components in
%       rad/sec
%
%   Outputs:
%       - theta_d - Nx3 element vector with the 3 rotation angle derivatives. Same order
%       as m-file filename. theta = [first second third] rad/sec
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%       - Shankar Kulumani 17 Feb 2013
%           - vecotrized
%
%   References
%       - AAE590 Omega Angle Rates pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta_d = body131dot(theta,w)

c2 = cos(theta(:,2));
c3 = cos(theta(:,3));
s2 = sin(theta(:,2));
s3 = sin(theta(:,3));

w1 = w(:,1);
w2 = w(:,2);
w3 = w(:,3);

theta_d(:,1) = (-w2.*c3 + w3.*s3)./s2;
theta_d(:,2) = w2.*s3+w3.*c3;
theta_d(:,3) = w1 +(w2.*c3-w3.*s3).*c2./s2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Body 3-1-3 Differential Equation
%
%   Purpose:
%       - Finds the rate of change of the Euler Body 3-1-3 angles
%
%   theta_d = body131dot(theta,w)
%
%   Inputs:
%       - theta - Nx3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%       - w - Nx3 angular velocity vector in teh body frame components in
%       rad/sec
%
%   Outputs:
%       - theta_d - Nx3 element vector with the 3 rotation angle derivatives. Same order
%       as m-file filename. theta = [first second third] rad/sec
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 12 Feb 2013
%           - vectorized code
%
%   References
%       - AAE590 Omega Angle Rates pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta_d = body313dot(theta,w)

c2 = cos(theta(:,2));
c3 = cos(theta(:,3));
s2 = sin(theta(:,2));
s3 = sin(theta(:,3));

w1 = w(:,1);
w2 = w(:,2);
w3 = w(:,3);

theta_d(:,1) = (w1.*s3 + w2.*c3)./s2;
theta_d(:,2) = w1.*c3-w2.*s3;
theta_d(:,3) = w3 -(w1.*s3+w2.*c3).*c2./s2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Euler Body 1-2-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Euler Body 1-2-1 Angles
%       about body fixed axes
%
%   theta = dcm2body121(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third]
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%       - Shankar Kulumani 2 Mar 2013
%           - vectorized teh code
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body121(dcm)

theta(:,1) = atan2(dcm(2,1,:),-dcm(3,1,:));
theta(:,2) = acos(dcm(1,1,:));
theta(:,3)= atan2(dcm(1,2,:),dcm(1,3,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Euler Body 1-2-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 1-2-3 Angles
%       Rotations about body fixed axes
%
%   theta = dcm2body123(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third]
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body123(dcm)

theta(1) = atan2(-dcm(2,3),dcm(3,3));
theta(2) = asin(dcm(1,3));
theta(3)= atan2(-dcm(1,2),dcm(1,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 1-3-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 1-3-1Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body131(dcm)
%
%   Inputs:
%       - dcm - 3x3xN rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - Nx3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%       - Shankar Kulumani 17 Feb 2013
%           - vectorized the code
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body131(dcm)

theta(:,1) = atan2(dcm(3,1,:),dcm(2,1,:));
theta(:,2) = acos(dcm(1,1,:));
theta(:,3)= atan2(dcm(1,3,:),-dcm(1,2,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Euler Body 1-3-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Euler Body 1-3-2 Angles
%       about body fixed axes
%
%   theta = dcm2body132(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third]
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 30 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2body132(dcm)

theta(1) = atan2(dcm(3,2),dcm(2,2));
theta(2) = asin(-dcm(1,2));
theta(3)= atan2(dcm(1,3),dcm(1,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Euler Body 2-1-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Euler Body 2-1-2 Angles
%       about body fixed axes
%
%   theta = dcm2body212(dcm)
%
%   Inputs:
%       - dcm - 3x3xN rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - Nx3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%       - Shankar Kulumani 20 Mar 2013
%           - vectorized the code
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta = dcm2body212(dcm)

theta(:,1) = atan2(dcm(1,2,:),dcm(3,2,:));
theta(:,2) = acos(dcm(2,2,:));
theta(:,3)= atan2(dcm(2,1,:),-dcm(2,3,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 2-1-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 2-1-3 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body323(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third]
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body213(dcm)

theta(1) = atan2(dcm(1,3),dcm(3,3));
theta(2) = asin(-dcm(2,3));
theta(3)= atan2(dcm(2,1),dcm(2,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 2-3-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 2-3-1 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body231(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body231(dcm)

theta(1) = atan2(-dcm(3,1),dcm(1,1));
theta(2) = asin(dcm(2,1));
theta(3)= atan2(-dcm(2,3),dcm(2,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 2-3-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 2-3-2 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body232(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body232(dcm)

theta(1) = atan2(dcm(3,2),-dcm(1,2));
theta(2) = acos(dcm(2,2));
theta(3)= atan2(dcm(2,3),dcm(2,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 3-1-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 3-1-2 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body312(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body312(dcm)

theta(1) = atan2(-dcm(1,2),dcm(2,2));
theta(2) = asin(dcm(3,2));
theta(3)= atan2(-dcm(3,1),dcm(3,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 3-1-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 3-1-3 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body313(dcm)
%
%   Inputs:
%       - dcm - 3x3xN rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - Nx3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%       - Shankar Kulumani 12 Feb 2013
%           - vectorize the code for speed
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body313(dcm)

theta(:,1) = atan2(dcm(1,3,:),-dcm(2,3,:));
theta(:,2) = acos(dcm(3,3,:));
theta(:,3)= atan2(dcm(3,1,:),dcm(3,2,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 3-2-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 3-2-1 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body321(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body321(dcm)

theta(1) = atan2(dcm(2,1),dcm(1,1));
theta(2) = asin(-dcm(3,1));
theta(3)= atan2(dcm(3,2),dcm(3,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Body Euler 3-2-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Body Euler 3-2-3 Angles.
%       Rotation about body fixed axes
%
%   theta = dcm2body323(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2body323(dcm)

theta(1) = atan2(dcm(2,3),dcm(1,3));
theta(2) = acos(dcm(3,3));
theta(3)= atan2(dcm(3,2),-dcm(3,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to quaternion
%
%   Purpose:
%       - Converts a direction cosine matrix to equivalent quaternion
%
%   quat = dcm2quat(C)
%
%   Inputs:
%       - C - 3x3 direction cosine matrix in row vector format b =
%       a*dcm_a2b
%
%   Outputs:
%       - quat - quaternion with [e n] where e - 1x3 vector and n is the
%       scalar magnitude assumes row vector format
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 19 Jan 2013
%       - Shankar Kulumani 24 Jan 2013
%           - modified for row vector format
%
%   References
%       - P. Hughes. Spacecraft attitude dynamics. Dover Publications, 2004.
%       - AAE590 Lesson 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function quat = dcm2quat(C)

tr = C(1,1)+C(2,2)+C(3,3)+1;

e = zeros(1,3);

n = sqrt(tr)*1/2;

e(1) = (C(3,2)- C(2,3))/4/n;
e(2) = (C(1,3)-C(3,1))/4/n;
e(3) = (C(2,1)-C(1,2))/4/n;

quat = [e n];

%
% [v,i] = max(b2);
% switch i
% 	case 1
% 		b(1) = sqrt(b2(1));
% 		b(2) = (C(2,3)-C(3,2))/4/b(1);
% 		b(3) = (C(3,1)-C(1,3))/4/b(1);
% 		b(4) = (C(1,2)-C(2,1))/4/b(1);
% 	case 2
% 		b(2) = sqrt(b2(2));
% 		b(1) = (C(2,3)-C(3,2))/4/b(2);
% 		if (b(1)<0)
% 			b(2) = -b(2);
% 			b(1) = -b(1);
% 		end
% 		b(3) = (C(1,2)+C(2,1))/4/b(2);
% 		b(4) = (C(3,1)+C(1,3))/4/b(2);
% 	case 3
% 		b(3) = sqrt(b2(3));
% 		b(1) = (C(3,1)-C(1,3))/4/b(3);
% 		if (b(1)<0)
% 			b(3) = -b(3);
% 			b(1) = -b(1);
% 		end
% 		b(2) = (C(1,2)+C(2,1))/4/b(3);
% 		b(4) = (C(2,3)+C(3,2))/4/b(3);
% 	case 4
% 		b(4) = sqrt(b2(4));
% 		b(1) = (C(1,2)-C(2,1))/4/b(4);
% 		if (b(1)<0)
% 			b(4) = -b(4);
% 			b(1) = -b(1);
% 		end
% 		b(2) = (C(3,1)+C(1,3))/4/b(4);
% 		b(3) = (C(2,3)+C(3,2))/4/b(4);
% end
% b = b';%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 1-2-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 1-2-1 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space121(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space121(dcm)

theta(1) = atan2(dcm(1,2),dcm(1,3));
theta(2) = acos(dcm(1,1));
theta(3) = atan2(dcm(2,1),-dcm(3,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 1-2-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 1-2-3 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space123(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2space123(dcm)

theta(1) = atan2(dcm(3,2),dcm(3,3));
theta(2) = asin(-dcm(3,1));
theta(3) = atan2(dcm(2,1),dcm(1,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 1-3-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 1-3-1Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space131(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2space131(dcm)

theta(1) = atan2(dcm(1,3),-dcm(1,2));
theta(2) = acos(dcm(1,1));
theta(3)= atan2(dcm(3,1),dcm(2,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 1-3-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 1-3-2 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space132(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space132(dcm)

theta(1) = atan2(-dcm(2,3),dcm(2,2));
theta(2) = asin(dcm(2,1));
theta(3) = atan2(-dcm(3,1),dcm(1,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 2-1-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 2-1-2 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space121(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space212(dcm)

theta(1) = atan2(dcm(2,1),-dcm(2,3));
theta(2) = acos(dcm(2,2));
theta(3) = atan2(dcm(1,2),dcm(3,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 2-1-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 2-1-3 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space213(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2space213(dcm)

theta(1) = atan2(-dcm(3,1),dcm(3,3));
theta(2) = asin(dcm(3,2));
theta(3)= atan2(-dcm(1,2),dcm(2,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 2-3-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 2-3-1 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space231(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = dcm2space231(dcm)

theta(1) = atan2(dcm(1,3),dcm(1,1));
theta(2) = asin(-dcm(1,2));
theta(3) = atan2(dcm(3,2),dcm(2,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 2-3-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 2-3-2 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space121(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space232(dcm)

theta(1) = atan2(dcm(2,3),dcm(2,1));
theta(2) = acos(dcm(2,2));
theta(3) = atan2(dcm(3,2),-dcm(1,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 3-1-2 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 3-1-2 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space312(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space312(dcm)

theta(1) = atan2(dcm(2,1),dcm(2,2));
theta(2) = asin(-dcm(2,3));
theta(3) = atan2(dcm(1,3),dcm(3,3));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 3-1-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 3-1-3 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space313(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space313(dcm)

theta(1) = atan2(dcm(3,1),dcm(3,2));
theta(2) = acos(dcm(3,3));
theta(3) = atan2(dcm(1,3),-dcm(2,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 3-2-1 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 3-2-1 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space321(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space321(dcm)

theta(1) = atan2(-dcm(1,2),dcm(1,1));
theta(2) = asin(dcm(1,3));
theta(3) = atan2(-dcm(2,3),dcm(3,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to Space Euler 3-2-3 Angles
%
%   Purpose:
%       - Converts the row vector format direction cosine matrix
%       representing a rotation into the equivalent Space Euler 3-2-3 Angles.
%       Rotation about space fixed axes
%
%   theta = dcm2space121(dcm)
%
%   Inputs:
%       - dcm - 3x3 rotation matrix assuming row vector format b = a*dcm
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta = dcm2space323(dcm)

theta(1) = atan2(dcm(3,2),-dcm(3,1));
theta(2) = acos(dcm(3,3));
theta(3) = atan2(dcm(2,3),dcm(1,3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Direction cosine matrix to simple rotation
%
%   Purpose:
%       - Convert direction cosine matrix to euler axis and angle
%
%   [lambda theta] = dcm2srt(C)
%
%   Inputs:
%       - C - 3x3 direction cosine matrix in row vector format b = a
%       *dcm_a2b
%
%   Outputs:
%       - lambda - euler axis of rotation 1x3 unit vector
%       - theta - euler angle of rotation in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 19 Jan 2013
%
%   References
%       - P. Hughes. Spacecraft attitude dynamics. Dover Publications, 2004.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lambda theta] = dcm2srt(C)

C = C';

sigma = trace(C);

theta = acos(1/2*(sigma-1));

sin_theta = sin(theta);
lambda = zeros(1,3);

switch sigma
    case 3
        disp('Error: undefined')
    case -1
        disp('error')
    otherwise
        lambda(1) = 1/2*(C(2,3)-C(3,2))/sin_theta;
        lambda(2) = 1/2*(C(3,1)-C(1,3))/sin_theta;
        lambda(3) = 1/2*(C(1,2)-C(2,1))/sin_theta;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Poisson Equation
%
%   Purpose:
%       - Calculates the kinematic differential equation for the direction
%       cosine matrix ( Poisson Equation)
%
%   dcm_dot = dcmdot(dcm,omega)
%
%   Inputs:
%       - dcm - 3x3 direction cosine matrix relating the rotation between
%       two frames. assumes row vector format b = a*dcm_a2b
%       - omega - 1x3 angular velocity between the two frames.
%
%   Outputs:
%       - dcm_dot - 3x3 derivative of direction cosine matrix
%
%   Dependencies:
%       - skew_matrix.m - calculates skew symmetric matrix
%
%   Author:
%       - Shankar Kulumani 7 April 2013
%           - created for AAE590 PS10
%
%   References
%       - AAE590
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dcm_dot = dcmdot(dcm,omega)

dcm_dot = dcm*skew_matrix(omega);% This script will orthogonalize a direction cosine matrix
function [orthoDCM] = orthodcm(DCM)

%#eml

delT = DCM*DCM'-eye(3);
orthoDCM = DCM*(eye(3)-(1/2)*delT+(1/2)*(3/4)*delT.^2-(1/2)*(3/4)*(5/6)*delT.^3+(1/2)*(3/4)*(5/6)*(7/8)*delT.^4-(1/2)*(3/4)*(5/6)*(7/8)*(9/10)*delT.^5+...
    (1/2)*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*delT.^6-(1/2)*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*delT.^7+(1/2)*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*delT.^8-...
(1/2)*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*(17/18)*delT.^9+(1/2)*(3/4)*(5/6)*(7/8)*(9/10)*(11/12)*(13/14)*(15/16)*(17/18)*(19/20)*delT^10);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion to Euler Body 1-3-1
%
%   Purpose:
%       - Convert a quaternion to the equivalaent Euler Body 1-3-1 sequence
%       about the body fixed axes.
%
%   function theta = quat2body131(quat)
%
%   Inputs:
%       - quat - 1x4 quaternion describing the orientation of the body
%       frame wrt inertial frame. The 4th element is the scalar value.
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] radians
%
%   Dependencies:
%       - quat2dcm.m - convert quaternion to dcm
%       - dcm2body131.m - convert dcm to euler angles
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = quat2body131(quat)

dcm = quat2dcm(quat);

theta = dcm2body131(dcm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion to direction cosine matrix
%
%   Purpose:
%       - Converts a 4 element quaternion into the corresponding direction
%       cosine matrix
%
%   [dcm] = quat2dcm(quat)
%
%   Inputs:
%       - quat - Nx4 element quaternion where the last element is the scalar
%       parameter ( [e1 e2 e3 n] ) describing the rotation from frame a to
%       frame b
%
%   Outputs:
%       - dcm - 3x3xN rotation matrix describing rotation from frame a to
%       frame b. Assumes row vector format b = a * dcm_a2b
%
%   Dependencies:
%       - quatnorm.m - normalize a quaternion - vectorized
%
%   Author:
%       - Shankar Kulumani 24 Jan 2013
%       - Shankar Kulumani 12 Feb 2013
%           - vectorized and quaternion norm check is implemented
%       - Shankar Kulumani 24 Feb 2013
%           - modified size check for vectorization
%
%   References
%       - AAE590 Lesson 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [dcm] = quat2dcm(quat)

N  = size(quat,1);
% perform normalization check on inpute quaternion
quat = quatnorm(quat);

% quaternion is 4 element parameter where the 4th element is the scalar
% element
e1 = quat(:,1);
e2 = quat(:,2);
e3 = quat(:,3);
e4 = quat(:,4);

% form the quaternion products ahead of time
e1e1=e1.*e1;
e1e2=e1.*e2;
e1e3=e1.*e3;
e1e4=e1.*e4;

e2e2=e2.*e2;
e2e3=e2.*e3;
e2e4=e2.*e4;

e3e3=e3.*e3;
e3e4=e3.*e4;

e4e4=e4.*e4;

% dcm is for row element vectors b = a*dcm_a2b (b - 1x3 a-1x3)
dcm = zeros(3,3,N);

dcm(1,1,:) = 1-2*e2e2-2*e3e3;
dcm(1,2,:) = 2*(e1e2 - e3e4);
dcm(1,3,:) = 2*(e1e3 + e2e4);
dcm(2,1,:) = 2*(e1e2+e3e4);
dcm(2,2,:) = 1-2*e3e3-2*e1e1;
dcm(2,3,:) = 2*(e2e3-e1e4);
dcm(3,1,:) = 2*(e1e3 - e2e4);
dcm(3,2,:) = 2*(e2e3+e1e4);
dcm(3,3,:) = 1-2*e1e1-2*e2e2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion to Euler Space 1-3-1
%
%   Purpose:
%       - Convert a quaternion to the equivalaent Euler Space 1-3-1 sequence
%       about the body fixed axes.
%
%   function theta = quat2space131(quat)
%
%   Inputs:
%       - quat - 1x4 quaternion describing the orientation of the body
%       frame wrt inertial frame. The 4th element is the scalar value.
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] radians
%
%   Dependencies:
%       - quat2dcm.m - convert quaternion to dcm
%       - dcm2space131.m - convert dcm to euler angles
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = quat2space131(quat)

dcm = quat2dcm(quat);

theta = dcm2space131(dcm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion to Euler Space 2-1-3
%
%   Purpose:
%       - Convert a quaternion to the equivalaent Euler Space 2-1-3 sequence
%       about the body fixed axes.
%
%   theta = quat2space213(quat)
%
%   Inputs:
%       - quat - 1x4 quaternion describing the orientation of the body
%       frame wrt inertial frame. The 4th element is the scalar value.
%
%   Outputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] radians
%
%   Dependencies:
%       - quat2dcm.m - convert quaternion to dcm
%       - dcm2space213.m - convert dcm to euler angles
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 9
%       - H. Schaub and J. Junkins. Matlab toolbox for rigid body kinematics. Spaceflight mechanics 1999, pages 549?560, 1999.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta = quat2space213(quat)

dcm = quat2dcm(quat);

theta = dcm2space213(dcm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion to Euler Axis/Angle
%
%   Purpose:
%       - Converts a quaternion to the equivalent euler axis and angle
%       describing the rotation
%
%   [lambda theta] = quat2srt(quat)
%
%   Inputs:
%       - quat - 4 element quaternion where the last element is the scalar
%       parameter ( [e1 e2 e3 n] ) describing the rotation from frame a to
%       frame b
%
%   Outputs:
%       - lambda - euler axis of rotation 3x1 unit vector describing
%       rotation from frame a to frame b
%       - theta - euler angle of rotation in radians
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 24 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [lambda theta] = quat2srt(quat)

e = quat(1:3);
n = quat(4);

lambda = zeros(3,1);

lambda = e/norm(e);
theta = 2*acos(n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion differential equation
%
%   Purpose:
%       - Calculates the quaternion differential equation in terms of
%       angular velocity in the body frame
%
%   [q_d] = quatdot(q, w)
%
%   Inputs:
%       - q - 1x4 quaternion representing the orientation of the body wrt
%       to inertial frame. the last element is the scalar component
%       - w - 1x3 angular velocity vector in teh body frame components.
%
%   Outputs:
%       - q_d - 1x4 quaternion derivative with components in either the
%       inertial or fixed frame
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [q_d] = quatdot(q, w)

E = [q(4) -q(3) q(2) q(1);...
    q(3) q(4) -q(1) q(2);...
    -q(2) q(1) q(4) q(3);...
    -q(1) -q(2) -q(3) q(4)];

w = [w 0];

q_d = 1/2*w*E';

function w = quatdot2omega(q,qd)

% form the E matrix
E = [q(4) -q(3) q(2) q(1);...
    q(3) q(4) -q(1) q(2);...
    -q(2) q(1) q(4) q(3);...
    -q(1) -q(2) -q(3) q(4)];

w = 2*qd*E;

w = w(1:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Quaternion Normalize
%
%   Purpose:
%       - Normalize quaternions to avoid problems - vectorized
%
%   quat_out = quatnorm(quat_in)
%
%   Inputs:
%       - quat_in - Nx4 element quaternion where the last element is the scalar
%       parameter ( [e1 e2 e3 n] ) describing the rotation from frame a to
%       frame b
%
%   Outputs:
%       - quat_out - Nx4 element normalized quaternion where the last element is the scalar
%       parameter ( [e1 e2 e3 n] ) describing the rotation from frame a to
%       frame b
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 12 Feb 2013
%       - Shankar Kulumani 2 Mar 2013
%           - modified repmat command
%
%   References
%       - AAE590
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quat_out = quatnorm(quat_in)

N = size(quat_in,1);

% Find the magnitude of each quaternion
quat_mag=sqrt(sum(quat_in.^2,2));

% resize quat mag
quat_mag = repmat(quat_mag,1,4);

% Divide each element of q by appropriate qmag
quat_out=quat_in./quat_mag;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1 line code description
%
%   Purpose:
%       - Converts a vector in frame a to the corresponding vector in frame
%       b given a quaternion
%
%    b = quat_rotvec(quat,a)
%
%   Inputs:
%       - quat - 4 element quaternion where the last element is the scalar
%       parameter ( [e1 e2 e3 n] ) describing the rotation from frame a to
%       frame b
%
%   Outputs:
%       - b - a vector rotated into frame b
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 24 Jan 2013
%           - list revisions
%
%   References
%       - AAE590 Lesson 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function b = quat_rotvec(quat,a)

e = quat(1:3);
n = quat(4);

a_e_cross = cross(a,e);
b = a - 2*n*a_e_cross + 2*cross(e,-a_e_cross);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Successive Quaternion Rotations
%
%   Purpose:
%       - Combines 2 successive quaternion represented rotations into a
%       single quaternion rotation
%
%   quat_A2B= quat_succesive(quat_A2B1,quat_B12B)
%
%   Inputs:
%       - quat_A2B1 - nx4 quaternion representing first rotation. Frame A
%       to B1
%       - quat_B12B - nx4 quaternion representing second rotation. Frame B1
%       to B
%
%   Outputs:
%       - quat_A2B - nx4 final quaternion rotation from frame A to B
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 26 Jan 2013
%       - Shankar Kulumani 5 Mar 2013
%           - went back to vector equations
%       - Shankar Kulumani 21 Mar 2013
%           - vectorized for many quaternions
%
%   References
%       - AAE590 Lesson 8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quat_A2B= quat_succesive(quat_A2B1,quat_B12B)

% e_A2B = zeros(1,3);
% n_A2B = 0;

e_A2B1 = quat_A2B1(:,1:3);
n_A2B1 = quat_A2B1(:,4);

e_B12B = quat_B12B(:,1:3);
n_B12B = quat_B12B(:,4);

e_A2B = e_A2B1.*repmat(n_B12B,1,3) + e_B12B.*repmat(n_A2B1,1,3) + cross(e_B12B,e_A2B1);
n_A2B = n_A2B1.*n_B12B - dot(e_A2B1,e_B12B,2);

quat_A2B = [e_A2B n_A2B];

% A = [n_A2B1 -e_A2B1(3) e_A2B1(2) e_A2B1(1);...
%     e_A2B1(3) n_A2B1 -e_A2B1(1) e_A2B1(2);...
%     -e_A2B1(2) e_A2B1(1) n_A2B1 e_A2B1(3);...
%     -e_A2B1(1) -e_A2B1(2) -e_A2B1(3) n_A2B1];
%
% quat_A2B = A*quat_B12B';
% quat_A2B = quat_A2B';%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Purpose: Rotation matrix about first axis (assumes row format)
%   b = a*dcm
%
%   Inputs:
%       - alpha - rotation angle (rad)
%
%   Outpus:
%       - rot1 - rotation matrix (3x3)
%
%   Dependencies:
%       - none
%
%   Author: Shankar Kulumani 18 Aug 2012
%           26 Jan 2013 - converted to row format representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rot1 = ROT1(alpha)

cos_alpha = cos(alpha);
sin_alpha = sin(alpha);

rot1 = [1     0          0      ;    ...
        0  cos_alpha  -sin_alpha ;    ...
        0 sin_alpha  cos_alpha ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Purpose: Rotation matrix about second axis (assumes row format)
%   b = a*dcm
%
%   Inputs:
%       - beta - rotation angle (rad)
%
%   Outpus:
%       - rot2 - rotation matrix (3x3)
%
%   Dependencies:
%       - none
%
%   Author: Shankar Kulumani 18 Aug 2012
%           26 Jan 2013 - converted to row format representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rot2 = ROT2(beta)


cos_beta = cos(beta);
sin_beta = sin(beta);

rot2 = [cos_beta  0  sin_beta;   ...
           0      1      0    ;   ...
        -sin_beta  0  cos_beta ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Purpose: Rotation matrix about thrid axis (assumes row format)
%   b = a*dcm_a2b
%
%   Inputs:
%       - gamma - rotation angle (rad)
%
%   Outpus:
%       - rot3 - rotation matrix (3x3)
%
%   Dependencies:
%       - none
%
%   Author: Shankar Kulumani 18 Aug 2012
%               - 15 Sept 2012 fixed error
%               - 26 Jan 2013 - converted to row format representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rot3 = ROT3(gamma)

cos_gamma = cos(gamma);
sin_gamma = sin(gamma);

rot3 = [ cos_gamma -sin_gamma  0 ;   ...
        sin_gamma cos_gamma  0 ;   ...
            0         0       1 ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Space 1-3-1 Differential Equation
%
%   Purpose:
%       - Finds the rate of change of the Euler Space 1-3-1 angles
%
%   theta_d = space131dot(theta,w)
%
%   Inputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%       - w - 1x3 angular velocity vector in teh body frame components.
%
%   Outputs:
%       - theta_d - 3 element vector with the 3 rotation angle derivatives. Same order
%       as m-file filename. theta = [first second third] rad/sec
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Omega Angle Rates pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta_d = space131dot(theta,w)

c2 = cos(theta(2));
c1 = cos(theta(1));
s2 = sin(theta(2));
s1 = sin(theta(1));

theta_d(1) = w(1) + (w(2)*c1 -w(3)*s1)*c2/s2;
theta_d(2) = w(2)*s1+w(3)*c1;
theta_d(3) = (-w(2)*c1 +w(3)*s1)/s2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Space 2-1-3 Differential Equation
%
%   Purpose:
%       - Finds the rate of change of the Euler Space 2-1-3 angles
%
%   theta_d = space213dot(theta,w)
%
%   Inputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] rad
%       - w - 1x3 angular velocity vector in teh body frame components in
%       rad/sec
%
%   Outputs:
%       - theta_d - 3 element vector with the 3 rotation angle derivatives. Same order
%       as m-file filename. theta = [first second third] rad/sec
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 2 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Omega Angle Rates pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function theta_d = space213dot(theta,w)

c1 = cos(theta(1));
c2 = cos(theta(2));
s1 = sin(theta(1));
s2 = sin(theta(2));

theta_d(1) = (w(1)*s1-w(3)*c1)*s2/c2+w(2);
theta_d(2) = w(1)*c1+w(3)*s1;
theta_d(3) = (-w(1)*s1+w(3)*c1)/c2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Space 2-3-1 Differential Equation
%
%   Purpose:
%       - Finds the rate of change of the Euler Space 2-3-1 angles
%
%   theta_d = space231dot(theta,w)
%
%   Inputs:
%       - theta - 3 element vector with the 3 rotation angles. Same order
%       as m-file filename. theta = [first second third] in radians
%       - w - 1x3 angular velocity vector in teh body frame components in
%       rad/sec
%
%   Outputs:
%       - theta_d - 3 element vector with the 3 rotation angle derivatives. Same order
%       as m-file filename. theta = [first second third] rad/sec
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 4 Feb 2013
%           - list revisions
%
%   References
%       - AAE590 Omega Angle Rates pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function theta_d = space231dot(theta,w)

ct1 = cos(theta(1));
st1 = sin(theta(1));
st2 = sin(theta(2));
ct2 = cos(theta(2));

theta_d = zeros(1,3);

theta_d(1) = (w(1)*ct1+w(3)*st1)*st2/ct2 + w(2);
theta_d(2) = -w(1)*st1 + w(3)*ct1;
theta_d(3) = (w(1)*ct1 + w(3)*st1)/ct2;

function OUTPUT=SpinCalc(CONVERSION,INPUT,tol,ichk)
%Function for the conversion of one rotation input type to desired output.
%Supported conversion input/output types are as follows:
%   1: Q        Rotation Quaternions
%   2: EV       Euler Vector and rotation angle (degrees)
%   3: DCM      Orthogonal DCM Rotation Matrix
%   4: EA###    Euler angles (12 possible sets) (degrees)
%
%Author: John Fuller
%National Institute of Aerospace
%Hampton, VA 23666
%John.Fuller@nianet.org
%
%Version 1.3
%June 30th, 2009
%
%Version 1.3 updates
%   SpinCalc now detects when input data is too close to Euler singularity, if user is choosing
%   Euler angle output. Prohibits output if middle angle is within 0.1 degree of singularity value.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%                OUTPUT=SpinCalc(CONVERSION,INPUT,tol,ichk)
%Inputs:
%CONVERSION - Single string value that dictates the type of desired
%             conversion.  The conversion strings are listed below.
%
%   'DCMtoEA###'  'DCMtoEV'    'DCMtoQ'       **for cases that involve
%   'EA###toDCM'  'EA###toEV'  'EA###toQ'       euler angles, ### should be
%   'EVtoDCM'     'EVtoEA###'  'EVtoQ'          replaced with the proper
%   'QtoDCM'      'QtoEA###'   'QtoEV'          order desired.  EA321 would
%   'EA###toEA###'                              be Z(yaw)-Y(pitch)-X(roll).
%
%INPUT - matrix or vector that corresponds to the first entry in the
%        CONVERSION string, formatted as follows:
%
%        DCM - 3x3xN multidimensional matrix which pre-multiplies a coordinate
%              frame column vector to calculate its coordinates in the desired
%              new frame.
%
%        EA### - [psi,theta,phi] (Nx3) row vector list dictating to the first angle
%                rotation (psi), the second (theta), and third (phi) (DEGREES)
%
%        EV - [m1,m2,m3,MU] (Nx4) row vector list dictating the components of euler
%             rotation vector (original coordinate frame) and the Euler
%             rotation angle about that vector (MU) (DEGREES)
%
%        Q - [q1,q2,q3,q4] (Nx4) row vector list defining quaternion of
%            rotation.  q4 = cos(MU/2) where MU is Euler rotation angle
%
%tol - tolerance value
%ichk - 0 disables warning flags
%          1 enables warning flags (near singularities)
%**NOTE: N corresponds to multiple orientations
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Output:
%OUTPUT - matrix or vector corresponding to the second entry in the
%         CONVERSION input string, formatted as shown above.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%Pre-processer to determine type of conversion from CONVERSION string input
%Types are numbered as follows:
%Q=1   EV=2   DCM=3   EA=4
i_type=strfind(lower(CONVERSION),'to');
length=size(CONVERSION,2);
if length>12 || length<4,   %no CONVERSION string can be shorter than 4 or longer than 12 chars
    error('Error: Invalid entry for CONVERSION input string');
end
o_type=length-i_type;
if i_type<5,
    i_type=i_type-1;
else
    i_type=i_type-2;
end
if o_type<5,
    o_type=o_type-1;
else
    o_type=o_type-2;
end
TYPES=cell(1,4);
TYPES{1,1}='Q'; TYPES{1,2}='EV'; TYPES{1,3}='DCM'; TYPES{1,4}='EA';
INPUT_TYPE=TYPES{1,i_type};
OUTPUT_TYPE=TYPES{1,o_type};
clear TYPES
%Confirm input as compared to program interpretation
if i_type~=4 && o_type~=4,  %if input/output are NOT Euler angles
    CC=[INPUT_TYPE,'to',OUTPUT_TYPE];
    if strcmpi(CONVERSION,CC)==0;
        error('Error: Invalid entry for CONVERSION input string');
    end
else
    if i_type==4,   %if input type is Euler angles, determine the order of rotations
        EULER_order_in=str2double(CONVERSION(1,3:5));
        rot_1_in=floor(EULER_order_in/100);     %first rotation
        rot_2_in=floor((EULER_order_in-rot_1_in*100)/10);   %second rotation
        rot_3_in=(EULER_order_in-rot_1_in*100-rot_2_in*10);   %third rotation
        if rot_1_in<1 || rot_2_in<1 || rot_3_in<1 || rot_1_in>3 || rot_2_in>3 || rot_3_in>3,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that all orders are between 1 and 3
        elseif rot_1_in==rot_2_in || rot_2_in==rot_3_in,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that no 2 consecutive orders are equal (invalid)
        end
        %check input dimensions to be 1x3x1
        if size(INPUT,2)~=3 || size(INPUT,3)~=1
            error('Error: Input euler angle data vector is not Nx3')
        end
        %identify singularities
        if rot_1_in==rot_3_in, %Type 2 rotation (first and third rotations about same axis)
            if INPUT(:,2)<=zeros(size(INPUT,1),1) | INPUT(:,2)>=180*ones(size(INPUT,1),1),  %confirm second angle within range
                error('Error: Second input Euler angle(s) outside 0 to 180 degree range')
            elseif abs(INPUT(:,2))<2*ones(size(INPUT,1),1) | abs(INPUT(:,2))>178*ones(size(INPUT,1),1),  %check for singularity
                if ichk==1,
                    errordlg('Warning: Input Euler angle rotation(s) near a singularity.               Second angle near 0 or 180 degrees.')
                end
            end
        else    %Type 1 rotation (all rotations about each of three axes)
            if abs(INPUT(:,2))>=90*ones(size(INPUT,1),1), %confirm second angle within range
                error('Error: Second input Euler angle(s) outside -90 to 90 degree range')
            elseif abs(INPUT(:,2))>88*ones(size(INPUT,1),1),  %check for singularity
                if ichk==1,
                    errordlg('Warning: Input Euler angle(s) rotation near a singularity.               Second angle near -90 or 90 degrees.')
                end
            end
        end
    end
    if o_type==4,   %if output type is Euler angles, determine order of rotations
        EULER_order_out=str2double(CONVERSION(1,length-2:length));
        rot_1_out=floor(EULER_order_out/100);   %first rotation
        rot_2_out=floor((EULER_order_out-rot_1_out*100)/10);    %second rotation
        rot_3_out=(EULER_order_out-rot_1_out*100-rot_2_out*10); %third rotation
        if rot_1_out<1 || rot_2_out<1 || rot_3_out<1 || rot_1_out>3 || rot_2_out>3 || rot_3_out>3,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that all orders are between 1 and 3
        elseif rot_1_out==rot_2_out || rot_2_out==rot_3_out,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that no 2 consecutive orders are equal
        end
    end
    if i_type==4 && o_type~=4,  %if input are euler angles but not output
        CC=['EA',num2str(EULER_order_in),'to',OUTPUT_TYPE]; %construct program conversion string for checking against user input
    elseif o_type==4 && i_type~=4,  %if output are euler angles but not input
        CC=[INPUT_TYPE,'to','EA',num2str(EULER_order_out)]; %construct program conversion string for checking against user input
    elseif i_type==4 && o_type==4,  %if both input and output are euler angles
        CC=['EA',num2str(EULER_order_in),'to','EA',num2str(EULER_order_out)];   %construct program conversion string
    end
    if strcmpi(CONVERSION,CC)==0; %check program conversion string against user input to confirm the conversion command
        error('Error: Invalid entry for CONVERSION input string');
    end
end
clear i_type o_type CC

%From the input, determine the quaternions that uniquely describe the
%rotation prescribed by that input.  The output will be calculated in the
%second portion of the code from these quaternions.
switch INPUT_TYPE
    case 'DCM'
        if size(INPUT,1)~=3 || size(INPUT,2)~=3  %check DCM dimensions
            error('Error: DCM matrix is not 3x3xN');
        end
        N=size(INPUT,3);    %number of orientations
        %Check if matrix is indeed orthogonal
        perturbed=NaN(3,3,N);
        DCM_flag=0;
        for ii=1:N,
            perturbed(:,:,ii)=abs(INPUT(:,:,ii)*INPUT(:,:,ii)'-eye(3)); %perturbed array shows difference between DCM*DCM' and I
            if abs(det(INPUT(:,:,ii))-1)>tol, %if determinant is off by one more than tol, user is warned.
                if ichk==1,
                    DCM_flag=1;
                end
            end
            if abs(det(INPUT(:,:,ii))+1)<0.05, %if determinant is near -1, DCM is improper
                error('Error: Input DCM(s) improper');
            end
            if DCM_flag==1,
                errordlg('Warning: Input DCM matrix determinant(s) off from 1 by more than tolerance.')
            end
        end
        DCM_flag=0;
        if ichk==1,
            for kk=1:N,
                for ii=1:3,
                    for jj=1:3,
                        if perturbed(ii,jj,kk)>tol,   %if any difference is larger than tol, user is warned.
                            DCM_flag=1;
                        end
                    end
                end
            end
            if DCM_flag==1,
                fprintf('Warning: Input DCM(s) matrix not orthogonal to precision tolerance.')
            end
        end
        clear perturbed DCM_flag
        Q=NaN(4,N);
        for ii=1:N,
            denom=NaN(4,1);
            denom(1)=0.5*sqrt(1+INPUT(1,1,ii)-INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(2)=0.5*sqrt(1-INPUT(1,1,ii)+INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(3)=0.5*sqrt(1-INPUT(1,1,ii)-INPUT(2,2,ii)+INPUT(3,3,ii));
            denom(4)=0.5*sqrt(1+INPUT(1,1,ii)+INPUT(2,2,ii)+INPUT(3,3,ii));
            %determine which Q equations maximize denominator
            switch find(denom==max(denom),1,'first')  %determines max value of qtests to put in denominator
                case 1
                    Q(1,ii)=denom(1);
                    Q(2,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(1,ii));
                    Q(3,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(1,ii));
                    Q(4,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(1,ii));
                case 2
                    Q(2,ii)=denom(2);
                    Q(1,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(2,ii));
                    Q(3,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(2,ii));
                    Q(4,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(2,ii));
                case 3
                    Q(3,ii)=denom(3);
                    Q(1,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(3,ii));
                    Q(2,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(3,ii));
                    Q(4,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(3,ii));
                case 4
                    Q(4,ii)=denom(4);
                    Q(1,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(4,ii));
                    Q(2,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(4,ii));
                    Q(3,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(4,ii));
            end
        end
        Q=Q';
        clear denom
    case 'EV'  %Euler Vector Input Type
        if size(INPUT,2)~=4 || size(INPUT,3)~=1   %check dimensions
            error('Error: Input euler vector and rotation data matrix is not Nx4')
        end
        N=size(INPUT,1);
        MU=INPUT(:,4)*pi/180;  %assign mu name for clarity
        if sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2)-ones(N,1)>tol*ones(N,1),  %check that input m's constitute unit vector
            error('Input euler vector(s) components do not constitute a unit vector')
        end
        if MU<zeros(N,1) | MU>2*pi*ones(N,1), %check if rotation about euler vector is between 0 and 360
            error('Input euler rotation angle(s) not between 0 and 360 degrees')
        end
        Q=[INPUT(:,1).*sin(MU/2),INPUT(:,2).*sin(MU/2),INPUT(:,3).*sin(MU/2),cos(MU/2)];   %quaternion
        clear m1 m2 m3 MU
    case 'EA'
        psi=INPUT(:,1)*pi/180;  theta=INPUT(:,2)*pi/180;  phi=INPUT(:,3)*pi/180;
        N=size(INPUT,1);    %number of orientations
        %Pre-calculate cosines and sines of the half-angles for conversion.
        c1=cos(psi./2); c2=cos(theta./2); c3=cos(phi./2);
        s1=sin(psi./2); s2=sin(theta./2); s3=sin(phi./2);
        c13=cos((psi+phi)./2);  s13=sin((psi+phi)./2);
        c1_3=cos((psi-phi)./2);  s1_3=sin((psi-phi)./2);
        c3_1=cos((phi-psi)./2);  s3_1=sin((phi-psi)./2);
        if EULER_order_in==121,
            Q=[c2.*s13,s2.*c1_3,s2.*s1_3,c2.*c13];
        elseif EULER_order_in==232,
            Q=[s2.*s1_3,c2.*s13,s2.*c1_3,c2.*c13];
        elseif EULER_order_in==313;
            Q=[s2.*c1_3,s2.*s1_3,c2.*s13,c2.*c13];
        elseif EULER_order_in==131,
            Q=[c2.*s13,s2.*s3_1,s2.*c3_1,c2.*c13];
        elseif EULER_order_in==212,
            Q=[s2.*c3_1,c2.*s13,s2.*s3_1,c2.*c13];
        elseif EULER_order_in==323,
            Q=[s2.*s3_1,s2.*c3_1,c2.*s13,c2.*c13];
        elseif EULER_order_in==123,
            Q=[s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==231,
            Q=[c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==312,
            Q=[c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==132,
            Q=[s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==213,
            Q=[c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==321,
            Q=[c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        else
            error('Error: Invalid input Euler angle order type (conversion string)');
        end
        clear c1 s1 c2 s2 c3 s3 c13 s13 c1_3 s1_3 c3_1 s3_1 psi theta phi
    case 'Q'
        if size(INPUT,2)~=4 || size(INPUT,3)~=1
            error('Error: Input quaternion matrix is not Nx4');
        end
        N=size(INPUT,1);    %number of orientations
        if ichk==1,
            if abs(sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2+INPUT(:,4).^2)-ones(N,1))>tol*ones(N,1)
                errordlg('Warning: Input quaternion norm(s) deviate(s) from unity by more than tolerance')
            end
        end
        Q=INPUT;
end
clear INPUT INPUT_TYPE EULER_order_in

%Normalize quaternions in case of deviation from unity.  User has already
%been warned of deviation.
Qnorms=sqrt(sum(Q.*Q,2));
Q=[Q(:,1)./Qnorms,Q(:,2)./Qnorms,Q(:,3)./Qnorms,Q(:,4)./Qnorms];

switch OUTPUT_TYPE
    case 'DCM'
        Q=reshape(Q',1,4,N);
        OUTPUT=[Q(1,1,:).^2-Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,1,:).*Q(1,2,:)+Q(1,3,:).*Q(1,4,:)),2*(Q(1,1,:).*Q(1,3,:)-Q(1,2,:).*Q(1,4,:));
                2*(Q(1,1,:).*Q(1,2,:)-Q(1,3,:).*Q(1,4,:)),-Q(1,1,:).^2+Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,2,:).*Q(1,3,:)+Q(1,1,:).*Q(1,4,:));
                2*(Q(1,1,:).*Q(1,3,:)+Q(1,2,:).*Q(1,4,:)),2*(Q(1,2,:).*Q(1,3,:)-Q(1,1,:).*Q(1,4,:)),-Q(1,1,:).^2-Q(1,2,:).^2+Q(1,3,:).^2+Q(1,4,:).^2];
    case 'EV'
        MU=2*atan2(sqrt(sum(Q(:,1:3).*Q(:,1:3),2)),Q(:,4));
        if sin(MU/2)~=zeros(N,1),
            OUTPUT=[Q(:,1)./sin(MU/2),Q(:,2)./sin(MU/2),Q(:,3)./sin(MU/2),MU*180/pi];
        else
            OUTPUT=NaN(N,4);
            for ii=1:N,
                if sin(MU(ii,1)/2)~=0,
                    OUTPUT(ii,1:4)=[Q(ii,1)/sin(MU(ii,1)/2),Q(ii,2)/sin(MU(ii,1)/2),Q(ii,3)/sin(MU(ii,1)/2),MU(ii,1)*180/pi];
                else
                    OUTPUT(ii,1:4)=[1,0,0,MU(ii,1)*180/pi];
                end
            end
        end
    case 'Q'
        OUTPUT=Q;
    case 'EA'
        if EULER_order_out==121,
            psi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
            theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
	  Euler_type=2;
        elseif EULER_order_out==232;
            psi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
            theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
	  Euler_type=2;
        elseif EULER_order_out==313;
            psi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
            theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==131;
            psi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
            theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
	  Euler_type=2;
        elseif EULER_order_out==212;
            psi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
            theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==323;
            psi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
            theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==123;
            psi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
            phi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==231;
            psi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
            phi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==312;
            psi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
            phi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,3).*Q(:,1)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==132;
            psi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
            phi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==213;
            psi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
            phi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==321;
            psi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
            phi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
	  Euler_type=1;
        else
            error('Error: Invalid output Euler angle order type (conversion string).');
        end
        if(isreal([psi,theta,phi]))==0,
	        error('Error: Unreal Euler output.  Input resides too close to singularity.  Please choose different output type.')
        end
        OUTPUT=mod([psi,theta,phi]*180/pi,360);  %deg
        if Euler_type==1,
	        sing_chk=find(abs(theta)*180/pi>89.9);
	        sing_chk=sort(sing_chk(sing_chk>0));
	        if size(sing_chk,1)>=1,
		        error('Error: Input rotation #%s resides too close to Type 1 Euler singularity.\nType 1 Euler singularity occurs when second angle is -90 or 90 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
	        end
        elseif Euler_type==2,
	        sing_chk=[find(abs(theta*180/pi)<0.1);find(abs(theta*180/pi-180)<0.1);find(abs(theta*180/pi-360))<0.1];
	        sing_chk=sort(sing_chk(sing_chk>0));
	        if size(sing_chk,1)>=1,
		        error('Error: Input rotation #%s resides too close to Type 2 Euler singularity.\nType 2 Euler singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
	        end
        end
end






function [] = SpinCalcVis()
%==========================================================================
%SpinCalcVis Rotational Visualization Tool  V1.0
%Author: John Fuller
%        imaginationdirect@gmail.com
%
%This is an instructional GUI to be used for learning how Euler angles,
%DCMs, quaternions, and Euler vector parameters relate in rotation of
%cartesian frames (A to B).
%
%For the function-based rotation conversion, please see SpinCalc:
%
%http://www.mathworks.com/matlabcentral/fileexchange/20696-function-to-conv
%ert-between-dcm-euler-angles-quaternions-and-euler-vectors
%
%Uses an enhanced uicontrol GUI function for support of LaTeX formatting:
%Function uibutton, Author: Douglas Schwarz
%
%Source:
%http://www.mathworks.com/matlabcentral/fileexchange/10743-uibutton-gui-pus
%hbuttons-with-better-labels
%==========================================================================
S.fh = figure('units','pixels',...
              'position',[300 150 940 500],...
              'menubar','none',...
              'resize','off',...
              'numbertitle','off',...
              'name','SpinCalcVis: Rotation Visualization GUI');

S.axes1=axes('units','pixels',...
             'position',[460 12 470 470],...
             'view',[58 28],...
             'xtick',[],...
             'ytick',[],...
             'ztick',[]);

plot3([0;1],[0;0],[0;0],':r')
hold on
axis equal
text(1.05,0,-0.01,'X_{A}','color','r')
plot3([0;0],[0;1],[0;0],':g')
text(0,1.05,-0.01,'Y_{A}','color','g')
plot3([0;0],[0;0],[0;1],':b')
text(-0.01,-0.01,1.05,'Z_{A}','color','b')
xlim([-1.0,1.0])
ylim([-1.0,1.0])
zlim([-1.0,1.0])
set(gca,'xtick',[],'ytick',[],'ztick',[],'box','off')
view([58 28])
hold off
rotate3d(S.axes1)

S.fr1 = uicontrol('style','frame',...
                 'units','pixels',...
                 'position',[10 405 440 85]);

S.title1 = uicontrol('style','text',...
                    'units','pixels',...
                    'position',[(450)/2-100 460-20 200 40],...
                    'string',{'SpinCalcVis';'Rotational Visualization Tool'},...
                    'FontSize',10,...
                    'FontWeight','bold');

S.title2 = uicontrol('style','text',...
                    'units','pixels',...
                    'position',get(S.title1,'position')+[0 -34 0 -5],...
                    'string',{'Author: John Fuller';'Version 1.0'});

S.fr2 = uicontrol('style','frame',...
                 'units','pixels',...
                 'position',[20 10 420 60]);

S.fr3 = uicontrol('style','frame',...
                 'units','pixels',...
                 'position',[358 255 90 58]);

S.txtradios = uicontrol('style','text',...
                        'units','pixels',...
                        'position',[358 320 80 12],...
                        'BackgroundColor',0.7967*[1 1 1],...
                        'string','Plotting');

S.dialog = uicontrol('style','text',...
                    'units','pixels',...
                    'position',[24 12 415 56],...
                    'HorizontalAlignment','left',...
                    'Foregroundcolor',0.3*[1 1 1],...
                    'string',{'No Output'});

S.dialogtxt = uicontrol('style','text',...
                     'units','pixels',...
                     'position',[12, 70 60 18],...
                     'BackgroundColor',0.7967*[1 1 1],...
                     'string','Dialog Box');

S.txt1 = uicontrol('style','text',...
                   'units','pixels',...
                   'position',[13 380 100 12],...
				   'BackgroundColor',0.7967*[1 1 1],...
                   'string','Euler Angles (deg)');

S.ea1 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 355 80 18],...
                  'BackgroundColor',[1 1 1],...
                  'String',0,...
                  'Value',0);

S.ea2 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.ea1,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.ea3 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.ea2,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.ea1txt = uibutton('style','text',...
                    'units','pixels',...
                    'position',[20 338 80 18],...
				    'BackgroundColor',0.7967*[1 1 1],...
                    'string','\psi');

S.ea2txt = uibutton('style','text',...
                    'units','pixels',...
                    'position',[110 338 80 18],...
				    'BackgroundColor',0.7967*[1 1 1],...
                    'string','\theta');

S.ea3txt = uibutton('style','text',...
                    'units','pixels',...
                    'position',[200 338 80 18],...
				    'BackgroundColor',0.7967*[1 1 1],...
                    'string','\phi');

set(S.ea1,'Callback',{@assignea1,S});
set(S.ea2,'Callback',{@assignea2,S});
set(S.ea3,'Callback',{@assignea3,S});

S.eamenu = uicontrol('style','popupmenu',...
                     'units','pixels',...
                     'position',get(S.ea3,'position')+[90 2 0 0],...
                     'BackgroundColor',[1 1 1],...
                     'String',{'XYZ  (123)';'XZY  (132)';'YXZ  (213)';'YZX  (231)';'ZXY  (312)';'ZYX  (321)';'XYX  (121)';'XZX  (131)';'YXY  (212)';'YZY  (232)';'ZXZ  (313)';'ZYZ  (323)'});

S.txt2 = uicontrol('style','text',...
                   'units','pixels',...
                   'position',get(S.eamenu,'position')+[-2 18 0 -5],...
				   'BackgroundColor',0.7967*[1 1 1],...
                   'string','Order');

S.txt3 = uicontrol('style','text',...
                   'units','pixels',...
                   'position',[13 320 160 12],...
				   'BackgroundColor',0.7967*[1 1 1],...
                   'string','Directions Cosine Matrix (DCM)');

S.dcm11 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 295 80 18],...
                  'String',1,...
                  'Value',1,...
                  'BackgroundColor',[1 1 1]);

S.dcm12 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm11,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm13 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm12,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm21 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 275 80 18],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm22 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm21,'position')+[90 0 0 0],...
                  'String',1,...
                  'Value',1,...
                  'BackgroundColor',[1 1 1]);

S.dcm23 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm22,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm31 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 255 80 18],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm32 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm31,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.dcm33 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.dcm32,'position')+[90 0 0 0],...
                  'String',1,...
                  'Value',1,...
                  'BackgroundColor',[1 1 1]);

set(S.dcm11,'Callback',{@assigndcm11,S});
set(S.dcm12,'Callback',{@assigndcm12,S});
set(S.dcm13,'Callback',{@assigndcm13,S});
set(S.dcm21,'Callback',{@assigndcm21,S});
set(S.dcm22,'Callback',{@assigndcm22,S});
set(S.dcm23,'Callback',{@assigndcm23,S});
set(S.dcm31,'Callback',{@assigndcm31,S});
set(S.dcm32,'Callback',{@assigndcm32,S});
set(S.dcm33,'Callback',{@assigndcm33,S});

S.txt4 = uicontrol('style','text',...
                   'units','pixels',...
                   'position',[13 225 64 12],...
				   'BackgroundColor',0.7967*[1 1 1],...
                   'string','Quaternion');

S.q1 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 200 80 18],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.q2 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.q1,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.q3 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.q2,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.q4 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.q3,'position')+[90 0 0 0],...
                  'String',1,...
                  'Value',1,...
                  'BackgroundColor',[1 1 1]);

set(S.q1,'Callback',{@assignq1,S});
set(S.q2,'Callback',{@assignq2,S});
set(S.q3,'Callback',{@assignq3,S});
set(S.q4,'Callback',{@assignq4,S});

S.q1txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[20 180 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{1}sin(\mu/2)');

S.q2txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[110 180 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{2}sin(\mu/2)');

S.q3txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[200 180 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{3}sin(\mu/2)');

S.q3txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[290 183 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','cos(\mu/2)');

S.txt6 = uicontrol('style','text',...
                   'units','pixels',...
                   'position',[13 155 90 12],...
				   'BackgroundColor',0.7967*[1 1 1],...
                   'string','Euler Parameters');

S.ep1 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',[20 130 80 18],...
                  'String',1,...
                  'Value',1,...
                  'BackgroundColor',[1 1 1]);

S.ep2 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.ep1,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.ep3 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.ep2,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

S.ep4 = uicontrol('style','edit',...
                  'units','pixels',...
                  'position',get(S.ep3,'position')+[90 0 0 0],...
                  'String',0,...
                  'Value',0,...
                  'BackgroundColor',[1 1 1]);

set(S.ep1,'Callback',{@assignep1,S});
set(S.ep2,'Callback',{@assignep2,S});
set(S.ep3,'Callback',{@assignep3,S});
set(S.ep4,'Callback',{@assignep4,S});

S.ep1txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[20 110 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{1}');

S.ep2txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[110 110 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{2}');

S.ep3txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[200 110 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','m_{3}');

S.ep3txt = uibutton('style','text',...
                  'units','pixels',...
                  'position',[290 115 80 18],...
				  'BackgroundColor',0.7967*[1 1 1],...
                  'string','\mu');

S.exitbutton = uibutton('style','pushbutton',...
                        'units','pixels',...
                        'position',[320 80 120 25],...
                        'callback',{@closegui,S},...
                        'string','Exit');


S.plotea = uicontrol('style','radio',...
                     'units','pixels',...
                     'position',[362 290 80 14],...
                     'string','Euler Angles',...
                     'fontsize',8,...
                     'value',1);



S.plotep = uicontrol('style','radio',...
                     'units','pixels',...
                     'position',[362 268 80 14],...
                     'string','Euler Vector',...
                     'fontsize',8,...
                     'value',1);

set(S.plotea,'callback',{@replot,S});
set(S.plotep,'callback',{@replot,S});

S.pbea = uicontrol('style','pushbutton',...
                   'units','pixels',...
                   'position',get(S.eamenu,'position')+[90 -2 -20 0],...
                   'callback',{@spincalcea,S},...
                   'String','Convert');


S.pbq = uicontrol('style','pushbutton',...
                   'units','pixels',...
                   'position',get(S.q4,'position')+[90 0 -20 0],...
                   'callback',{@spincalcq,S},...
                   'String','Convert');

S.pbnormq = uicontrol('style','pushbutton',...
                      'units','pixels',...
                      'position',get(S.pbq,'position')+[0 20 0 0],...
                      'callback',{@normq,S},...
                      'String','Normalize');

S.pbep = uicontrol('style','pushbutton',...
                   'units','pixels',...
                   'position',get(S.ep4,'position')+[90 0 -20 0],...
                   'callback',{@spincalcep,S},...
                   'String','Convert');

S.pbnormep = uicontrol('style','pushbutton',...
                       'units','pixels',...
                       'position',get(S.pbep,'position')+[0 20 0 0],...
                       'callback',{@normep,S},...
                       'String','Normalize');

S.pbdcm = uicontrol('style','pushbutton',...
                    'units','pixels',...
                    'position',get(S.dcm33,'position')+[90 0 -20 0],...
                    'callback',{@spincalcdcm,S},...
                    'String','Convert');

S.pbtransdcm = uicontrol('style','pushbutton',...
                         'units','pixels',...
                         'position',get(S.pbdcm,'position')+[0 20 0 0],...
                         'callback',{@transdcm,S},...
                         'String','Transpose');


function []=replot(varargin)
    S=varargin{3};
    S=plotter(S);


function S=plotter(S)
    axes(S.axes1);
    currentview=get(S.axes1,'view');
    plot3([0;1],[0;0],[0;0],':r')
    hold on
    axis equal
    text(1.05,0,-0.01,'X_{A}','color','r')
    plot3([0;0],[0;1],[0;0],':g')
    text(0,1.05,-0.01,'Y_{A}','color','g')
    plot3([0;0],[0;0],[0;1],':b')
    text(-0.01,-0.01,1.05,'Z_{A}','color','b')
    xlim([-1.0,1.0])
    ylim([-1.0,1.0])
    zlim([-1.0,1.0])
    set(gca,'xtick',[],'ytick',[],'ztick',[],'box','off')
    xlim([-1.0,1.0])
    ylim([-1.0,1.0])
    zlim([-1.0,1.0])
    dcm11=get(S.dcm11,'value');
    dcm12=get(S.dcm12,'value');
    dcm13=get(S.dcm13,'value');
    dcm21=get(S.dcm21,'value');
    dcm22=get(S.dcm22,'value');
    dcm23=get(S.dcm23,'value');
    dcm31=get(S.dcm31,'value');
    dcm32=get(S.dcm32,'value');
    dcm33=get(S.dcm33,'value');
    S.XB=plot3([0;dcm11],[0;dcm12],[0;dcm13],'-r');
    S.XBtext=text(dcm11*1.05,dcm12*1.05,dcm13*1.05,'X_{B}','color','r');
    S.YB=plot3([0;dcm21],[0;dcm22],[0;dcm23],'-g');
    S.YBtext=text(dcm21*1.05,dcm22*1.05,dcm23*1.05,'Y_{B}','color','g');
    S.ZB=plot3([0;dcm31],[0;dcm32],[0;dcm33],'-b');
    S.ZBtext=text(dcm31*1.05,dcm32*1.05,dcm33*1.05,'Z_{B}','color','b');
    if get(S.plotea,'value')==1
        ea1=get(S.ea1,'value');
        ea2=get(S.ea2,'value');
        ea3=get(S.ea3,'value');
        EA_rotation_order_index=get(S.eamenu,'value');
        %EA_rotations={'XYZ';'XZY';'YXZ';'YZX';'ZXY';'ZYX';'XYX';'XZX';'YXY';'YZY';'ZXZ';'ZYZ'};
        EA_rotations={'EA123';'EA132';'EA213';'EA231';'EA312';'EA321';'EA121';'EA131';'EA212';'EA232';'EA313';'EA323'};
        EA_rotation_order=EA_rotations{EA_rotation_order_index,:};

        n1=max([ceil(abs(ea1)/0.5),19]);
        n2=max([ceil(abs(ea2)/0.5),19]);
        n3=max([ceil(abs(ea3)/0.5),19]);
        EA_sweep1=[linspace(0,ea1,n1)',zeros(n1,1),zeros(n1,1)];
        EA_sweep2=[ea1*ones(n2,1),linspace(0,ea2,n2)',zeros(n2,1)];
        EA_sweep3=[ea1*ones(n3,1),ea2*ones(n3,1),linspace(0,ea3,n3)'];
        full_sweep=[EA_sweep1;EA_sweep2;EA_sweep3];
        if EA_rotation_order_index<=6
            [DCM_sweep,errorstring]=spincalcmod([EA_rotation_order,'toDCM'],full_sweep,eps,0);
        else
            EA_rotation_order_mod=EA_rotations{EA_rotation_order_index-6,:};
            [DCM_sweep1,errorstring]=spincalcmod([EA_rotation_order_mod,'toDCM'],full_sweep(1:(n1+1),1:3),eps,0);
            [DCM_sweep2,errorstring]=spincalcmod([EA_rotation_order,'toDCM'],full_sweep((n1+2):end,1:3),eps,0);
            DCM_sweep=cat(3,DCM_sweep1,DCM_sweep2);
        end
        sweep_vector1a=NaN(n1,3);
        sweep_vector1b=NaN(n1,3);
        sweep_vector2a=NaN(n2,3);
        sweep_vector2b=NaN(n2,3);
        sweep_vector3a=NaN(n3,3);
        sweep_vector3b=NaN(n3,3);
        initvec1a=[0,0,0];
        initvec1b=[0,0,0];
        initvec2a=[0,0,0];
        initvec2b=[0,0,0];
        initvec3a=[0,0,0];
        initvec3b=[0,0,0];
        colors={'r';'g';'b'};
        angle1_color=colors{str2double(EA_rotations{EA_rotation_order_index}(3))};
        angle2_color=colors{str2double(EA_rotations{EA_rotation_order_index}(4))};
        angle3_color=colors{str2double(EA_rotations{EA_rotation_order_index}(5))};
        if EA_rotation_order_index<=6
            eval(['initvec1a(1,',EA_rotations{EA_rotation_order_index}(4),')=0.8;colorsweep1a=colors{',EA_rotations{EA_rotation_order_index}(4),'};'])
            eval(['initvec1b(1,',EA_rotations{EA_rotation_order_index}(5),')=0.5;colorsweep1b=colors{',EA_rotations{EA_rotation_order_index}(5),'};'])
            eval(['initvec2a(1,',EA_rotations{EA_rotation_order_index}(3),')=0.65;colorsweep2a=colors{',EA_rotations{EA_rotation_order_index}(3),'};'])
            eval(['initvec2b(1,',EA_rotations{EA_rotation_order_index}(5),')=0.5;'])
            eval(['initvec3a(1,',EA_rotations{EA_rotation_order_index}(3),')=0.65;'])
            eval(['initvec3b(1,',EA_rotations{EA_rotation_order_index}(4),')=0.8;'])
        else
            if str2double(EA_rotations{EA_rotation_order_index}(3))==1
                if str2double(EA_rotations{EA_rotation_order_index}(4))==2
                    other_axis=3; %#ok<*NASGU>
                else
                    other_axis=2;
                end
            elseif str2double(EA_rotations{EA_rotation_order_index}(3))==2
                if str2double(EA_rotations{EA_rotation_order_index}(4))==1
                    other_axis=3;
                else
                    other_axis=1;
                end
            elseif str2double(EA_rotations{EA_rotation_order_index}(3))==3
                if str2double(EA_rotations{EA_rotation_order_index}(4))==1
                    other_axis=2;
                else
                    other_axis=1;
                end
            end
            eval(['initvec1a(1,',EA_rotations{EA_rotation_order_index}(4),')=0.8;colorsweep1a=colors{',EA_rotations{EA_rotation_order_index}(4),'};'])
            eval('initvec1b(1,other_axis)=0.5;colorsweep1b=colors{other_axis};')
            eval(['initvec2a(1,',EA_rotations{EA_rotation_order_index}(3),')=0.65;colorsweep2a=colors{',EA_rotations{EA_rotation_order_index}(3),'};'])
            eval('initvec2b(1,other_axis)=0.5;')
            eval('initvec3a(1,other_axis)=0.5;')
            eval(['initvec3b(1,',EA_rotations{EA_rotation_order_index}(4),')=0.8;'])
        end
        for ii=1:n1
            sweep_vector1a(ii,1:3)=initvec1a*DCM_sweep(1:3,1:3,ii);
            sweep_vector1b(ii,1:3)=initvec1b*DCM_sweep(1:3,1:3,ii);
        end
        for ii=1:n2
            sweep_vector2a(ii,1:3)=initvec2a*DCM_sweep(1:3,1:3,n1+ii);
            sweep_vector2b(ii,1:3)=initvec2b*DCM_sweep(1:3,1:3,n1+ii);
        end
        for ii=1:n3
            sweep_vector3a(ii,1:3)=initvec3a*DCM_sweep(1:3,1:3,n1+n2+ii);
            sweep_vector3b(ii,1:3)=initvec3b*DCM_sweep(1:3,1:3,n1+n2+ii);
        end
        if abs(ea1)<1e-10
            sweep_vector1a(:,:)=NaN;
            sweep_vector1b(:,:)=NaN;
        end
        if abs(ea2)<1e-10
            sweep_vector2a(:,:)=NaN;
            sweep_vector2b(:,:)=NaN;
        end
        if abs(ea3)<1e-10
            sweep_vector3a(:,:)=NaN;
            sweep_vector3b(:,:)=NaN;
        end
        %Determine coordinates of arrowtips
        for ii=1:6
            if ii==1,
                temp=sweep_vector1a;
            elseif ii==2
                temp=sweep_vector1b;
            elseif ii==3
                temp=sweep_vector2a;
            elseif ii==4
                temp=sweep_vector2b;
            elseif ii==5
                temp=sweep_vector3a;
            elseif ii==6
                temp=sweep_vector3b;
            end
            Pend=temp(end,1:3);
            Pstart=temp(end-18,1:3);
            Pmid=temp(end-9,1:3);
            Pnear=(Pstart+Pend)/2;
            Pa=8*(Pnear-Pmid)+Pmid;
            Pb=-8*(Pnear-Pmid)+Pmid;
            x=[Pend(1);Pa(1);Pb(1);Pend(1)];
            y=[Pend(2);Pa(2);Pb(2);Pend(2)];
            z=[Pend(3);Pa(3);Pb(3);Pend(3)];
            S.arrowheads=plot3(x,y,z,'-k');
        end
        S.start1a=plot3(sweep_vector1a(1,1),sweep_vector1a(1,2),sweep_vector1a(1,3),'ok','markersize',5);
        S.sweep1a=plot3(sweep_vector1a(1:end-9,1),sweep_vector1a(1:end-9,2),sweep_vector1a(1:end-9,3),'-k');
        S.sweep1atick=plot3([sweep_vector1a(ceil(n1/2),1);sweep_vector1a(ceil(n1/2),1)/0.92],[sweep_vector1a(ceil(n1/2),2);sweep_vector1a(ceil(n1/2),2)/0.92],[sweep_vector1a(ceil(n1/2),3);sweep_vector1a(ceil(n1/2),3)/0.92],'-k');
        S.psi_a=text(sweep_vector1a(ceil(n1/2),1)/0.9,sweep_vector1a(ceil(n1/2),2)/0.9,sweep_vector1a(ceil(n1/2),3)/0.9,'\psi','fontsize',11,'color',angle1_color);
        S.axis1a=plot3([0,sweep_vector1a(end,1)],[0,sweep_vector1a(end,2)],[0,sweep_vector1a(end,3)],[':',colorsweep1a]);

        S.start1b=plot3(sweep_vector1b(1,1),sweep_vector1b(1,2),sweep_vector1b(1,3),'ok','markersize',5);
        S.sweep1b=plot3(sweep_vector1b(1:end-9,1),sweep_vector1b(1:end-9,2),sweep_vector1b(1:end-9,3),'-k');
        S.sweep1btick=plot3([sweep_vector1b(ceil(n1/2),1);sweep_vector1b(ceil(n1/2),1)/0.92],[sweep_vector1b(ceil(n1/2),2);sweep_vector1b(ceil(n1/2),2)/0.92],[sweep_vector1b(ceil(n1/2),3);sweep_vector1b(ceil(n1/2),3)/0.92],'-k');
        S.psi_b=text(sweep_vector1b(ceil(n1/2),1)/0.9,sweep_vector1b(ceil(n1/2),2)/0.9,sweep_vector1b(ceil(n1/2),3)/0.9,'\psi','fontsize',11,'color',angle1_color);
        S.axis1b=plot3([0,sweep_vector1b(end,1)],[0,sweep_vector1b(end,2)],[0,sweep_vector1b(end,3)],[':',colorsweep1b]);

        S.start2a=plot3(sweep_vector2a(1,1),sweep_vector2a(1,2),sweep_vector2a(1,3),'ok','markersize',5);
        S.sweep2a=plot3(sweep_vector2a(1:end-9,1),sweep_vector2a(1:end-9,2),sweep_vector2a(1:end-9,3),'-k');
        S.sweep2atick=plot3([sweep_vector2a(ceil(n2/2),1);sweep_vector2a(ceil(n2/2),1)/0.92],[sweep_vector2a(ceil(n2/2),2);sweep_vector2a(ceil(n2/2),2)/0.92],[sweep_vector2a(ceil(n2/2),3);sweep_vector2a(ceil(n2/2),3)/0.92],'-k');
        S.theta_a=text(sweep_vector2a(ceil(n2/2),1)/0.9,sweep_vector2a(ceil(n2/2),2)/0.9,sweep_vector2a(ceil(n2/2),3)/0.9,'\theta','fontsize',11,'color',angle2_color);
        S.axis2a=plot3([0,sweep_vector2a(end,1)],[0,sweep_vector2a(end,2)],[0,sweep_vector2a(end,3)],[':',colorsweep2a]);

        S.start2b=plot3(sweep_vector2b(1,1),sweep_vector2b(1,2),sweep_vector2b(1,3),'ok','markersize',5);
        S.sweep2b=plot3(sweep_vector2b(1:end-9,1),sweep_vector2b(1:end-9,2),sweep_vector2b(1:end-9,3),'-k');
        S.sweep2btick=plot3([sweep_vector2b(ceil(n2/2),1);sweep_vector2b(ceil(n2/2),1)/0.92],[sweep_vector2b(ceil(n2/2),2);sweep_vector2b(ceil(n2/2),2)/0.92],[sweep_vector2b(ceil(n2/2),3);sweep_vector2b(ceil(n2/2),3)/0.92],'-k');
        S.theta_b=text(sweep_vector2b(ceil(n2/2),1)/0.9,sweep_vector2b(ceil(n2/2),2)/0.9,sweep_vector2b(ceil(n2/2),3)/0.9,'\theta','fontsize',11,'color',angle2_color);

        S.start3a=plot3(sweep_vector3a(1,1),sweep_vector3a(1,2),sweep_vector3a(1,3),'ok','markersize',5);
        S.sweep3a=plot3(sweep_vector3a(1:end-9,1),sweep_vector3a(1:end-9,2),sweep_vector3a(1:end-9,3),'-k');
        S.sweep3atick=plot3([sweep_vector3a(ceil(n3/2),1);sweep_vector3a(ceil(n3/2),1)/0.92],[sweep_vector3a(ceil(n3/2),2);sweep_vector3a(ceil(n3/2),2)/0.92],[sweep_vector3a(ceil(n3/2),3);sweep_vector3a(ceil(n3/2),3)/0.92],'-k');
        S.phi_a=text(sweep_vector3a(ceil(n3/2),1)/0.9,sweep_vector3a(ceil(n3/2),2)/0.9,sweep_vector3a(ceil(n3/2),3)/0.9,'\phi','fontsize',11,'color',angle3_color);
        if EA_rotation_order_index>6
            S.axis3a=plot3([0,sweep_vector2b(end,1)],[0,sweep_vector2b(end,2)],[0,sweep_vector2b(end,3)],[':',colorsweep1b]);
        end

        S.start3b=plot3(sweep_vector3b(1,1),sweep_vector3b(1,2),sweep_vector3b(1,3),'ok','markersize',5);
        S.sweep3b=plot3(sweep_vector3b(1:end-9,1),sweep_vector3b(1:end-9,2),sweep_vector3b(1:end-9,3),'-k');
        S.sweep3btick=plot3([sweep_vector3b(ceil(n3/2),1);sweep_vector3b(ceil(n3/2),1)/0.92],[sweep_vector3b(ceil(n3/2),2);sweep_vector3b(ceil(n3/2),2)/0.92],[sweep_vector3b(ceil(n3/2),3);sweep_vector3b(ceil(n3/2),3)/0.92],'-k');
        S.phi_b=text(sweep_vector3b(ceil(n3/2),1)/0.9,sweep_vector3b(ceil(n3/2),2)/0.9,sweep_vector3b(ceil(n3/2),3)/0.9,'\phi','fontsize',11,'color',angle3_color);
    end
    if get(S.plotep,'value')==1
        m1=get(S.ep1,'value');
        m2=get(S.ep2,'value');
        m3=get(S.ep3,'value');
        plot3([0 m1],[0 m2],[0 m3],'-m');
        text(m1*1.02,m2*1.02,m3*1.02,'m','color','m');
    end
    hold off
    set(gca,'xtick',[],'ytick',[],'ztick',[],'box','off')
    view(currentview)


function []=normq(varargin)
    S=varargin{3};
    q1=get(S.q1,'value');
	q2=get(S.q2,'value');
	q3=get(S.q3,'value');
	q4=get(S.q4,'value');
    qnorm=norm([q1,q2,q3,q4]);
    set(S.q1,'string',num2str(q1/qnorm,'%8.6f'),'value',q1/qnorm);
    set(S.q2,'string',num2str(q2/qnorm,'%8.6f'),'value',q2/qnorm);
    set(S.q3,'string',num2str(q3/qnorm,'%8.6f'),'value',q3/qnorm);
    set(S.q4,'string',num2str(q4/qnorm,'%8.6f'),'value',q4/qnorm);


function []=normep(varargin)
    S=varargin{3};
    m1=get(S.ep1,'value');
	m2=get(S.ep2,'value');
	m3=get(S.ep3,'value');
    mnorm=norm([m1,m2,m3]);
    set(S.ep1,'string',num2str(m1/mnorm,'%8.6f'),'value',m1/mnorm);
    set(S.ep2,'string',num2str(m2/mnorm,'%8.6f'),'value',m2/mnorm);
    set(S.ep3,'string',num2str(m3/mnorm,'%8.6f'),'value',m3/mnorm);


function []=transdcm(varargin)
    S=varargin{3};
	dcm12=get(S.dcm12,'value');
	dcm13=get(S.dcm13,'value');
    dcm21=get(S.dcm21,'value');
	dcm23=get(S.dcm23,'value');
    dcm31=get(S.dcm31,'value');
	dcm32=get(S.dcm32,'value');
    set(S.dcm12,'string',num2str(dcm21,'%8.6f'),'value',dcm21);
    set(S.dcm13,'string',num2str(dcm31,'%8.6f'),'value',dcm31);
    set(S.dcm21,'string',num2str(dcm12,'%8.6f'),'value',dcm12);
    set(S.dcm23,'string',num2str(dcm32,'%8.6f'),'value',dcm32);
    set(S.dcm31,'string',num2str(dcm13,'%8.6f'),'value',dcm13);
    set(S.dcm32,'string',num2str(dcm23,'%8.6f'),'value',dcm23);


function []=spincalcea(varargin)
	S=varargin{3};
	ea1=get(S.ea1,'value');
	ea2=get(S.ea2,'value');
	ea3=get(S.ea3,'value');
	EA_rotation_order_index=get(S.eamenu,'value');
	EA_rotations={'EA123';'EA132';'EA213';'EA231';'EA312';'EA321';'EA121';'EA131';'EA212';'EA232';'EA313';'EA323'};
	EA_rotation_order=EA_rotations{EA_rotation_order_index,:};
	[DCM,errorstring]=spincalcmod([EA_rotation_order,'toDCM'],[ea1,ea2,ea3],eps,1);
	[EV,errorstring]=spincalcmod([EA_rotation_order,'toEV'],[ea1,ea2,ea3],eps,1);
	[Q,errorstring]=spincalcmod([EA_rotation_order,'toQ'],[ea1,ea2,ea3],eps,1);
	set(S.dcm11,'string',num2str(DCM(1,1),'%8.6f'),'value',DCM(1,1));
	set(S.dcm12,'string',num2str(DCM(1,2),'%8.6f'),'value',DCM(1,2));
	set(S.dcm13,'string',num2str(DCM(1,3),'%8.6f'),'value',DCM(1,3));
	set(S.dcm21,'string',num2str(DCM(2,1),'%8.6f'),'value',DCM(2,1));
	set(S.dcm22,'string',num2str(DCM(2,2),'%8.6f'),'value',DCM(2,2));
	set(S.dcm23,'string',num2str(DCM(2,3),'%8.6f'),'value',DCM(2,3));
	set(S.dcm31,'string',num2str(DCM(3,1),'%8.6f'),'value',DCM(3,1));
	set(S.dcm32,'string',num2str(DCM(3,2),'%8.6f'),'value',DCM(3,2));
	set(S.dcm33,'string',num2str(DCM(3,3),'%8.6f'),'value',DCM(3,3));
	set(S.q1,'string',num2str(Q(1),'%8.6f'),'value',Q(1));
	set(S.q2,'string',num2str(Q(2),'%8.6f'),'value',Q(2));
	set(S.q3,'string',num2str(Q(3),'%8.6f'),'value',Q(3));
	set(S.q4,'string',num2str(Q(4),'%8.6f'),'value',Q(4));
	set(S.ep1,'string',num2str(EV(1),'%8.6f'),'value',EV(1));
	set(S.ep2,'string',num2str(EV(2),'%8.6f'),'value',EV(2));
	set(S.ep3,'string',num2str(EV(3),'%8.6f'),'value',EV(3));
	set(S.ep4,'string',num2str(EV(4),'%8.6f'),'value',EV(4));
	set(S.dialog,'string',errorstring);
    if ~strcmpi(errorstring(1,1:5),'error')
        S=plotter(S);
    end


function []=spincalcdcm(varargin)
    S=varargin{3};
	dcm11=get(S.dcm11,'value');
	dcm12=get(S.dcm12,'value');
	dcm13=get(S.dcm13,'value');
    dcm21=get(S.dcm21,'value');
	dcm22=get(S.dcm22,'value');
	dcm23=get(S.dcm23,'value');
    dcm31=get(S.dcm31,'value');
	dcm32=get(S.dcm32,'value');
	dcm33=get(S.dcm33,'value');
    dcm=[dcm11,dcm12,dcm13;dcm21,dcm22,dcm23;dcm31,dcm32,dcm33];
	EA_rotation_order_index=get(S.eamenu,'value');
	EA_rotations={'EA123';'EA132';'EA213';'EA231';'EA312';'EA321';'EA121';'EA131';'EA212';'EA232';'EA313';'EA323'};
	EA_rotation_order=EA_rotations{EA_rotation_order_index,:};
	[EA,errorstring]=spincalcmod(['DCMto',EA_rotation_order],dcm,1e-5,1);
	[EV,errorstring]=spincalcmod('DCMtoEV',dcm,1e-10,1);
	[Q,errorstring]=spincalcmod('DCMtoQ',dcm,1e-10,1);
    set(S.q1,'string',num2str(Q(1),'%8.6f'),'value',Q(1));
	set(S.q2,'string',num2str(Q(2),'%8.6f'),'value',Q(2));
	set(S.q3,'string',num2str(Q(3),'%8.6f'),'value',Q(3));
	set(S.q4,'string',num2str(Q(4),'%8.6f'),'value',Q(4));
	set(S.ep1,'string',num2str(EV(1),'%8.6f'),'value',EV(1));
	set(S.ep2,'string',num2str(EV(2),'%8.6f'),'value',EV(2));
	set(S.ep3,'string',num2str(EV(3),'%8.6f'),'value',EV(3));
	set(S.ep4,'string',num2str(EV(4),'%8.6f'),'value',EV(4));
    set(S.ea1,'string',num2str(EA(1),'%8.6f'),'value',EA(1));
	set(S.ea2,'string',num2str(EA(2),'%8.6f'),'value',EA(2));
	set(S.ea3,'string',num2str(EA(3),'%8.6f'),'value',EA(3));
	set(S.dialog,'string',errorstring);
    if ~strcmpi(errorstring(1,1:5),'error')
        S=plotter(S);
    end


function []=spincalcq(varargin)
	S=varargin{3};
	q1=get(S.q1,'value');
	q2=get(S.q2,'value');
	q3=get(S.q3,'value');
	q4=get(S.q4,'value');
	EA_rotation_order_index=get(S.eamenu,'value');
	EA_rotations={'EA123';'EA132';'EA213';'EA231';'EA312';'EA321';'EA121';'EA131';'EA212';'EA232';'EA313';'EA323'};
	EA_rotation_order=EA_rotations{EA_rotation_order_index,:};
	[DCM,errorstring]=spincalcmod('QtoDCM',[q1,q2,q3,q4],eps,1);
	[EV,errorstring]=spincalcmod('QtoEV',[q1,q2,q3,q4],eps,1);
	[EA,errorstring]=spincalcmod(['Qto',EA_rotation_order],[q1,q2,q3,q4],eps,1);
	set(S.dcm11,'string',num2str(DCM(1,1),'%8.6f'),'value',DCM(1,1));
	set(S.dcm12,'string',num2str(DCM(1,2),'%8.6f'),'value',DCM(1,2));
	set(S.dcm13,'string',num2str(DCM(1,3),'%8.6f'),'value',DCM(1,3));
	set(S.dcm21,'string',num2str(DCM(2,1),'%8.6f'),'value',DCM(2,1));
	set(S.dcm22,'string',num2str(DCM(2,2),'%8.6f'),'value',DCM(2,2));
	set(S.dcm23,'string',num2str(DCM(2,3),'%8.6f'),'value',DCM(2,3));
	set(S.dcm31,'string',num2str(DCM(3,1),'%8.6f'),'value',DCM(3,1));
	set(S.dcm32,'string',num2str(DCM(3,2),'%8.6f'),'value',DCM(3,2));
	set(S.dcm33,'string',num2str(DCM(3,3),'%8.6f'),'value',DCM(3,3));
	set(S.ep1,'string',num2str(EV(1),'%8.6f'),'value',EV(1));
	set(S.ep2,'string',num2str(EV(2),'%8.6f'),'value',EV(2));
	set(S.ep3,'string',num2str(EV(3),'%8.6f'),'value',EV(3));
	set(S.ep4,'string',num2str(EV(4),'%8.6f'),'value',EV(4));
	set(S.ea1,'string',num2str(EA(1),'%8.6f'),'value',EA(1));
	set(S.ea2,'string',num2str(EA(2),'%8.6f'),'value',EA(2));
	set(S.ea3,'string',num2str(EA(3),'%8.6f'),'value',EA(3));
	set(S.dialog,'string',errorstring);
    if ~strcmpi(errorstring(1,1:5),'error')
        S=plotter(S);
    end


function []=spincalcep(varargin)
	S=varargin{3};
	m1=get(S.ep1,'value');
	m2=get(S.ep2,'value');
	m3=get(S.ep3,'value');
	mu=get(S.ep4,'value');
	EA_rotation_order_index=get(S.eamenu,'value');
	EA_rotations={'EA123';'EA132';'EA213';'EA231';'EA312';'EA321';'EA121';'EA131';'EA212';'EA232';'EA313';'EA323'};
	EA_rotation_order=EA_rotations{EA_rotation_order_index,:};
	[DCM,errorstring]=spincalcmod('EVtoDCM',[m1,m2,m3,mu],eps,1);
	[Q,errorstring]=spincalcmod('EVtoQ',[m1,m2,m3,mu],eps,1);
	[EA,errorstring]=spincalcmod(['EVto',EA_rotation_order],[m1,m2,m3,mu],eps,1);
	set(S.dcm11,'string',num2str(DCM(1,1),'%8.6f'),'value',DCM(1,1));
	set(S.dcm12,'string',num2str(DCM(1,2),'%8.6f'),'value',DCM(1,2));
	set(S.dcm13,'string',num2str(DCM(1,3),'%8.6f'),'value',DCM(1,3));
	set(S.dcm21,'string',num2str(DCM(2,1),'%8.6f'),'value',DCM(2,1));
	set(S.dcm22,'string',num2str(DCM(2,2),'%8.6f'),'value',DCM(2,2));
	set(S.dcm23,'string',num2str(DCM(2,3),'%8.6f'),'value',DCM(2,3));
	set(S.dcm31,'string',num2str(DCM(3,1),'%8.6f'),'value',DCM(3,1));
	set(S.dcm32,'string',num2str(DCM(3,2),'%8.6f'),'value',DCM(3,2));
	set(S.dcm33,'string',num2str(DCM(3,3),'%8.6f'),'value',DCM(3,3));
	set(S.q1,'string',num2str(Q(1),'%8.6f'),'value',Q(1));
	set(S.q2,'string',num2str(Q(2),'%8.6f'),'value',Q(2));
	set(S.q3,'string',num2str(Q(3),'%8.6f'),'value',Q(3));
	set(S.q4,'string',num2str(Q(4),'%8.6f'),'value',Q(4));
	set(S.ea1,'string',num2str(EA(1),'%8.6f'),'value',EA(1));
	set(S.ea2,'string',num2str(EA(2),'%8.6f'),'value',EA(2));
	set(S.ea3,'string',num2str(EA(3),'%8.6f'),'value',EA(3));
	set(S.dialog,'string',errorstring);
    if ~strcmpi(errorstring(1,1:5),'error')
        S=plotter(S);
    end


function [OUTPUT,errorstring]=spincalcmod(CONVERSION,INPUT,tol,ichk)
%Function for the conversion of one rotation input type to desired output.
%Supported conversion input/output types are as follows:
%   1: Q        Rotation Quaternions
%   2: EV       Euler Vector and rotation angle (degrees)
%   3: DCM      Orthogonal DCM Rotation Matrix
%   4: EA###    Euler angles (12 possible sets) (degrees)
%
%Author: John Fuller
%National Institute of Aerospace
%Hampton, VA 23666
%John.Fuller@nianet.org
%
%Version 1.3
%June 30th, 2009
%
%Version 1.3 updates
%   SpinCalc now detects when input data is too close to Euler singularity, if user is choosing
%   Euler angle output. Prohibits output if middle angle is within 0.1 degree of singularity value.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%                OUTPUT=spincalcmod(CONVERSION,INPUT,tol,ichk)
%Inputs:
%CONVERSION - Single string value that dictates the type of desired
%             conversion.  The conversion strings are listed below.
%
%   'DCMtoEA###'  'DCMtoEV'    'DCMtoQ'       **for cases that involve
%   'EA###toDCM'  'EA###toEV'  'EA###toQ'       euler angles, ### should be
%   'EVtoDCM'     'EVtoEA###'  'EVtoQ'          replaced with the proper
%   'QtoDCM'      'QtoEA###'   'QtoEV'          order desired.  EA321 would
%   'EA###toEA###'                              be Z(yaw)-Y(pitch)-X(roll).
%
%INPUT - matrix or vector that corresponds to the first entry in the
%        CONVERSION string, formatted as follows:
%
%        DCM - 3x3xN multidimensional matrix which pre-multiplies by a coordinate
%              frame vector to rotate it to the desired new frame.
%
%        EA### - [psi,theta,phi] (Nx3) row vector list dictating to the first angle
%                rotation (psi), the second (theta), and third (phi) (DEGREES)
%
%        EV - [m1,m2,m3,MU] (Nx4) row vector list dictating the components of euler
%             rotation vector (original coordinate frame) and the Euler
%             rotation angle about that vector (MU) (DEGREES)
%
%        Q - [q1,q2,q3,q4] (Nx4) row vector list defining quaternion of
%            rotation.  q4 = cos(MU/2) where MU is Euler rotation angle
%
%tol - tolerance value
%ichk - 0 disables warning flags
%          1 enables warning flags (near singularities)
%**NOTE: N corresponds to multiple orientations
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Output:
%OUTPUT - matrix or vector corresponding to the second entry in the
%         CONVERSION input string, formatted as shown above.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%Pre-processer to determine type of conversion from CONVERSION string input
%Types are numbered as follows:
%Q=1   EV=2   DCM=3   EA=4
i_type=strfind(lower(CONVERSION),'to');
length=size(CONVERSION,2);
error_flag=0;
errorstring='No errors found.';
if length>12 || length<4,   %no CONVERSION string can be shorter than 4 or longer than 12 chars
    error('Error: Invalid entry for CONVERSION input string');
end
o_type=length-i_type;
if i_type<5,
    i_type=i_type-1;
else
    i_type=i_type-2;
end
if o_type<5,
    o_type=o_type-1;
else
    o_type=o_type-2;
end
TYPES=cell(1,4);
TYPES{1,1}='Q'; TYPES{1,2}='EV'; TYPES{1,3}='DCM'; TYPES{1,4}='EA';
INPUT_TYPE=TYPES{1,i_type};
OUTPUT_TYPE=TYPES{1,o_type};
clear TYPES
%Confirm input as compared to program interpretation
if i_type~=4 && o_type~=4,  %if input/output are NOT Euler angles
    CC=[INPUT_TYPE,'to',OUTPUT_TYPE];
    if strcmpi(CONVERSION,CC)==0;
        error('Error: Invalid entry for CONVERSION input string');
    end
else
    if i_type==4,   %if input type is Euler angles, determine the order of rotations
        EULER_order_in=str2double(CONVERSION(1,3:5));
        rot_1_in=floor(EULER_order_in/100);     %first rotation
        rot_2_in=floor((EULER_order_in-rot_1_in*100)/10);   %second rotation
        rot_3_in=(EULER_order_in-rot_1_in*100-rot_2_in*10);   %third rotation
        if rot_1_in<1 || rot_2_in<1 || rot_3_in<1 || rot_1_in>3 || rot_2_in>3 || rot_3_in>3,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that all orders are between 1 and 3
        elseif rot_1_in==rot_2_in || rot_2_in==rot_3_in,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that no 2 consecutive orders are equal (invalid)
        end
        %check input dimensions to be 1x3x1
        if size(INPUT,2)~=3 || size(INPUT,3)~=1
            error('Error: Input euler angle data vector is not Nx3')
        end
        %identify singularities
        if rot_1_in==rot_3_in, %Type 2 rotation (first and third rotations about same axis)
            if INPUT(:,2)<=zeros(size(INPUT,1),1) | INPUT(:,2)>=180*ones(size(INPUT,1),1),  %#ok<OR2> %confirm second angle within range
                %error('Error: Second input Euler angle(s) outside 0 to 180 degree range')
				errorstring='Error: Second input Euler angle(s) outside 0 to 180 degree range';
				error_flag=1;
            elseif abs(INPUT(:,2))<2*ones(size(INPUT,1),1) | abs(INPUT(:,2))>178*ones(size(INPUT,1),1),  %#ok<OR2> %check for singularity
                if ichk==1,
                    %errordlg('Warning: Input Euler angle rotation(s) near a singularity.               Second angle near 0 or 180 degrees.')
					errorstring={'Warning: Input Euler angle rotation(s) near a singularity.';'Second angle near 0 or 180 degrees.'};
                end
            end
        else    %Type 1 rotation (all rotations about each of three axes)
            if abs(INPUT(:,2))>=90*ones(size(INPUT,1),1), %confirm second angle within range
                %error('Error: Second input Euler angle(s) outside -90 to 90 degree range')
				errorstring='Error: Second input Euler angle(s) outside -90 to 90 degree range';
				error_flag=1;
            elseif abs(INPUT(:,2))>88*ones(size(INPUT,1),1),  %check for singularity
                if ichk==1, %#ok<ALIGN>
                    %errordlg('Warning: Input Euler angle(s) rotation near a singularity.               Second angle near -90 or 90 degrees.')
					errorstring={'Warning: Input Euler angle(s) rotation near a singularity.';'Second angle near -90 or 90 degrees.'};
				end
            end
        end
    end
    if o_type==4,   %if output type is Euler angles, determine order of rotations
        EULER_order_out=str2double(CONVERSION(1,length-2:length));
        rot_1_out=floor(EULER_order_out/100);   %first rotation
        rot_2_out=floor((EULER_order_out-rot_1_out*100)/10);    %second rotation
        rot_3_out=(EULER_order_out-rot_1_out*100-rot_2_out*10); %third rotation
        if rot_1_out<1 || rot_2_out<1 || rot_3_out<1 || rot_1_out>3 || rot_2_out>3 || rot_3_out>3,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that all orders are between 1 and 3
        elseif rot_1_out==rot_2_out || rot_2_out==rot_3_out,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that no 2 consecutive orders are equal
        end
    end
    if i_type==4 && o_type~=4,  %if input are euler angles but not output
        CC=['EA',num2str(EULER_order_in),'to',OUTPUT_TYPE]; %construct program conversion string for checking against user input
    elseif o_type==4 && i_type~=4,  %if output are euler angles but not input
        CC=[INPUT_TYPE,'to','EA',num2str(EULER_order_out)]; %construct program conversion string for checking against user input
    elseif i_type==4 && o_type==4,  %if both input and output are euler angles
        CC=['EA',num2str(EULER_order_in),'to','EA',num2str(EULER_order_out)];   %construct program conversion string
    end
    if strcmpi(CONVERSION,CC)==0; %check program conversion string against user input to confirm the conversion command
        error('Error: Invalid entry for CONVERSION input string');
    end
end
clear i_type o_type CC

%From the input, determine the quaternions that uniquely describe the
%rotation prescribed by that input.  The output will be calculated in the
%second portion of the code from these quaternions.
switch INPUT_TYPE
    case 'DCM'
        if size(INPUT,1)~=3 || size(INPUT,2)~=3  %check DCM dimensions
            error('Error: DCM matrix is not 3x3xN');
        end
        N=size(INPUT,3);    %number of orientations
        %Check if matrix is indeed orthogonal
        perturbed=NaN(3,3,N);
        DCM_flag=0;
        for ii=1:N,
            perturbed(:,:,ii)=abs(INPUT(:,:,ii)*INPUT(:,:,ii)'-eye(3)); %perturbed array shows difference between DCM*DCM' and I
            if abs(det(INPUT(:,:,ii))-1)>tol, %if determinant is off by one more than tol, user is warned.
                if ichk==1,
                    DCM_flag=1;
                end
            end
            if abs(det(INPUT(:,:,ii))+1)<0.05, %if determinant is near -1, DCM is improper
                %error('Error: Input DCM(s) improper');
				errorstring='Error: Input DCM(s) improper.';
				error_flag=1;
				break
            end
            if DCM_flag==1,
                %errordlg('Warning: Input DCM matrix determinant(s) off from 1 by more than tolerance.')
				errorstring='Warning: Input DCM matrix determinant(s) off from 1 by more than tolerance.';
            end
        end
        DCM_flag=0;
        if ichk==1,
            for kk=1:N,
                for ii=1:3,
                    for jj=1:3,
                        if perturbed(ii,jj,kk)>tol,   %if any difference is larger than tol, user is warned.
                            DCM_flag=1;
                        end
                    end
                end
            end
            if DCM_flag==1,
                %fprintf('Warning: Input DCM(s) matrix not orthogonal to precision tolerance.')
				errorstring='Warning: Input DCM(s) matrix not orthogonal to precision tolerance.';
            end
        end
        clear perturbed DCM_flag
        Q=NaN(4,N);
        for ii=1:N,
            denom=NaN(4,1);
            denom(1)=0.5*sqrt(1+INPUT(1,1,ii)-INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(2)=0.5*sqrt(1-INPUT(1,1,ii)+INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(3)=0.5*sqrt(1-INPUT(1,1,ii)-INPUT(2,2,ii)+INPUT(3,3,ii));
            denom(4)=0.5*sqrt(1+INPUT(1,1,ii)+INPUT(2,2,ii)+INPUT(3,3,ii));
            %determine which Q equations maximize denominator
            switch find(denom==max(denom),1,'first')  %determines max value of qtests to put in denominator
                case 1
                    Q(1,ii)=denom(1);
                    Q(2,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(1,ii));
                    Q(3,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(1,ii));
                    Q(4,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(1,ii));
                case 2
                    Q(2,ii)=denom(2);
                    Q(1,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(2,ii));
                    Q(3,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(2,ii));
                    Q(4,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(2,ii));
                case 3
                    Q(3,ii)=denom(3);
                    Q(1,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(3,ii));
                    Q(2,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(3,ii));
                    Q(4,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(3,ii));
                case 4
                    Q(4,ii)=denom(4);
                    Q(1,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(4,ii));
                    Q(2,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(4,ii));
                    Q(3,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(4,ii));
            end
        end
        Q=Q';
        clear denom
    case 'EV'  %Euler Vector Input Type
        if size(INPUT,2)~=4 || size(INPUT,3)~=1   %check dimensions
            error('Error: Input euler vector and rotation data matrix is not Nx4')
        end
        N=size(INPUT,1);
        MU=INPUT(:,4)*pi/180;  %assign mu name for clarity
        if abs(sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2)-ones(N,1))>tol*ones(N,1),  %check that input m's constitute unit vector
            %error('Input euler vector(s) components do not constitute a unit vector')
			errorstring='Error: Input euler vector(s) components do not constitute a unit vector.';
			error_flag=1;
        end
        if MU<-2*pi*ones(N,1) || MU>2*pi*ones(N,1), %check if rotation about euler vector is between 0 and 360
            %error('Input euler rotation angle(s) not between -360 and 360 degrees')
			errorstring='Error: Input mu rotation angle(s) not between -360 and 360 degrees.';
			error_flag=1;
        end
        Q=[INPUT(:,1).*sin(MU/2),INPUT(:,2).*sin(MU/2),INPUT(:,3).*sin(MU/2),cos(MU/2)];   %quaternion
        clear m1 m2 m3 MU
    case 'EA'
        psi=INPUT(:,1)*pi/180;  theta=INPUT(:,2)*pi/180;  phi=INPUT(:,3)*pi/180;
        N=size(INPUT,1);    %number of orientations
        %Pre-calculate cosines and sines of the half-angles for conversion.
        c1=cos(psi./2); c2=cos(theta./2); c3=cos(phi./2);
        s1=sin(psi./2); s2=sin(theta./2); s3=sin(phi./2);
        c13=cos((psi+phi)./2);  s13=sin((psi+phi)./2);
        c1_3=cos((psi-phi)./2);  s1_3=sin((psi-phi)./2);
        c3_1=cos((phi-psi)./2);  s3_1=sin((phi-psi)./2);
        if EULER_order_in==121,
            Q=[c2.*s13,s2.*c1_3,s2.*s1_3,c2.*c13];
        elseif EULER_order_in==232,
            Q=[s2.*s1_3,c2.*s13,s2.*c1_3,c2.*c13];
        elseif EULER_order_in==313;
            Q=[s2.*c1_3,s2.*s1_3,c2.*s13,c2.*c13];
        elseif EULER_order_in==131,
            Q=[c2.*s13,s2.*s3_1,s2.*c3_1,c2.*c13];
        elseif EULER_order_in==212,
            Q=[s2.*c3_1,c2.*s13,s2.*s3_1,c2.*c13];
        elseif EULER_order_in==323,
            Q=[s2.*s3_1,s2.*c3_1,c2.*s13,c2.*c13];
        elseif EULER_order_in==123,
            Q=[s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==231,
            Q=[c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==312,
            Q=[c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==132,
            Q=[s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==213,
            Q=[c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==321,
            Q=[c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        else
            error('Error: Invalid input Euler angle order type (conversion string)');
        end
        clear c1 s1 c2 s2 c3 s3 c13 s13 c1_3 s1_3 c3_1 s3_1 psi theta phi
    case 'Q'
        if size(INPUT,2)~=4 || size(INPUT,3)~=1
            error('Error: Input quaternion matrix is not Nx4');
        end
        N=size(INPUT,1);    %number of orientations
        if ichk==1,
            if abs(sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2+INPUT(:,4).^2)-ones(N,1))>tol*ones(N,1)
                %errordlg('Warning: Input quaternion norm(s) deviate(s) from unity by more than tolerance')
				errorstring='Warning: Input quaternion norm(s) deviate(s) from unity by more than tolerance';
            end
        end
        Q=INPUT;
end
clear INPUT INPUT_TYPE EULER_order_in

%Normalize quaternions in case of deviation from unity.  User has already
%been warned of deviation.
Qnorms=sqrt(sum(Q.*Q,2));
Q=[Q(:,1)./Qnorms,Q(:,2)./Qnorms,Q(:,3)./Qnorms,Q(:,4)./Qnorms];

switch OUTPUT_TYPE
	case 'DCM'
		OUTPUT=NaN(3,3);
	case 'EV'
		OUTPUT=NaN(1,4);
	case 'Q'
		OUTPUT=NaN(1,4);
	case 'EA'
		OUTPUT=NaN(1,3);
end

if error_flag==0
	switch OUTPUT_TYPE
		case 'DCM'
			Q=reshape(Q',1,4,N);
			OUTPUT=[Q(1,1,:).^2-Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,1,:).*Q(1,2,:)+Q(1,3,:).*Q(1,4,:)),2*(Q(1,1,:).*Q(1,3,:)-Q(1,2,:).*Q(1,4,:));
					2*(Q(1,1,:).*Q(1,2,:)-Q(1,3,:).*Q(1,4,:)),-Q(1,1,:).^2+Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,2,:).*Q(1,3,:)+Q(1,1,:).*Q(1,4,:));
					2*(Q(1,1,:).*Q(1,3,:)+Q(1,2,:).*Q(1,4,:)),2*(Q(1,2,:).*Q(1,3,:)-Q(1,1,:).*Q(1,4,:)),-Q(1,1,:).^2-Q(1,2,:).^2+Q(1,3,:).^2+Q(1,4,:).^2];
		case 'EV'
			MU=2*atan2(sqrt(sum(Q(:,1:3).*Q(:,1:3),2)),Q(:,4));
			if sin(MU/2)~=zeros(N,1),
				OUTPUT=[Q(:,1)./sin(MU/2),Q(:,2)./sin(MU/2),Q(:,3)./sin(MU/2),MU*180/pi];
			else
				OUTPUT=NaN(N,4);
				for ii=1:N,
					if sin(MU(ii,1)/2)~=0,
						OUTPUT(ii,1:4)=[Q(ii,1)/sin(MU(ii,1)/2),Q(ii,2)/sin(MU(ii,1)/2),Q(ii,3)/sin(MU(ii,1)/2),MU(ii,1)*180/pi];
					else
						OUTPUT(ii,1:4)=[1,0,0,MU(ii,1)*180/pi];
					end
				end
			end
		case 'Q'
			OUTPUT=Q;
		case 'EA'
			if EULER_order_out==121,
				psi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
				theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
				phi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
		  Euler_type=2;
			elseif EULER_order_out==232;
				psi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
				theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
				phi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
		  Euler_type=2;
			elseif EULER_order_out==313;
				psi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
				theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
				phi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
		  Euler_type=2;
			elseif EULER_order_out==131;
				psi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
				theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
				phi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
		  Euler_type=2;
			elseif EULER_order_out==212;
				psi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
				theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
				phi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
		  Euler_type=2;
			elseif EULER_order_out==323;
				psi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
				theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
				phi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
		  Euler_type=2;
			elseif EULER_order_out==123;
				psi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
				theta=asin(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
				phi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
		  Euler_type=1;
			elseif EULER_order_out==231;
				psi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
				theta=asin(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
				phi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
		  Euler_type=1;
			elseif EULER_order_out==312;
				psi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
				theta=asin(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
				phi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,3).*Q(:,1)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
		  Euler_type=1;
			elseif EULER_order_out==132;
				psi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
				theta=asin(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
				phi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
		  Euler_type=1;
			elseif EULER_order_out==213;
				psi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
				theta=asin(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
				phi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
		  Euler_type=1;
			elseif EULER_order_out==321;
				psi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
				theta=asin(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
				phi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
		  Euler_type=1;
			else
				error('Error: Invalid output Euler angle order type (conversion string).');
			end
			if(isreal([psi,theta,phi]))==0,
				%error('Error: Unreal Euler output.  Input resides too close to singularity.  Please choose different output type.')
				errorstring={'Error: Unreal Euler Angle output.  Input resides too close to singularity.';'Please choose different output type.'};
				OUTPUT=[NaN,NaN,NaN];
			end
			OUTPUT=mod([psi,theta,phi]*180/pi,360);  %deg
			if Euler_type==1, %#ok<ALIGN>
				sing_chk=find(abs(theta)*180/pi>89.9);
				sing_chk=sort(sing_chk(sing_chk>0));
				if size(sing_chk,1)>=1,
					%error('Error: Input rotation #%s resides too close to Type 1 Euler singularity.\nType 1 Euler singularity occurs when second angle is -90 or 90 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
					errorstring={'Error: Input rotation resides too close to Type 1 Euler singularity.';'Type 1 Euler singularity occurs when second angle is -90 or 90 degrees.';'Please choose a different EA order.'};
					OUTPUT=[NaN,NaN,NaN];
				end
			elseif Euler_type==2,
				sing_chk=[find(abs(theta*180/pi)<0.1);find(abs(theta*180/pi-180)<0.1);find(abs(theta*180/pi-360))<0.1];
				sing_chk=sort(sing_chk(sing_chk>0));
				if size(sing_chk,1)>=1, %#ok<ALIGN>
					%error('Error: Input rotation #%s resides too close to Type 2 Euler singularity.\nType 2 Euler singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
					errorstring={'Error: Input rotation resides too close to Type 2 Euler singularity.';'Type 2 Euler singularity occurs when second angle is 0 or 180 degrees.';'Please choose a different EA order.'};
					OUTPUT=[NaN,NaN,NaN];
                end
            end
            %Modified output Euler angles to be between -180 and 180
            temp=OUTPUT(:,1);
            temp(temp>180)=temp(temp>180)-360;
            OUTPUT(:,1)=temp;
            temp=OUTPUT(:,2);
            temp(temp>180)=temp(temp>180)-360;
            OUTPUT(:,2)=temp;
            temp=OUTPUT(:,3);
            temp(temp>180)=temp(temp>180)-360;
            OUTPUT(:,3)=temp;
	end
end
OUTPUT(abs(OUTPUT)<1e-14)=0;


function []=closegui(varargin)
    S=varargin{3};
    close(S.fh)


function []=assignep1(varargin)
    S=varargin{3};
    set(S.ep1,'value',str2double(get(S.ep1,'string')));


function []=assignep2(varargin)
    S=varargin{3};
    set(S.ep2,'value',str2double(get(S.ep2,'string')));


function []=assignep3(varargin)
    S=varargin{3};
    set(S.ep3,'value',str2double(get(S.ep3,'string')));


function []=assignea1(varargin)
    S=varargin{3};
    set(S.ea1,'value',str2double(get(S.ea1,'string')));


function []=assignea2(varargin)
    S=varargin{3};
    set(S.ea2,'value',str2double(get(S.ea2,'string')));


function []=assignea3(varargin)
    S=varargin{3};
    set(S.ea3,'value',str2double(get(S.ea3,'string')));


function []=assignep4(varargin)
    S=varargin{3};
    set(S.ep4,'value',str2double(get(S.ep4,'string')));


function []=assignq1(varargin)
    S=varargin{3};
    set(S.q1,'value',str2double(get(S.q1,'string')));


function []=assignq2(varargin)
    S=varargin{3};
    set(S.q2,'value',str2double(get(S.q2,'string')));


function []=assignq3(varargin)
    S=varargin{3};
    set(S.q3,'value',str2double(get(S.q3,'string')));


function []=assignq4(varargin)
    S=varargin{3};
    set(S.q4,'value',str2double(get(S.q4,'string')));


function []=assigndcm11(varargin)
    S=varargin{3};
    set(S.dcm11,'value',str2double(get(S.dcm11,'string')));


function []=assigndcm12(varargin)
    S=varargin{3};
    set(S.dcm12,'value',str2double(get(S.dcm12,'string')));


function []=assigndcm13(varargin)
    S=varargin{3};
    set(S.dcm13,'value',str2double(get(S.dcm13,'string')));


function []=assigndcm21(varargin)
    S=varargin{3};
    set(S.dcm21,'value',str2double(get(S.dcm21,'string')));


function []=assigndcm22(varargin)
    S=varargin{3};
    set(S.dcm22,'value',str2double(get(S.dcm22,'string')));


function []=assigndcm23(varargin)
    S=varargin{3};
    set(S.dcm23,'value',str2double(get(S.dcm23,'string')));


function []=assigndcm31(varargin)
    S=varargin{3};
    set(S.dcm31,'value',str2double(get(S.dcm31,'string')));


function []=assigndcm32(varargin)
    S=varargin{3};
    set(S.dcm32,'value',str2double(get(S.dcm32,'string')));


function []=assigndcm33(varargin)
    S=varargin{3};
    set(S.dcm33,'value',str2double(get(S.dcm33,'string')));


function [hout,ax_out] = uibutton(varargin)
%uibutton: Create pushbutton with more flexible labeling than uicontrol.
% Usage:
%   uibutton accepts all the same arguments as uicontrol except for the
%   following property changes:
%
%     Property      Values
%     -----------   ------------------------------------------------------
%     Style         'pushbutton', 'togglebutton' or 'text', default =
%                   'pushbutton'.
%     String        Same as for text() including cell array of strings and
%                   TeX or LaTeX interpretation.
%     Interpreter   'tex', 'latex' or 'none', default = default for text()
%     Rotation      text rotation angle, default = 0
%
% Syntax:
%   handle = uibutton('PropertyName',PropertyValue,...)
%   handle = uibutton(parent,'PropertyName',PropertyValue,...)
%   [text_obj,axes_handle] = uibutton('Style','text',...
%       'PropertyName',PropertyValue,...)
%
% uibutton creates a temporary axes and text object containing the text to
% be displayed, captures the axes as an image, deletes the axes and then
% displays the image on the uicontrol.  The handle to the uicontrol is
% returned.  If you pass in a handle to an existing uicontol as the first
% argument then uibutton will use that uicontrol and not create a new one.
%
% If the Style is set to 'text' then the axes object is not deleted and the
% text object handle is returned (as well as the handle to the axes in a
% second output argument).
%
% See also UICONTROL.

% Version: 1.8, 10 March 2010
% Author:  Douglas M. Schwarz
% Email:   dmschwarz=ieee*org, dmschwarz=urgrad*rochester*edu
% Real_email = regexprep(Email,{'=','*'},{'@','.'})


% Detect if first argument is a uicontrol handle.
keep_handle = false;
if nargin > 0
	h = varargin{1};
	if isscalar(h) && ishandle(h) && strcmp(get(h,'Type'),'uicontrol')
		keep_handle = true;
		varargin(1) = [];
	end
end

% Parse arguments looking for 'Interpreter' property.  If found, note its
% value and then remove it from where it was found.
interp_value = get(0,'DefaultTextInterpreter');
rotation_value = get(0,'DefaultTextRotation');
arg = 1;
remove = [];
while arg <= length(varargin)
	v = varargin{arg};
	if isstruct(v)
		fn = fieldnames(v);
		for i = 1:length(fn)
			if strncmpi(fn{i},'interpreter',length(fn{i}))
				interp_value = v.(fn{i});
				v = rmfield(v,fn{i});
			elseif strncmpi(fn{i},'rotation',length(fn{i}))
				rotation_value = v.(fn{i});
				v = rmfield(v,fn{i});
			end
		end
		varargin{arg} = v;
		arg = arg + 1;
	elseif ischar(v)
		if strncmpi(v,'interpreter',length(v))
			interp_value = varargin{arg+1};
			remove = [remove,arg,arg+1]; %#ok<AGROW>
		elseif strncmpi(v,'rotation',length(v))
			rotation_value = varargin{arg+1};
			remove = [remove,arg,arg+1]; %#ok<AGROW>
		end
		arg = arg + 2;
	elseif arg == 1 && isscalar(v) && ishandle(v) && ...
			any(strcmp(get(h,'Type'),{'figure','uipanel'}))
		arg = arg + 1;
	else
		error('Invalid property or uicontrol parent.')
	end
end
varargin(remove) = [];

% Create uicontrol, get its properties then hide it.
if keep_handle
	set(h,varargin{:})
else
	h = uicontrol(varargin{:});
end
s = get(h);
if ~any(strcmp(s.Style,{'pushbutton','togglebutton','text'}))
	delete(h)
	error('''Style'' must be pushbutton, togglebutton or text.')
end
set(h,'Visible','off')

% Create axes.
parent = get(h,'Parent');
ax = axes('Parent',parent,...
	'Units',s.Units,...
	'Position',s.Position,...
	'XTick',[],'YTick',[],...
	'XColor',s.BackgroundColor,...
	'YColor',s.BackgroundColor,...
	'Box','on',...
	'Color',s.BackgroundColor);
% Adjust size of axes for best appearance.
set(ax,'Units','pixels')
pos = round(get(ax,'Position'));
if strcmp(s.Style,'text')
	set(ax,'Position',pos + [0 1 -1 -1])
else
	set(ax,'Position',pos + [4 4 -8 -8])
end
switch s.HorizontalAlignment
	case 'left'
		x = 0.0;
	case 'center'
		x = 0.5;
	case 'right'
		x = 1;
end
% Create text object.
text_obj = text('Parent',ax,...
	'Position',[x,0.5],...
	'String',s.String,...
	'Interpreter',interp_value,...
	'Rotation',rotation_value,...
	'HorizontalAlignment',s.HorizontalAlignment,...
	'VerticalAlignment','middle',...
	'FontName',s.FontName,...
	'FontSize',s.FontSize,...
	'FontAngle',s.FontAngle,...
	'FontWeight',s.FontWeight,...
	'Color',s.ForegroundColor);

% If we are creating something that looks like a text uicontrol then we're
% all done and we return the text object and axes handles rather than a
% uicontrol handle.
if strcmp(s.Style,'text')
	delete(h)
	if nargout
		hout = text_obj;
		ax_out = ax;
	end
	return
end

% Capture image of axes and then delete the axes.
frame = getframe(ax);
delete(ax)

% Build RGB image, set background pixels to NaN and put it in 'CData' for
% the uicontrol.
if isempty(frame.colormap)
	rgb = frame.cdata;
else
	rgb = reshape(frame.colormap(frame.cdata,:),[pos([4,3]),3]);
end
size_rgb = size(rgb);
rgb = double(rgb)/255;
back = repmat(permute(s.BackgroundColor,[1 3 2]),size_rgb(1:2));
isback = all(rgb == back,3);
rgb(repmat(isback,[1 1 3])) = NaN;
set(h,'CData',rgb,'String','','Visible',s.Visible)

% Assign output argument if necessary.
if nargout
	hout = h;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Axis/Angle to DCM
%
%   Purpose:
%       - Converts a given rotation vector and rotation angle into the
%       corresponding direction cosine matrix
%
%   dcm = srt2dcm(lambda,theta)
%
%   Inputs:
%       - lambda - 3 element unit vector of principle rotation axis
%       - theta - scalar rotation angle in radians
%
%   Outputs:
%       - dcm - direction cosine matrix to convert from inertial to fixed
%       frames assuming row vector notation (v_b = v_a * dcm_a2b)
%
%   Dependencies:
%       - skew_matrix.m - create skew symmetric matrix for cross product
%       operations
%
%   Author:
%       - Shankar Kulumani 19 Jan 2013
%
%   References
%       - AAE590 Lesson 5
%       - P. Hughes. Spacecraft attitude dynamics. Dover Publications, 2004.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dcm = srt2dcm(lambda,theta)

lambda = reshape(lambda,3,1);
cos_theta = cos(theta);
sin_theta = sin(theta);
lambda_skew = skew_matrix(lambda);

dcm = eye(3,3)*cos(theta) - sin_theta*lambda_skew + (1-cos_theta)*(lambda*lambda');

dcm = dcm';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Euler Axis/Angle to Quaternion
%
%   Purpose:
%       - Converts a given rotation vector and rotation angle into the
%       corresponding quaternion
%
%   dcm = srt2quat(lambda,theta)
%
%   Inputs:
%       - lambda - 3 element unit vector of principle rotation axis
%       - theta - scalar rotation angle in radians
%
%   Outputs:
%       - quat - quaternion representation of rotation (a to b )
%
%   Dependencies:
%       - none
%
%   Author:
%       - Shankar Kulumani 19 Jan 2013
%
%   References
%       - AAE590 Lesson 7
%       - P. Hughes. Spacecraft attitude dynamics. Dover Publications, 2004.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quat = srt2quat(lambda,theta)


e = lambda*sin(theta/2);

n = cos(theta/2);

quat = [e n];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simple rotation theorem
%
%   Purpose:
%       - Converts a vector in frame a to the corresponding vector in frame
%       b given a euler axis and angle of rotation
%
%   b = srt_rotvec(lambda,theta,a)
%
%   Inputs:
%       - lambda - euler axis of rotation 3x1 unit vector
%       - theta - euler angle of rotation in radians
%       - a - original vector in frame a
%
%   Outputs:
%       - b - a vector rotated into frame b
%
%   Dependencies:
%       - skew_matrix.m - creates skew symmetric matrix
%
%   Author:
%       - Shankar Kulumani 24 Jan 2013
%           - list revisions
%
%   References
%       - AAE590
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function b = srt_rotvec(lambda,theta,a)

cos_theta = cos(theta);
sin_theta = sin(theta);
a_skew = skew_matrix(a);

b = a*cos_theta - cross(a,lambda)*sin_theta + dot(a,lambda)*lambda*(1-cos_theta);
#####################################################################################################################
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Purpose: Rotation matrix about thrid axis (assumes row format)
%   b = a*dcm_a2b
%
%   Inputs:
%       - gamma - rotation angle (rad)
%
%   Outpus:
%       - rot3 - rotation matrix (3x3)
%
%   Dependencies:
%       - none
%
%   Author: Shankar Kulumani 18 Aug 2012
%               - 15 Sept 2012 fixed error
%               - 26 Jan 2013 - converted to row format representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

