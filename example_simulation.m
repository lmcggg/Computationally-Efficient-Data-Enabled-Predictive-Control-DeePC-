%% Example Simulation of Data-Enabled Predictive Control with Efficient Reformulation

clear; clc; close all;

A = [0.921  0      0.041  0    ;
     0      0.918  0      0.033;
     0      0      0.924  0    ;
     0      0      0      0.937];

B = [0.017  0.001;
     0.001  0.023;
     0      0.061;
     0.072  0    ];

C = [1  0  0  0;
     0  1  0  0];

D = zeros(2, 2);

nx = size(A, 1);
nu = size(B, 2);
ny = size(C, 1);

sys_func = @(x, u) deal(A*x + B*u, C*x + D*u);

T_train = 400;
u_train = 2 * rand(nu, T_train) - 1;

x_train = zeros(nx, T_train+1);
y_train = zeros(ny, T_train);
x_train(:, 1) = randn(nx, 1);

noise_bound = 0.002;

for t = 1:T_train
    [x_train(:, t+1), y_train(:, t)] = sys_func(x_train(:, t), u_train(:, t));
    y_train(:, t) = y_train(:, t) + noise_bound * (2 * rand(ny, 1) - 1);
end

n = 4;
L = 30;

Q = 5 * eye(ny);
R = 1e-4 * eye(nu);

noise_bound = 0.002;
lambda_alpha_times_e = 0.1;
lambda_alpha = lambda_alpha_times_e / noise_bound;
lambda_sigma = 2000;

u_min = -1 * ones(nu, 1);
u_max = 2 * ones(nu, 1);

controller = DeePC_Efficient(nu, ny, n, L, u_train, y_train, ...
    'Q', Q, 'R', R, ...
    'lambda_g', lambda_alpha, ...
    'lambda_sigma', lambda_sigma, ...
    'u_min', u_min, 'u_max', u_max, ...
    'rho', 0.95, 'sigma_max', 0.1);

info = controller.getInfo();

T_sim = 100;
t_sim = 0:T_sim-1;

y_setpoint = [0.65; 0.77];
u_setpoint = [1; 1];

y_ref = zeros(ny, T_sim);
u_ref = zeros(nu, T_sim);

for t = 1:T_sim
    y_ref(:, t) = y_setpoint;
    u_ref(:, t) = u_setpoint;
end

x0_sim = [0.1; 0.1; 0.05; 0.05];
[u_traj, y_traj, cost_traj] = controller.simulate(y_ref, u_ref, x0_sim, sys_func, T_sim);

tracking_error = y_traj - y_ref;
rmse_total = sqrt(mean(tracking_error(:).^2));
rmse_output1 = sqrt(mean(tracking_error(1,:).^2));
rmse_output2 = sqrt(mean(tracking_error(2,:).^2));
control_effort = sum(u_traj(:).^2);

figure('Position', [100, 100, 1400, 1000]);

subplot(3, 2, 1);
plot(t_sim, y_ref(1, :), 'r--', 'LineWidth', 2, 'DisplayName', 'Reference');
hold on;
plot(t_sim, y_traj(1, :), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Tank 1 Level');
plot(t_sim, y_setpoint(1)*ones(size(t_sim)), 'k:', 'LineWidth', 1, 'DisplayName', 'Setpoint');
xlabel('Time Step');
ylabel('Water Level (Tank 1)');
title('Tank 1 Water Level Tracking');
legend('Location', 'best');
grid on;

subplot(3, 2, 2);
plot(t_sim, y_ref(2, :), 'r--', 'LineWidth', 2, 'DisplayName', 'Reference');
hold on;
plot(t_sim, y_traj(2, :), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Tank 2 Level');
plot(t_sim, y_setpoint(2)*ones(size(t_sim)), 'k:', 'LineWidth', 1, 'DisplayName', 'Setpoint');
xlabel('Time Step');
ylabel('Water Level (Tank 2)');
title('Tank 2 Water Level Tracking');
legend('Location', 'best');
grid on;

subplot(3, 2, 3);
stairs(t_sim, u_traj(1, :), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pump 1');
hold on;
plot(t_sim, u_min(1)*ones(size(t_sim)), 'r--', 'LineWidth', 1, 'DisplayName', 'Constraints');
plot(t_sim, u_max(1)*ones(size(t_sim)), 'r--', 'LineWidth', 1);
plot(t_sim, u_setpoint(1)*ones(size(t_sim)), 'k:', 'LineWidth', 1, 'DisplayName', 'Steady-state');
xlabel('Time Step');
ylabel('Pump 1 Input');
title('Pump 1 Control Signal');
legend('Location', 'best');
grid on;

subplot(3, 2, 4);
stairs(t_sim, u_traj(2, :), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Pump 2');
hold on;
plot(t_sim, u_min(2)*ones(size(t_sim)), 'r--', 'LineWidth', 1, 'DisplayName', 'Constraints');
plot(t_sim, u_max(2)*ones(size(t_sim)), 'r--', 'LineWidth', 1);
plot(t_sim, u_setpoint(2)*ones(size(t_sim)), 'k:', 'LineWidth', 1, 'DisplayName', 'Steady-state');
xlabel('Time Step');
ylabel('Pump 2 Input');
title('Pump 2 Control Signal');
legend('Location', 'best');
grid on;

subplot(3, 2, 5);
plot(t_sim, tracking_error(1, :), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Tank 1 Error');
hold on;
plot(t_sim, tracking_error(2, :), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Tank 2 Error');
xlabel('Time Step');
ylabel('Tracking Error');
title('Water Level Tracking Errors');
legend('Location', 'best');
grid on;

subplot(3, 2, 6);
semilogy(t_sim, cost_traj, 'b-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Cost (log scale)');
title('Cost Evolution');
grid on;

sgtitle('DeePC Efficient Reformulation - Four-Tank System Results', 'FontSize', 14, 'FontWeight', 'bold');




