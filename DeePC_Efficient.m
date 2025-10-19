%% Data-Enabled Predictive Control with Computationally Efficient Reformulation

classdef DeePC_Efficient < handle
    properties
        % System parameters
        nu          % Number of inputs
        ny          % Number of outputs
        n           % System order (past window length)
        L           % Prediction horizon
        
        % Data matrices
        W_bak       % Backup data matrix (fixed)
        W_exp       % Online data matrix (exponentially forgetting)
        K_bak       % Backup weighting matrix
        K_exp       % Online weighting matrix
        
        % Low-dimensional matrices
        G_bak       % G_backup = W_bak * K_bak^(-1) * W_bak^T
        G_exp       % G_online = W_exp * K_exp^(-1) * W_exp^T
        G           % Combined G = G_bak + G_exp
        
        % Selection matrices
        S_u         % Input selection matrix
        S_y         % Output selection matrix
        
        % Control parameters
        Q           % Output tracking weight
        R           % Input regularization weight
        lambda_g    % Regularization parameter for alpha (λα)
        lambda_sigma % Regularization parameter for slack variable (λσ)
        rho         % Forgetting factor (0 < rho < 1)
        
        % Constraints
        u_min       % Input lower bounds
        u_max       % Input upper bounds
        sigma_max   % Slack variable bounds
        
        % Reference and state
        y_ref       % Reference trajectory
        u_past      % Past input sequence
        y_past      % Past output sequence
        
        % Dimensions
        r           % Number of rows in W (total constraint dimension)
        c_bak       % Number of columns in W_bak
        c_exp       % Number of columns in W_exp (current)
        
        % Online data buffers for proper Hankel structure
        u_online_buffer % Buffer for recent L+n inputs
        y_online_buffer % Buffer for recent L+n outputs
    end
    
    methods
        function obj = DeePC_Efficient(nu, ny, n, L, u_data, y_data, varargin)
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'Q', eye(ny), @(x) isnumeric(x) && size(x,1)==ny);
            addParameter(p, 'R', eye(nu), @(x) isnumeric(x) && size(x,1)==nu);
            addParameter(p, 'lambda_g', 1e-3, @(x) isscalar(x) && x > 0);
            addParameter(p, 'lambda_sigma', 1e6, @(x) isscalar(x) && x >= 0);
            addParameter(p, 'rho', 0.95, @(x) isscalar(x) && x > 0 && x < 1);
            addParameter(p, 'u_min', -inf(nu,1), @(x) isnumeric(x) && length(x)==nu);
            addParameter(p, 'u_max', inf(nu,1), @(x) isnumeric(x) && length(x)==nu);
            addParameter(p, 'sigma_max', 1e-2, @(x) isscalar(x) && x > 0);
            parse(p, varargin{:});
            
            % Store system parameters
            obj.nu = nu;
            obj.ny = ny;
            obj.n = n;
            obj.L = L;
            
            % Store control parameters
            obj.Q = p.Results.Q;
            obj.R = p.Results.R;
            obj.lambda_g = p.Results.lambda_g;
            obj.lambda_sigma = p.Results.lambda_sigma;
            obj.rho = p.Results.rho;
            obj.u_min = p.Results.u_min;
            obj.u_max = p.Results.u_max;
            obj.sigma_max = p.Results.sigma_max;
            
            % Initialize with backup data
            obj.initializeBackupData(u_data, y_data);
            
            % Initialize online components
            obj.initializeOnlineData();
            
            % Construct selection matrices
            obj.constructSelectionMatrices();
            
            % Initialize past sequences
            obj.u_past = zeros(nu, n);
            obj.y_past = zeros(ny, n);
            obj.y_ref = zeros(ny, L);
            
            % Initialize online data buffers with zeros
            buffer_len = obj.L + obj.n;
            obj.u_online_buffer = zeros(obj.nu, buffer_len);
            obj.y_online_buffer = zeros(obj.ny, buffer_len);
        end
        
        function initializeBackupData(obj, u_data, y_data)
            
            % Validate data dimensions
            T_data = size(u_data, 2);
            assert(size(y_data, 2) == T_data, 'Input and output data must have same length');
            assert(T_data >= obj.L + obj.n, 'Data length must be at least L+n');
            
            H_u_bak = obj.constructHankelMatrix(u_data, obj.L + obj.n);
            H_y_bak = obj.constructHankelMatrix(y_data, obj.L + obj.n);
            
            ones_row = ones(1, size(H_u_bak, 2));
            obj.W_bak = [ones_row; H_u_bak; H_y_bak];
            
            obj.r = size(obj.W_bak, 1);
            obj.c_bak = size(obj.W_bak, 2);
            
            obj.K_bak = eye(obj.c_bak);
            obj.G_bak = obj.W_bak * (obj.K_bak \ obj.W_bak');
        end
        
        function initializeOnlineData(obj)
            obj.W_exp = zeros(obj.r, 0);
            obj.K_exp = zeros(0, 0);
            obj.c_exp = 0;
            obj.G_exp = zeros(obj.r, obj.r);
            obj.G = obj.G_bak + obj.G_exp;
        end
        
        function constructSelectionMatrices(obj)
            obj.S_u = zeros(obj.L * obj.nu, obj.r);
            u_future_start = 1 + obj.n * obj.nu + 1;
            
            for i = 1:obj.L * obj.nu
                obj.S_u(i, u_future_start + i - 1) = 1;
            end
            
            obj.S_y = zeros(obj.L * obj.ny, obj.r);
            y_future_start = 1 + (obj.n + obj.L) * obj.nu + obj.n * obj.ny + 1;
            
            for i = 1:obj.L * obj.ny
                obj.S_y(i, y_future_start + i - 1) = 1;
            end
        end
        
        function H = constructHankelMatrix(obj, data, window_length)
            [dim, T] = size(data);
            num_cols = T - window_length + 1;
            
            if num_cols <= 0
                error('Data length must be at least window_length');
            end
            
            H = zeros(dim * window_length, num_cols);
            
            for i = 1:num_cols
                col_data = data(:, i:i+window_length-1);
                H(:, i) = col_data(:);
            end
        end
        
        function updateOnlineData(obj, u_new, y_new)
            obj.u_online_buffer = [obj.u_online_buffer(:, 2:end), u_new(:)];
            obj.y_online_buffer = [obj.y_online_buffer(:, 2:end), y_new(:)];
            
            w_new = obj.constructNewDataColumn();
            obj.G_exp = obj.rho * obj.G_exp + w_new * w_new';
            obj.W_exp = [obj.W_exp, w_new];
            obj.c_exp = obj.c_exp + 1;
            obj.G = obj.G_bak + obj.G_exp;
            obj.updatePastSequences(u_new, y_new);
        end
        
        function w_new = constructNewDataColumn(obj)
            w_new = [1; obj.u_online_buffer(:); obj.y_online_buffer(:)];
        end
        
        function updatePastSequences(obj, u_new, y_new)
            obj.u_past = [obj.u_past(:, 2:end), u_new(:)];
            obj.y_past = [obj.y_past(:, 2:end), y_new(:)];
        end
        
        function u_opt = solve(obj, y_ref_new, u_ref_new)
            obj.y_ref = y_ref_new;
            
            if nargin < 3
                u_ref_new = zeros(obj.nu, obj.L);
            end
            
            y_ref_extended = [obj.y_past(:); obj.y_ref(:)];
            
            if ~exist('cvx_begin', 'file')
                error('CVX is required but not found. Please install CVX.');
            end
            
            epsilon = 1e-7;
            G_reg = obj.G + epsilon * eye(obj.r);
            
            cvx_begin quiet
                variable alpha_var(obj.r)
                variable sigma_var(obj.ny * obj.n)
                
                G_alpha = G_reg * alpha_var;
                y_pred = obj.S_y * G_alpha;
                u_pred = obj.S_u * G_alpha;
                y_past_pred = obj.extractPastOutputs(G_alpha);
                
                tracking_cost = quad_form(y_pred - obj.y_ref(:), kron(eye(obj.L), obj.Q));
                input_reg_cost = quad_form(u_pred - u_ref_new(:), kron(eye(obj.L), obj.R));
                alpha_reg_cost = obj.lambda_g * quad_form(alpha_var, G_reg);
                slack_cost = obj.lambda_sigma * sum_square(sigma_var);
                
                minimize(tracking_cost + input_reg_cost + alpha_reg_cost + slack_cost)
                
                subject to
                    u_pred_mat = reshape(u_pred, obj.nu, obj.L);
                    u_pred_mat >= obj.u_min * ones(1, obj.L);
                    u_pred_mat <= obj.u_max * ones(1, obj.L);
                    y_past_pred == obj.y_past(:) + sigma_var;
                    -obj.sigma_max <= sigma_var <= obj.sigma_max;
                    G_alpha(1) == 1;
            cvx_end
            
            if ~strcmp(cvx_status, 'Solved')
                warning('CVX solver status: %s', cvx_status);
            end
            
            u_pred_opt = reshape(u_pred, obj.nu, obj.L);
            u_opt = u_pred_opt(:, 1);
        end
        
        function y_past_pred = extractPastOutputs(obj, G_alpha)
            y_start_idx = 2 + (obj.L + obj.n) * obj.nu;
            y_past_end_idx = y_start_idx + obj.n * obj.ny - 1;
            y_past_pred = G_alpha(y_start_idx:y_past_end_idx);
        end
        
        function [u_traj, y_traj, cost_traj] = simulate(obj, y_ref_traj, u_ref_traj, x0, sys, T_sim)
            u_traj = zeros(obj.nu, T_sim);
            y_traj = zeros(obj.ny, T_sim);
            cost_traj = zeros(1, T_sim);
            
            if isnumeric(sys)
                x = x0;
            else
                x = x0;
            end
            
            for t = 1:T_sim
                if size(y_ref_traj, 2) >= t + obj.L - 1
                    y_ref_current = y_ref_traj(:, t:t+obj.L-1);
                else
                    y_ref_current = [y_ref_traj(:, t:end), ...
                                   repmat(y_ref_traj(:, end), 1, obj.L - (size(y_ref_traj,2) - t + 1))];
                end
                
                if size(u_ref_traj, 2) >= t + obj.L - 1
                    u_ref_current = u_ref_traj(:, t:t+obj.L-1);
                else
                    u_ref_current = [u_ref_traj(:, t:end), ...
                                   repmat(u_ref_traj(:, end), 1, obj.L - (size(u_ref_traj,2) - t + 1))];
                end
                
                u_opt = obj.solve(y_ref_current, u_ref_current);
                u_traj(:, t) = u_opt;
                
                if isnumeric(sys) && size(sys, 1) > 0
                    A = sys(1:size(x,1), 1:size(x,1));
                    B = sys(1:size(x,1), size(x,1)+1:end);
                    C = sys(size(x,1)+1:end, 1:size(x,1));
                    D = sys(size(x,1)+1:end, size(x,1)+1:end);
                    
                    y_traj(:, t) = C * x + D * u_opt;
                    x = A * x + B * u_opt;
                else
                    [x, y_traj(:, t)] = sys(x, u_opt);
                end
                
                obj.updateOnlineData(u_opt, y_traj(:, t));
                cost_traj(t) = norm(y_traj(:, t) - y_ref_traj(:, min(t, size(y_ref_traj,2))))^2 + ...
                              obj.lambda_g * norm(u_opt)^2;
            end
        end
        
        function info = getInfo(obj)
            info.r = obj.r;
            info.c_bak = obj.c_bak;
            info.c_exp = obj.c_exp;
            info.G_condition = cond(obj.G);
            info.G_rank = rank(obj.G);
            info.G_bak_condition = cond(obj.G_bak);
            info.G_exp_condition = cond(obj.G_exp + 1e-12*eye(size(obj.G_exp)));
        end
    end
end
