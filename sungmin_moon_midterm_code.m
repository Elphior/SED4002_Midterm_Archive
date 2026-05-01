%% Introduction to Neural Networks(SED4002) – Midterm 2026
% Sung Min Moon
% (2023199113)
% Seed is 9113
%rng(9113);
seed = 9113;

%% 1.2
N = 80;
T_raw = 350 + 200 * rand(N, 1);
P_raw = 10 +  20 * rand(N, 1);

noise = 0.05 * randn(N, 1);
y_raw = 85 + 5*sin(0.02*T_raw) + 0.3*P_raw.*tanh((T_raw - 450)/50) + noise;

figure('Name','P1.2 — Scatter plots');
subplot(1,2,1);
scatter(T_raw, y_raw, 30, 'b', 'filled');
xlabel('T (K)');
ylabel('Octane number y');
title('y vs T');
grid on;

subplot(1,2,2);
scatter(P_raw, y_raw, 30, 'r', 'filled');
xlabel('P (bar)');
ylabel('Octane number y');
title('y vs P');
grid on;


%% 1.3
[~, sort_idx] = sort(T_raw);
T_sorted = T_raw(sort_idx);
P_sorted = P_raw(sort_idx);
y_sorted = y_raw(sort_idx);

test_mask2 = false(N, 1);
test_idx = 1:6:N;
test_mask2(test_idx(1:12)) = true;
train_mask = ~test_mask2;

T_train = T_sorted(train_mask);
P_train = P_sorted(train_mask); 
y_train = y_sorted(train_mask);

T_test = T_sorted(test_mask2);
P_test = P_sorted(test_mask2);
y_test = y_sorted(test_mask2);

fprintf('P1.3: Training samples: %d,  Test samples: %d\n\n', sum(train_mask), sum(test_mask2));

%% 1.4
T_min = 350;
T_max = 550;
P_min = 10;
P_max = 30;

t_min = min(y_train);
t_max = max(y_train);

norm11 = @(x, lo, hi) 2*(x - lo)/(hi - lo) - 1;
inv11  = @(xn, lo, hi) (xn + 1)/2 * (hi - lo) + lo;

Tn_train = norm11(T_train, T_min, T_max);
Pn_train = norm11(P_train, P_min, P_max);
tn_train = norm11(y_train, t_min, t_max);

Tn_test  = norm11(T_test, T_min, T_max);
Pn_test  = norm11(P_test, P_min, P_max);
tn_test  = norm11(y_test, t_min, t_max);

% Input matrices: rows = features, cols = samples
X_train = [Tn_train, Pn_train]';
X_test  = [Tn_test,  Pn_test ]';
t_tr    = tn_train';
t_te    = tn_test';

fprintf('P1.4 Normalization ranges:\n');
fprintf('T:  [%.1f, %.1f] K\n', T_min, T_max);
fprintf('P:  [%.1f, %.1f] bar\n', P_min, P_max);
fprintf('y:  [%.4f, %.4f]\n',  t_min, t_max);
fprintf('Inverse map: y_hat = (a^n+1)/2 * (%.4f-%.4f) + %.4f\n\n', t_max, t_min, t_min);

figure('Name','P1.4 — Scaled data');
subplot(1,2,1);
scatter(Tn_train, tn_train, 30, 'b', 'filled');
xlabel('T^n');
ylabel('y^n');
title('Scaled: y^n vs T^n');
grid on;

subplot(1,2,2);
scatter(Pn_train, tn_train, 30, 'r', 'filled');
xlabel('P^n');
ylabel('y^n');
title('Scaled: y^n vs P^n');
grid on;
sgtitle('Problem 1.4 — Normalized scatter plots');

%% 3.2
R = 2;
S1_vals = [3, 5, 8, 10, 20];
param_counts = 4*S1_vals + 1;
S1 = 10;
fprintf('P3.2 — Training S1=10 network\n');
rng(seed);
[W1, b1, W2, b2] = nw_init(R, S1);
[W1, b1, W2, b2, sse_hist] = bayes_reg_train(X_train, t_tr, W1, b1, W2, b2, 100);

figure('Name','P3.2 — SSE vs iteration');
loglog(1:length(sse_hist), sse_hist, 'b-o', 'MarkerSize', 4, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('SSE (log scale)');
title('Problem 3.2 — SSE vs Iteration (log-log)');
grid on;
fprintf('Final SSE (run 1): %.6f\n\n', sse_hist(end));

%% 3.3
fprintf('P3.3 — Five restarts for S1=10:\n');
fprintf(' Run | Final SSE\n');
[best_W1, best_b1, best_W2, best_b2, best_gam] = best_of_n_runs(R, S1, X_train, t_tr, 100, 5, 2061);
fprintf('\n');

%% 3.4
figure('Name','P3.4 — Gamma vs iteration');
plot(1:length(best_gam), best_gam, 'r-o', 'MarkerSize',4,'LineWidth',1.5);
xlabel('Iteration'); ylabel('\gamma (effective parameters)');
title('Problem 3.4 — Effective parameters \gamma vs Iteration'); grid on;
total_p10 = 4*S1 + 1;
yline(total_p10, 'b--', sprintf('Total params = %d', total_p10), 'LabelHorizontalAlignment','left');
fprintf('P3.4: Converged gamma = %.2f, Total params = %d\n', best_gam(end), total_p10);

%% 4.1
fprintf('P4.1 — Architecture search (5 restarts each):\n');
fprintf('  S1 | Best SSE   | Conv. gamma | Total params\n');

results = zeros(length(S1_vals), 2);   % [best_sse, conv_gamma]

for si = 1:length(S1_vals)
    s = S1_vals(si);
    [W1r, b1r, W2r, b2r, gam_r, bsse] = best_of_n_runs(R, s, X_train, t_tr, 100, 5, seed + si);
    bgamma = gam_r(end);
    results(si,:) = [bsse, bgamma];
    fprintf('  %2d  | %.6f | %11.2f | %d\n', s, bsse, bgamma, param_counts(si));
end
fprintf('\n');

%% 4.2
figure('Name','P4.2 — SSE and gamma vs S1');
yyaxis left;
h1 = plot(S1_vals, results(:,1), 'b-o', 'LineWidth',2,'MarkerSize',8,'MarkerFaceColor','b');
ylabel('Best training SSE');
yyaxis right;
h2 = plot(S1_vals, results(:,2), 'r-s', 'LineWidth',2,'MarkerSize',8,'MarkerFaceColor','r');
ylabel('\gamma (effective parameters)');
xlabel('S^1 (hidden neurons)');
title('Problem 4.2 — SSE and \gamma vs S^1');
grid on;
legend([h1 h2], {'SSE','\gamma'}, 'Location','best');

%% 4.3
rec_idx = find(S1_vals == 5);
[W1f, b1f, W2f, b2f] = best_of_n_runs(R, S1_vals(rec_idx), X_train, t_tr, 100, 5, seed + rec_idx*10);
fprintf('P4.3 Recommended S1 = %d  (best SSE = %.6f,  gamma = %.2f)\n\n', S1_vals(rec_idx), results(rec_idx,1), results(rec_idx,2));

%% 5.1
a_train = net_fwd(X_train, W1f, b1f, W2f, b2f);
a_test  = net_fwd(X_test,  W1f, b1f, W2f, b2f);
figure('Name','P5.1 — Output vs Target (normalised)');
subplot(1,2,1);
scatter45(t_tr, a_train, 'b', 'Training set');
subplot(1,2,2);
scatter45(t_te, a_test, 'r', 'Test set');
sgtitle('Problem 5.1 — Network output vs target (normalised)');

%% 5.2
y_train_hat = inv11(a_train, t_min, t_max);
y_test_hat = inv11(a_test,  t_min, t_max);

err_train = y_train' - y_train_hat;
err_test = y_test'  - y_test_hat;

figure('Name','P5.2 — Error histogram');
histogram([err_train(:); err_test(:)], 15, 'FaceColor','b');
xlabel('Error (octane number)');
ylabel('Count');
title('Problem 5.2 — Network errors in octane units'); grid on;
xline( 0.2,'r--','+0.2 ON','LabelVerticalAlignment','bottom');
xline(-0.2,'r--','-0.2 ON','LabelVerticalAlignment','bottom');

mae_train = mean(abs(err_train));
mae_test = mean(abs(err_test));
wce_test = max(abs(err_test));
fprintf('P5.2 Results (octane units):\n');
fprintf('MAE (train): %.4f ON\n', mae_train);
fprintf('MAE (test): %.4f ON\n', mae_test);
fprintf('Worst-case (test): %.4f ON\n', wce_test);

%% 5.3
Tg = linspace(350, 550, 50);
Pg = linspace(10, 30, 50);
[TG, PG] = meshgrid(Tg, Pg);

TGn = norm11(TG, T_min, T_max);
PGn = norm11(PG, P_min, P_max);
Xg = [TGn(:)'; PGn(:)'];
ag = net_fwd(Xg, W1f, b1f, W2f, b2f);
YG = inv11(reshape(ag, size(TG)), t_min, t_max);

figure('Name','P5.3 — Response surface');
surf(TG, PG, YG, 'FaceAlpha', 0.7, 'EdgeColor','none');
colorbar;
hold on;
scatter3(T_train, P_train, y_train, 40, 'r', 'filled');
xlabel('T (K)');
ylabel('P (bar)');
zlabel('Octane number');
title('Problem 5.3 — Network response surface + training samples');
legend('Network surface','Training samples','Location','best');

%% LOCAL FUNCTIONS
function [W1, b1, W2, b2, best_gam, best_sse] = best_of_n_runs(R, S1, X, t, max_iter, n_runs, base_seed)
% Run n_runs independent Nguyen-Widrow + Bayesian-reg training restarts
% and return the weights from the run with the lowest final SSE.
    best_sse = inf;
    W1=[];
    b1=[];
    W2=[];
    b2=[];
    best_gam=[];
    for run = 1:n_runs
        rng(base_seed + run);
        [W1r, b1r, W2r, b2r] = nw_init(R, S1);
        [W1r, b1r, W2r, b2r, sse_r, gam_r] = bayes_reg_train(X, t, W1r, b1r, W2r, b2r, max_iter);
        fprintf('%2d | %.6f\n', run, sse_r(end));
        if sse_r(end) < best_sse
            best_sse = sse_r(end);
            W1=W1r; b1=b1r; W2=W2r; b2=b2r;
            best_gam=gam_r;
        end
    end
end

function scatter45(tgt, out, clr, ttl)
% Scatter plot of network output vs target with a 45-degree reference line.
    lo = min([tgt(:); out(:)]) - 0.05;
    hi = max([tgt(:); out(:)]) + 0.05;
    scatter(tgt, out, 40, clr, 'filled'); hold on;
    plot([lo hi],[lo hi],'w--','LineWidth',1.5);
    xlabel('Target t^n'); ylabel('Network output a^n');
    title(ttl); axis equal; grid on;
    xlim([lo hi]);
    ylim([lo hi]);
end

function [W1, b1, W2, b2] = nw_init(R, S1)
% Nguyen-Widrow hidden-layer init; random linear output.
%   W1 : S1 x R    b1 : S1 x 1
%   W2 :  1 x S1   b2 :  1 x 1
    beta = 0.7 * S1^(1/R);
    W1 = rand(S1, R) - 0.5;
    for i = 1:S1
        n = norm(W1(i,:));
        if n < 1e-12, n = 1; end
        W1(i,:) = beta * W1(i,:) / n;
    end
    b1 = (2*rand(S1,1) - 1) * beta;
    W2 = rand(1, S1) - 0.5;
    b2 = rand(1,1) - 0.5;
end

% -------------------------------------------------------------------------
function a = net_fwd(X, W1, b1, W2, b2)
% Forward pass through tansig hidden layer + linear output
    N = size(X, 2);
    H = tanh(W1 * X + b1 * ones(1,N));
    a = W2  * H + b2 * ones(1,N);
end

function [a, J] = net_jacobian(X, W1, b1, W2, b2)
% Eq. 12.37
    N = size(X, 2);
    R = size(X, 1);
    S1 = size(W1, 1);
    n_p = S1*(R+2) + 1;

    Z = W1 * X + b1 * ones(1,N);
    H = tanh(Z);
    a = W2 * H + b2 * ones(1,N);
    sech2 = 1 - H.^2;

    J = zeros(N, n_p);
    col = 0;
    for i = 1:S1
        for r = 1:R
            col = col + 1;
            J(:, col) = ( W2(i) * sech2(i,:) .* X(r,:) )';
        end
    end

    for i = 1:S1
        col = col + 1;
        J(:, col) = ( W2(i) * sech2(i,:) )';
    end
    for j = 1:S1
        col = col + 1;
        J(:, col) = H(j,:)';
    end

    % d a_q / d b2 = 1
    col = col + 1;
    J(:, col) = ones(N,1);
end

function [W1, b1, W2, b2, sse_hist, gamma_hist] = bayes_reg_train(X, t, W1, b1, W2, b2, max_iter)
    N = size(X, 2);
    S1 = size(W1, 1);
    R = size(X, 1);
    n_p = S1*(R+2) + 1;

    function w = pack(W1_, b1_, W2_, b2_)
        w = [W1_(:); b1_(:); W2_(:); b2_(:)];   % n_p x 1
    end
    function [W1_, b1_, W2_, b2_] = unpack(w)
        i1 = S1*R;
        W1_ = reshape(w(1:i1), S1, R);
        i2 = i1+S1;
        b1_ = reshape(w(i1+1:i2), S1, 1);
        i3 = i2+S1;
        W2_ = reshape(w(i2+1:i3), 1, S1);
        b2_ = w(i3+1);
    end

    w = pack(W1, b1, W2, b2);

    % Hyperparameter initialisation
    alpha = 0.005;
    beta_h = 1 / (0.05^2);

    mu = 0.005;
    mu_max = 1e10;
    mu_inc = 10;
    mu_dec = 0.1;

    sse_hist = zeros(max_iter, 1);
    gamma_hist = zeros(max_iter, 1);

    for iter = 1:max_iter

        [W1, b1, W2, b2] = unpack(w);

        [a, J] = net_jacobian(X, W1, b1, W2, b2);
        Jt  = J';

        e = t - a;
        SSE = sum(e .^ 2);
        SW = sum(w .^ 2);

        H_gn = beta_h * (Jt * J) + alpha * eye(n_p);
        g = -beta_h * (Jt * e') + alpha * w;

        dw = -(H_gn + mu * eye(n_p)) \ g;
        w_new = w + dw;

        [W1n,b1n,W2n,b2n] = unpack(w_new);
        a_new = net_fwd(X, W1n, b1n, W2n, b2n);
        F_old = beta_h * SSE + alpha * SW;
        F_new = beta_h * sum((t - a_new).^2) + alpha * sum(w_new.^2);

        if F_new < F_old
            w = w_new;
            mu = max(mu * mu_dec, 1e-20);
        else
            mu = min(mu * mu_inc, mu_max);
        end

        [W1, b1, W2, b2] = unpack(w);
        [a_cur, J_cur] = net_jacobian(X, W1, b1, W2, b2);

        e_cur = t - a_cur;
        SSE_cur = sum(e_cur.^ 2);
        SW_cur = sum(w.^ 2);

        H_cur = beta_h * (J_cur' * J_cur) + alpha * eye(n_p);

        d_cur = diag(H_cur);
        d_cur(abs(d_cur) < 1e-12) = 1e-12;
        gamma = n_p - alpha * sum(1 ./ d_cur);
        gamma = max(gamma, 1e-6);

        alpha = gamma / (2 * SW_cur  + eps);
        beta_h = max((N - gamma) / (2 * SSE_cur + eps), 1e-6);

        sse_hist(iter) = SSE_cur;
        gamma_hist(iter) = gamma;
    end

    [W1, b1, W2, b2] = unpack(w);
end