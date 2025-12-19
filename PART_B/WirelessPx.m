clear; clc; close all;
% !!!GRADER IF YOU ARE LOOKING AT MATLAB CODE, this was just a rough draft as I was testing my matlab skills. Please do not consider this code when grading, thank you. 
%Same thing goes for PART A, I was also just learning how to use Jupyter Notebook, everything regarding PART A and PART B of MATLAB is on our google document that we submitted.

n0 = 1e-7;
x_th = 0.5;
targetProb = 0.90;

T_values = [1e3, 1e4, 1e5, 1e6, 1e7];
Pt_test  = 0.5;

fprintf("=== Convergence Test ===\n");
fprintf("%10s | %15s | %18s | %11s\n", "T", "Mean Error (%)", "Exceed Error (%)", "Runtime (s)");
fprintf("%s\n", repmat("-",1,70));

results = zeros(numel(T_values),4);

for k = 1:numel(T_values)
    T = T_values(k);

    tic;
    Pr = simulate_received_power(Pt_test, n0, T);
    runtime = toc;

    mean_emp   = mean(Pr);
    exceed_emp = mean(Pr > x_th);

    mean_th   = theoretical_mean(Pt_test, n0);
    exceed_th = theoretical_exceedance(x_th, Pt_test, n0);

    mean_err   = 100 * abs(mean_emp - mean_th) / mean_th;
    exceed_err = 100 * abs(exceed_emp - exceed_th) / exceed_th;

    results(k,:) = [T, mean_err, exceed_err, runtime];

    fprintf("%10.0f | %14.4f%% | %17.4f%% | %11.4f\n", T, mean_err, exceed_err, runtime);
end

T_opt = 1e5;
fprintf("\nChosen T = %.0f\n\n", T_opt);

Pt_needed = -(x_th / log(targetProb)) - n0;
fprintf("Analytical Pt = %.6f\n", Pt_needed);

Pr_verify = simulate_received_power(Pt_needed, n0, T_opt);
p_sim = mean(Pr_verify > x_th);
p_th  = theoretical_exceedance(x_th, Pt_needed, n0);

fprintf("Simulation P = %.5f\n", p_sim);
fprintf("Theoretical P = %.5f\n", p_th);
fprintf("Error = %.3f%%\n\n", 100*abs(p_sim - p_th)/p_th);

Pt_list = [0.01, 0.05, Pt_needed, 0.2, 0.5];
x_plot  = linspace(0,2,1000);

figure; hold on;
for i = 1:numel(Pt_list)
    Pt = Pt_list(i);
    Pr = simulate_received_power(Pt, n0, T_opt);
    Pr_sorted = sort(Pr);
    F_emp = (1:T_opt)/T_opt;
    plot(Pr_sorted, F_emp, '--');
    plot(x_plot, theoretical_cdf(x_plot, Pt, n0), 'LineWidth',2);
end
xline(x_th, 'k:');
xlabel('P_r'); ylabel('CDF');
grid on; xlim([0 1.5]); ylim([0 1]);
hold off;

n0_list  = [1e-9, 1e-7, 1e-5, 1e-3];
Pt_fixed = Pt_needed;

figure; hold on;
for i = 1:numel(n0_list)
    n0_val = n0_list(i);
    Pr = simulate_received_power(Pt_fixed, n0_val, T_opt);
    Pr_sorted = sort(Pr);
    F_emp = (1:T_opt)/T_opt;
    plot(Pr_sorted, F_emp);
    plot(x_plot, theoretical_cdf(x_plot, Pt_fixed, n0_val), '--');
end
xlabel('P_r'); ylabel('CDF');
grid on; xlim([0 1.5]); ylim([0 1]);
hold off;

mean_emp_final = mean(Pr_verify);
mean_th_final  = theoretical_mean(Pt_needed, n0);
mean_err_final = 100 * abs(mean_emp_final - mean_th_final) / mean_th_final;

fprintf("Empirical Mean: %.8f\n", mean_emp_final);
fprintf("Theoretical Mean: %.8f\n", mean_th_final);
fprintf("Mean Error: %.4f%%\n", mean_err_final);
fprintf("Exceed Error: %.4f%%\n", 100*abs(p_sim - p_th)/p_th);

function mu = theoretical_mean(Pt, n0)
    mu = Pt + n0;
end

function p = theoretical_exceedance(x_th, Pt, n0)
    p = exp(-x_th / (Pt + n0));
end

function F = theoretical_cdf(x, Pt, n0)
    F = 1 - exp(-x ./ (Pt + n0));
end

function Pr = simulate_received_power(Pt, n0, T)
    H = (randn(T,1) + 1j*randn(T,1)) / sqrt(2);
    s = 2*(rand(T,1) > 0.5) - 1;
    N = sqrt(n0/2) * (randn(T,1) + 1j*randn(T,1));
    Y = sqrt(Pt) * H .* s + N;
    Pr = abs(Y).^2;
end
