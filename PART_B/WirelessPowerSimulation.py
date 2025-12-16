import numpy as np
import matplotlib.pyplot as plt


# Parameters (can be changed)

n0 = 1e-7                    # Noise power 
x_th = 0.5                   # Threshold for exceedance probability
target_prob = 0.90           # Desired P(Pr > x_th)


# Helper: Theoretical functions from Part A

def theoretical_mean(Pt, n0):
    return Pt + n0

def theoretical_exceedance(x_th, Pt, n0):
    return np.exp(-x_th / (Pt + n0))

def theoretical_cdf(x, Pt, n0):
    """CDF of exponential distribution with mean = Pt + n0"""
    mu = Pt + n0
    return 1 - np.exp(-x / mu)


# Monte Carlo Simulation Function

def simulate_received_power(Pt, n0, T):
    """
    Simulate T trials of received signal power Pr = |Y|^2
    Y = sqrt(Pt) * H * s + N
    """
    # Generate channel H ~ Rayleigh: (X + jY)/sqrt(2), X,Y ~ N(0,1)
    H = (np.random.randn(T) + 1j * np.random.randn(T)) / np.sqrt(2)
    
    # Generate symbols s in {+1, -1} with equal prob
    s = np.random.choice([-1, 1], size=T)
    
    # Generate noise N = sqrt(n0/2) * (Z + jW), Z,W ~ N(0,1)
    N = np.sqrt(n0 / 2) * (np.random.randn(T) + 1j * np.random.randn(T))
    
    # Received signal
    Y = np.sqrt(Pt) * H * s + N
    
    # Received power
    Pr = np.abs(Y) ** 2
    
    return Pr


# Task: Find best number of trials T

T_values = [10**3, 10**4, 10**5, 10**6, 10**7]
Pt_test = 0.5  # temporary Pt for testing convergence

print("=== Convergence Test for T ===")
print(f"{'T':>10} | {'Mean Error (%)':>15} | {'Exceed Error (%)':>18} | Runtime (s)")
print("-" * 60)

convergence_results = []
for T in T_values:
    import time
    start = time.time()
    Pr = simulate_received_power(Pt_test, n0, T)
    runtime = time.time() - start
    
    # Empirical stats
    mean_emp = np.mean(Pr)
    exceed_emp = np.mean(Pr > x_th)
    
    # Theoretical
    mean_th = theoretical_mean(Pt_test, n0)
    exceed_th = theoretical_exceedance(x_th, Pt_test, n0)
    
    # % errors
    err_mean = 100 * abs(mean_emp - mean_th) / mean_th
    err_exceed = 100 * abs(exceed_emp - exceed_th) / exceed_th
    
    convergence_results.append((T, err_mean, err_exceed, runtime))
    print(f"{T:>10} | {err_mean:>14.4f}% | {err_exceed:>17.4f}% | {runtime:.4f}")

# Choose T = 1e5 as balance (as discussed)
T_opt = 10**5
print(f"\nChosen T = {T_opt} for rest of simulation.\n")


# Task: Find Pt such that P(Pr > 0.5) = 0.90

# Use analytical inverse from Part A3:
Pt_needed = (-x_th / np.log(target_prob)) - n0
print(f"Analytically required Pt = {Pt_needed:.6f}")

# Verify via simulation
Pr_verify = simulate_received_power(Pt_needed, n0, T_opt)
p_sim = np.mean(Pr_verify > x_th)
p_th = theoretical_exceedance(x_th, Pt_needed, n0)

print(f"Simulation P(Pr > {x_th}) = {p_sim:.5f}")
print(f"Theoretical P(Pr > {x_th}) = {p_th:.5f}")
print(f"Error = {100*abs(p_sim - p_th)/p_th:.3f}%\n")


# Task: Plot CDF for different Pt values

Pt_list = [0.01, 0.05, Pt_needed, 0.2, 0.5]
x_plot = np.linspace(0, 2, 1000)

plt.figure(figsize=(8, 5))
for Pt in Pt_list:
    # Simulate
    Pr = simulate_received_power(Pt, n0, T_opt)
    Pr_sorted = np.sort(Pr)
    cdf_emp = np.arange(1, T_opt + 1) / T_opt
    plt.plot(Pr_sorted, cdf_emp, linestyle='--', alpha=0.7, label=f'Sim Pt={Pt:.3f}')
    
    # Theoretical CDF
    cdf_th = theoretical_cdf(x_plot, Pt, n0)
    plt.plot(x_plot, cdf_th, linewidth=2, label=f'Theory Pt={Pt:.3f}')

plt.axvline(x=x_th, color='k', linestyle=':', label='Threshold x_th=0.5')
plt.xlabel('Received Power $P_r$')
plt.ylabel('CDF $F_{P_r}(x)$')
plt.title('Empirical vs Theoretical CDF for Different $P_t$')
plt.legend()
plt.grid(True)
plt.xlim(0, 1.5)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# Task: Plot CDF for different n0 (with fixed Pt = Pt_needed)

n0_list = [1e-9, 1e-7, 1e-5, 1e-3]  # increasing noise
Pt_fixed = Pt_needed

plt.figure(figsize=(8, 5))
for n0_val in n0_list:
    Pr = simulate_received_power(Pt_fixed, n0_val, T_opt)
    Pr_sorted = np.sort(Pr)
    cdf_emp = np.arange(1, T_opt + 1) / T_opt
    plt.plot(Pr_sorted, cdf_emp, label=f'n0 = {n0_val:.0e}')

    # Optional: add theoretical curve
    cdf_th = theoretical_cdf(x_plot, Pt_fixed, n0_val)
    plt.plot(x_plot, cdf_th, '--', linewidth=1)

plt.xlabel('Received Power $P_r$')
plt.ylabel('CDF $F_{P_r}(x)$')
plt.title(f'Effect of Noise Power $n_0$ on CDF (Fixed $P_t = {Pt_needed:.4f}$)')
plt.legend()
plt.grid(True)
plt.xlim(0, 1.5)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# Final Verification Table

print("\n=== Final Verification (T = 1e5, Pt chosen analytically) ===")
mean_emp_final = np.mean(Pr_verify)
mean_th_final = theoretical_mean(Pt_needed, n0)
err_mean_final = 100 * abs(mean_emp_final - mean_th_final) / mean_th_final

print(f"Empirical Mean Pr: {mean_emp_final:.8f}")
print(f"Theoretical Mean:  {mean_th_final:.8f}")
print(f"Mean Error:        {err_mean_final:.4f}%")
print(f"Exceedance Error:  {100*abs(p_sim - p_th)/p_th:.4f}%")