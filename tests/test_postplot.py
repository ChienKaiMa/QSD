import numpy as np
from plots import plot_prob_noise_tol

case_id = f"test_mix_scan_noise"
tol = [0.5 - 0.04 * i for i in range(10)]
noise_levels = [0.01 * i for i in range(1, 11)]

# Load probs
prob = np.load(f"prob_q2_n3_s42_ideal_mix_scan_noise.npy")

plot_prob_noise_tol(tol, noise_levels, prob, case_id)