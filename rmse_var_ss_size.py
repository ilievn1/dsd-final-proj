import numpy as np
import matplotlib.pyplot as plt
from .utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC

# Parameters
M = 4  # Number of elements in ULA
d = 0.5  # Element spacing in wavelengths
num_trials = 1000

inc_ang_deg = [20, 23]
thetas_deg = np.array(inc_ang_deg).reshape(1, -1)  # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1]  # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

snapshot_sizes = 2**np.arange(14)  # Snapshot sizes
snr = 10  # Fixed SNR in dB
snr_linear = 10 ** (snr / 10)  # SNR in linear scale
noise_power = 1 / snr_linear

# Generate ULA steering matrix
A = ula_steering_matrix(M, d, inc_ang_deg)  # (M x K)

# Generate scanning steering matrix
ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=1)  # (M x P)

# RMSE storage
rmse_matrix = np.zeros((4, len(snapshot_sizes)))  # Bartlett, Capon, MUSIC, CRLB

# Monte Carlo trials using matrix operations
for i, N in enumerate(snapshot_sizes):
    # Generate signals and noise
    noise = (np.random.randn(M, N, num_trials) + 1j * np.random.randn(M, N, num_trials)) * np.sqrt(noise_power / 2)
    signals = (np.random.randn(K, N, num_trials) + 1j * np.random.randn(K, N, num_trials)) / np.sqrt(2)

    # Generate transmitted signal
    steered_soi_matrix = np.tensordot(A, signals, axes=(1, 0))
    tx_signal = steered_soi_matrix + noise  # (M x N x num_trials)

    # Covariance matrix estimation
    R = np.tensordot(tx_signal, np.conj(tx_signal), axes=(1, 1)) / N  # (M x M x num_trials)

    # Bartlett
    estimated_angs, _ = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[0, i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))

    # Capon
    estimated_angs, _ = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[1, i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))

    # MUSIC
    estimated_angs, _ = MUSIC.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[2, i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))

    # CRLB (Placeholder)
    rmse_matrix[3, i] = 0  # Compute analytically based on Fisher information

# Plot RMSE as a function of Snapshot Size
plt.figure(figsize=(8, 6))
plt.plot(snapshot_sizes, rmse_matrix[0, :], label="Bartlett", marker="o:r")
plt.plot(snapshot_sizes, rmse_matrix[1, :], label="Capon", marker="x:g")
plt.plot(snapshot_sizes, rmse_matrix[2, :], label="MUSIC", marker="d:b")
plt.plot(snapshot_sizes, rmse_matrix[3, :], label="CRLB", marker="-k")
plt.title(f"RMSE as a Function of Snapshot Size \n M={M}, d={d}, inc_thetas={inc_ang_deg}, SNR={snr} dB")
plt.xlabel("Snapshot Size (N)")
plt.ylabel("RMSE (Degrees)")
plt.grid()
plt.legend()
plt.show()
