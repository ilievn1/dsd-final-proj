import numpy as np
import matplotlib.pyplot as plt
from .utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC

# Parameters
M = 4  # Number of elements in ULA
d = 0.5  # Element spacing in wavelengths
N = 100  # Number of snapshots
num_trials = 1000

inc_ang_deg = [20, 23]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

ang_separations = np.arange(0, 30, 0.1) # in deg

# RMSE storage
rmse_matrix = np.zeros(4,len(ang_separations)) # 3 + 1 = num of estimator, currently CBF,MVDR,MUSIC + CRLB

# Monte Carlo trials using matrix operations
for i, sep in enumerate(ang_separations):
    inc_ang_deg = [20, 20 + sep]
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)

    # Generate ULA steering matrix
    A = ula_steering_matrix(M, d, inc_ang_deg)  # (M x K)

    # Generate scanning steering matrix
    ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=0.1 + sep)  # (M x P)

    # Generate signals and noise
    signals = np.random.randn(K, N, num_trials)
    noise = 0.5 * (np.random.randn(M, N, num_trials) + 1j * np.random.randn(M, N, num_trials))

    steered_soi_matrix = np.tensordot(A, signals, axes=(1, 0))
    tx_signal = steered_soi_matrix + noise  # (M x N x num_trials)

    # Covariance matrix estimation
    R = np.tensordot(tx_signal, np.conj(tx_signal), axes=(1, 1)) / N  # (M x N x num_trials)

    estimated_angs, _ = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[0,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
    estimated_angs, _ = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[1,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
    estimated_angs, _ = MUSIC.estimate_doa(R, ula_st_vectors, K)
    rmse_matrix[2,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
    # CRLB (Placeholder)
    rmse_matrix[3, i] = 0  # Compute analytically based on Fisher information


# Plot RMSE as a function of SNR
plt.figure(figsize=(8, 6))
plt.plot(ang_separations, rmse_matrix[0,:], label="Bartlett", marker="o:r")
plt.plot(ang_separations, rmse_matrix[1,:], label="Capon", marker="x:g")
plt.plot(ang_separations, rmse_matrix[2,:], label="MUSIC", marker="d:b")
plt.plot(ang_separations, rmse_matrix[3,:], label="CRLB", marker="-k")
plt.title(f"RMSE as a Function of angular separation \n M={M}, d={d}, equal SNR= 0 dB, ss_size={N}")
plt.xlabel("Angular separation (in deg)")
plt.ylabel("RMSE (Degrees)")
plt.grid()
plt.legend()
plt.show()
