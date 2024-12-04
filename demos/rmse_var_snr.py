import numpy as np
import matplotlib.pyplot as plt
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC
import ESPRIT

def rmse_var_snr():
    # Parameters
    M = 4  # Number of elements in ULA
    d = 0.5  # Element spacing in wavelengths
    N = 100  # Number of snapshots
    num_trials = 1000

    inc_ang_deg = [20, 23]
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)

    SNRs = np.linspace(-40, 20, 10)  # SNR range in dB

    # Generate ULA steering matrix
    A = ula_steering_matrix(M, d, thetas_rad)  # (M x K)

    # Generate scanning steering matrix
    ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=1)  # (M x P)

    # RMSE storage
    rmse_matrix = np.zeros((5,len(SNRs)),dtype=np.complex_) # 4 + 1 = num of estimator, currently CBF,MVDR,MUSIC,RMUSIC + CRLB

    # Monte Carlo trials using matrix operations
    for i, snr in enumerate(SNRs):
      for t in range(num_trials):

        snr_linear = 10 ** (snr / 10)
        noise_power = 1 / snr_linear

        # Generate signals and noise
        soi = np.random.randn(K, N)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(noise_power / 2)

        steered_soi_matrix =  A @ soi  # (M x N)
        tx_signal = steered_soi_matrix + noise  # (M x N x num_trials)

        # Covariance matrix estimation
        # R = np.einsum('imt,jmt->ijt', tx_signal, np.conj(tx_signal)) / N # (M x M x num_trials)
        R = cov(tx_signal)

        # Bartlett
        estimated_angs, _ = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[0,i] += np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # Capon
        estimated_angs, _ = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[1,i] += np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # MUSIC
        estimated_angs, _ = MUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[2,i] += np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # ROOTMUSIC
        estimated_angs, _ = RMUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[3,i] += np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))

        # CRLB
        # Formula: P. Stoica, A. Nehorai "MUSIC, Maximum Likelihood, and Cramer-Rao Bound"
        # Sec. VII Eq. 7.7b
        da_dth = (A.T * 1j*np.arange(M)).T # (M,) * (M x K) || Hadamard prod w/o repeating d along all K cols

        h = da_dth.conj().T @ (np.eye(M) - A @ np.linalg.inv( A.conj().T @ A) @ A.conj().T) @ da_dth   

        var_crlb = (1/(2*N*snr_linear)) / np.diag(h) # (K,) diag extracts i-th est for i-th true inc angle

        rmse_matrix[4, i] += np.sqrt(np.mean(var_crlb))

    rmse_matrix /= num_trials  # Average over trials

    # Plot RMSE as a function of SNR
    plt.figure(figsize=(8, 6))
    plt.plot(SNRs, rmse_matrix[0,:], "o:r", label="Bartlett")
    plt.plot(SNRs, rmse_matrix[1,:], "o:g", label="Capon")
    plt.plot(SNRs, rmse_matrix[2,:], "d:b", label="MUSIC")
    plt.plot(SNRs, rmse_matrix[3,:], "x:m", label="ROOT")
    plt.plot(SNRs, rmse_matrix[4,:], "-k", label="CRLB")
    plt.title(f"RMSE as a Function of SNR")
    plt.suptitle(f"Uncorrelated M={M}, d={d}, inc_thetas={inc_ang_deg}, ss_size={N}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE (Degrees)")
    plt.grid()
    plt.legend()
    plt.show()
