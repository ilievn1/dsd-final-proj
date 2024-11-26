import numpy as np
import matplotlib.pyplot as plt
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC

def rmse_var_ang_res():
    # Parameters
    M = 4  # Number of elements in ULA
    d = 0.5  # Element spacing in wavelengths
    N = 100  # Number of snapshots
    num_trials = 100
    
    inc_ang_deg = [20, 23]
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)
    
    snr = 10  # Fixed SNR in dB
    snr_linear = 10 ** (snr / 10)  # SNR in linear scale
    noise_power = 1 / snr_linear
    
    ang_separations = np.arange(0, 30, 0.1) # in deg
    
    # RMSE storage
    rmse_matrix = np.zeros(5,len(ang_separations)) # 4 + 1 = num of estimator, currently CBF,MVDR,MUSIC,RMUSIC + CRLB
    
    # Monte Carlo trials using matrix operations
    for i, sep in enumerate(ang_separations):
        inc_ang_deg = [20, 20 + sep]
        thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
        K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
        thetas_rad = np.deg2rad(thetas_deg)
    
        # Generate ULA steering matrix
        A = ula_steering_matrix(M, d, thetas_rad)  # (M x K)
    
        # Generate scanning steering matrix
        ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=0.1 + sep)  # (M x P)
    
        # Generate signals and noise
        soi = np.random.randn(K, N, num_trials)
        noise = (np.random.randn(M, N, num_trials) + 1j * np.random.randn(M, N, num_trials)) * np.sqrt(noise_power / 2)
    
        steered_soi_matrix = np.tensordot(A, soi, axes=(1, 0))
        tx_signal = steered_soi_matrix + noise  # (M x N x num_trials)
    
        # Covariance matrix estimation
        R = np.tensordot(tx_signal, np.conj(tx_signal), axes=(1, 1)) / N  # (M x N x num_trials)
    
        # Bartlett
        estimated_angs, _ = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[0,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # Capon
        estimated_angs, _ = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[1,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # MUSIC
        estimated_angs, _ = MUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[2,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        # ROOTMUSIC
        estimated_angs, _ = RMUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[3,i] = np.sqrt(np.mean((inc_ang_deg - estimated_angs) ** 2))
        
        # CRLB
        # Formula: P. Stoica, A. Nehorai "MUSIC, Maximum Likelihood, and Cramer-Rao Bound"
        # Sec. VII Eq. 7.7b
        d = (A.T * 1j*np.arange(M)).T # (M,) * (M x K) || Hadamard prod w/o repeating d along all K cols
    
        h = d.conj().T @ (np.eye(M) - A @ np.linalg.inv( A.conj().T @ A) @ A.conj().T) @ d   
        
        var_crlb = (1/(2*N*snr_linear)) / np.diag(h) # (K,) diag extracts i-th est for i-th true inc angle
        
        rmse_matrix[4, i] = np.sqrt(np.mean(var_crlb))
    
    
    # Plot RMSE as a function of SNR
    plt.figure(figsize=(8, 6))
    plt.plot(ang_separations, rmse_matrix[0,:], label="Bartlett", marker="o:r")
    plt.plot(ang_separations, rmse_matrix[1,:], label="Capon", marker="x:g")
    plt.plot(ang_separations, rmse_matrix[2,:], label="MUSIC", marker="d:b")
    plt.plot(ang_separations, rmse_matrix[3,:], label="ROOT", marker="+:m")
    plt.plot(ang_separations, rmse_matrix[4,:], label="CRLB", marker="-k")
    
    plt.title(f"RMSE as a Function of angular separation")
    plt.suptitle(f"Uncorrelated M={M}, d={d}, equal SNR = {snr} dB, ss_size={N}")
    plt.xlabel("Angular separation (in deg)")
    plt.ylabel("RMSE (Degrees)")
    plt.grid()
    plt.legend()
    plt.show()
