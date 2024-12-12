import numpy as np
import matplotlib.pyplot as plt
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC
import ESPRIT

def rmse_var_ss_size(M = 8,d = 0.5, inc_ang_deg = [18, 25], snr = 10, T = 100, save_fig=True, fig_name='rmse_ss_size_demo'):
    num_trials = T
    
    thetas_deg = np.array(inc_ang_deg).reshape(1, -1)  # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1]  # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)
    
    snapshot_sizes = 2**np.arange(14)  # Snapshot sizes

    snr_linear = 10 ** (snr / 10)  # SNR in linear scale
    noise_power = 1 / snr_linear
    
    # Generate ULA steering matrix
    A = ula_steering_matrix(M, d, thetas_rad)  # (M x K)
    
    # Generate scanning steering matrix
    ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=1)  # (M x P)
    
    # RMSE storage
    rmse_matrix = np.zeros((6,len(snapshot_sizes)),dtype=np.complex_) # 5 + 1 = num of estimator, currently CBF,MVDR,MUSIC,RMUSIC,ESPRIT + CRLB
    
    # Monte Carlo trials using matrix operations
    for i, N in enumerate(snapshot_sizes):
      for t in range(num_trials):

        # Generate signals and noise
        soi = np.random.randn(K, N)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * np.sqrt(noise_power / 2)
    
        # Generate transmitted signal
        steered_soi_matrix =  A @ soi  # (M x N)
        tx_signal = steered_soi_matrix + noise  # (M x N x num_trials)
    
        # Covariance matrix estimation
        # R = np.einsum('imt,jmt->ijt', tx_signal, np.conj(tx_signal)) / N # (M x M x num_trials)
        R = cov(tx_signal)

        # Bartlett
        estimated_angs, _ = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[0, i] += np.sqrt(np.mean((thetas_deg - estimated_angs) ** 2))
        # Capon
        estimated_angs, _ = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[1, i] += np.sqrt(np.mean((thetas_deg - estimated_angs) ** 2))
        # MUSIC
        estimated_angs, _ = MUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[2, i] += np.sqrt(np.mean((thetas_deg - estimated_angs) ** 2))
        # ROOTMUSIC
        estimated_angs, _ = RMUSIC.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[3,i] += np.sqrt(np.mean((thetas_deg - estimated_angs) ** 2))
        # ESPRIT
        estimated_angs, _ = ESPRIT.estimate_doa(R, ula_st_vectors, K)
        rmse_matrix[4,i] += np.sqrt(np.mean((thetas_deg - estimated_angs) ** 2))
        
        # CRLB
        # Formula: P. Stoica, A. Nehorai "MUSIC, Maximum Likelihood, and Cramer-Rao Bound"
        # Sec. VII Eq. 7.7b
        da_dth = (A.T * 1j*np.arange(M)).T # (M,) * (M x K) || Hadamard prod w/o repeating d along all K cols
    
        h = da_dth.conj().T @ (np.eye(M) - A @ np.linalg.inv( A.conj().T @ A) @ A.conj().T) @ da_dth   
        
        var_crlb = (1/(2*N*snr_linear)) / np.diag(h) # (K,) diag extracts i-th est for i-th true inc angle
        rmse_matrix[5, i] += np.sqrt(np.mean(var_crlb))
        """
        # Small N formula CRLB
        # Formula: P. Stoica, A. Nehorai "MUSIC, Maximum Likelihood, and Cramer-Rao Bound"
        # Sec. IV Eq. 4.1
        da_dth = (A.T * 1j*np.arange(M)).T # (M,) * (M x K) || Hadamard prod w/o repeating d along all K cols        
        D = da_dth # (M x K)
        Pi_orth = (np.eye(M) - A @ np.linalg.inv( A.conj().T @ A) @ A.conj().T) # (M x M)
        X = np.eye(K)[None, :, :] * soi.T[:, :, None] # (ss_size x K x K)
        # As loop;
        # crlb = np.zeros((K, K), dtype=np.complex128)
        # for t in range(N):
        #     X_i = X[t]
        #     crlb += np.real(X_i.conj().T @ D.conj().T @ Pi_orth @ D @ X_i)
        # As tensor;
        crlb = np.einsum(
            'nij,jk,nkl->il',
            X.conj().transpose(0, 2, 1),
            D.conj().T @ Pi_orth @ D, # (K x K)
            X
        )
        crlb = np.real(crlb)    

        crlb = (np.sqrt(noise_power) / 2) * np.reciprocal (crlb)   
        crlb = np.diag(crlb)
        
        rmse_matrix[5, i] += np.sqrt(np.mean(crlb))
        """
    
    rmse_matrix /= num_trials  # Average over trials


    # Plot RMSE as a function of Snapshot Size
    plt.figure(figsize=(8, 6))
    plt.plot(snapshot_sizes, rmse_matrix[0,:], "o:r", label="Bartlett")
    plt.plot(snapshot_sizes, rmse_matrix[1,:], "o:g", label="Capon")
    plt.plot(snapshot_sizes, rmse_matrix[2,:], "d:b", label="MUSIC")
    plt.plot(snapshot_sizes, rmse_matrix[3,:], "x:m", label="ROOT")
    plt.plot(snapshot_sizes, rmse_matrix[4,:], "*:", label="ESPRIT")
    plt.plot(snapshot_sizes, rmse_matrix[5,:], "-k", label="CRLB")

    plt.title(f"RMSE as a Function of Snapshot Size")
    plt.suptitle(f"Uncorrelated M={M}, d={d}, inc_thetas={inc_ang_deg}, equal SNR={snr} dB, MC Trials = {num_trials}")
    plt.xlabel("Snapshot Size (N)")
    plt.ylabel("RMSE (Degrees)")
    plt.grid()
    plt.legend()
    if save_fig == True:
        if fig_name == None:
            fig_name = os.urandom(15).hex()
        plt.savefig(f'{fig_name}.eps', format='eps')
    plt.show()