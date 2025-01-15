import numpy as np
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC
import ESPRIT

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 7          | Correlated sources | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_1 is -15dB, AWGN, highly correlated

def demo_seven(M = 8,d = 0.5,N = 100, figName = None):        
    inc_ang_deg = [-50,-30,-10]
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)

    Bartlett_PAD_list = []
    Capon_PAD_list = []
    MUSIC_ORTAD_list = []
    rm_estimates_list = []
    esp_estimates_list = []
    for t in range(20):
      # Generate source signals
      soi = np.random.randn(K, N)   # Signal(s) of Interest

      # Augment generated signals with the given SNR
      snr = np.asarray([0,-5,-10]) # (K,)
      power = np.sqrt(10**(snr / 10))
      power = np.diag(power) # (K x K)

      #Correlate using Cholesky
      correlation_matrix = np.random.uniform(low=0.97, high=0.99, size=(K,K))  # High correlation
      np.fill_diagonal(correlation_matrix, 1)  # Set diagonal elements to 1
      soi = np.linalg.cholesky(correlation_matrix).dot(soi)

      soi = power @ soi # (K x K) @ (K x N)

      A = ula_steering_matrix(M,d,thetas_rad) # (M x K)

      steered_soi_matrix = A @ soi # (M x K) @ (K x N) = (M x N)

      # Generate multichannel uncorrelated noise
      noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
      noise = 0.5 * noise # split between Re and Im components.

      # Create received signal
      tx_signal = steered_soi_matrix + noise

      # R matrix calculation
      # outside lib methds to allow different ways of calculating and augmenting

      R_main = cov(tx_signal)
      R_ss = spatial_smoothing(R_main, subarray_length=(M-1))
      # Generate steering vectors
      ula_st_vectors = ula_scan_steering_matrix((M-1),d,angular_resolution=1)

      # DOA estimation
      _ ,Bartlett_PAD = BartlettBeamformer.estimate_doa(R_ss, ula_st_vectors, K)
      _ ,Capon_PAD = CaponBeamformer.estimate_doa(R_ss, ula_st_vectors, K)
      _ ,MUSIC_ORTAD = MUSIC.estimate_doa(R_ss, ula_st_vectors, K)
      rm_estimates,_ = RMUSIC.estimate_doa(R_ss, ula_st_vectors, K)
      esp_estimates,_ = ESPRIT.estimate_doa(R_ss, ula_st_vectors, K)
      Bartlett_PAD_list.append(Bartlett_PAD.reshape(1,-1))
      Capon_PAD_list.append(Capon_PAD.reshape(1,-1))
      MUSIC_ORTAD_list.append(MUSIC_ORTAD.reshape(1,-1))
      rm_estimates_list.append(rm_estimates.reshape(1,-1))
      esp_estimates_list.append(esp_estimates.reshape(1,-1))

    Bartlett_PAD = np.concatenate( Bartlett_PAD_list, axis=0 ).mean(axis=0)
    Capon_PAD = np.concatenate( Capon_PAD_list, axis=0 ).mean(axis=0)
    MUSIC_ORTAD = np.concatenate( MUSIC_ORTAD_list, axis=0 ).mean(axis=0)
    rm_estimates = np.concatenate( rm_estimates_list, axis=0 ).mean(axis=0)
    esp_estimates = np.concatenate( esp_estimates_list, axis=0 ).mean(axis=0)

    DOA_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =figName)
    DOA_polar_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =f'{figName}_polar')
