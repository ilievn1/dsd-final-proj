import numpy as np
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC
import ESPRIT

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 9          | Off-grid resolution | 2 sources at -23.7° and 23.3°, SNR_1 is 15dB, SNR_1 is -15dB, AWGN, 1 deg angular scan resolution
def demo_nine(M = 8,d = 0.5,N = 100, figName = None):        

    
    inc_ang_deg = [-23.7, 23.3]
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
    K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
    thetas_rad = np.deg2rad(thetas_deg)
    
    # Generate source signals
    soi = np.random.randn(K, N)   # Signal(s) of Interest
    
    # Augment generated signals with the given SNR
    snr = np.asarray([15,-15]) # (K,)
    power = np.sqrt(10**(snr / 10)) 
    power = np.diag(power) # (K x K)
    
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
    R = cov(tx_signal)
    
    # Generate steering vectors
    ula_st_vectors = ula_scan_steering_matrix(M,d,angular_resolution=5)
    
    # DOA estimation
    _ ,Bartlett_PAD = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
    _ ,Capon_PAD = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
    _ ,MUSIC_ORTAD = MUSIC.estimate_doa(R, ula_st_vectors, K)
    rm_estimates,_ = RMUSIC.estimate_doa(R, ula_st_vectors, K)
    esp_estimates,_ = ESPRIT.estimate_doa(R, ula_st_vectors, K)
    print("ROOT MUSIC estimates", rm_estimates)
    print("ESPRIT estimates", esp_estimates)
    DOA_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =figName)
    DOA_polar_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =f'{figName}_polar')
