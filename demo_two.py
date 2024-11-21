import numpy as np
from .utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 2          | Low SNR/ Hign noise | 2 sources at -20° and 20°, SNR reduced to -40 dB, AWGN noise dominates signal,

M = 4
d = 0.5 # in wavelengths
N = 100  # sample size

inc_ang_deg = [-20, 20]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

# Generate source signals
soi = np.random.randn(K, N)   # Signal(s) of Interest

# Augment generated signals with the given SNR
snr = [-40,-40]
snr = np.asarray(snr) # (K,)
power = 10**(snr / 10) 
power = np.sqrt(power) 
power = np.diag(power) # (K x K)

soi = power @ soi # (K x K) @ (K x N)

A = ula_steering_matrix(M,d,thetas_rad) # (M x K)

soi_matrix = A @ soi # (M x K) @ (K x N) = (M x N)

# Generate multichannel uncorrelated noise
noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
noise = 0.5 * noise # split between Re and Im components.

# Create received signal
tx_signal = soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = (tx_signal @ tx_signal.conj().T)/tx_signal.shape[1]

# Generate steering vectors
ula_st_vectors = ula_scan_steering_matrix(M,0.5,angular_resolution=1)

# DOA estimation
estimates,Bartlett_PAD = BartlettBeamformer.estimate_doa(R, ula_st_vectors)
print("Bartlett_PAD estimates", estimates)
estimates,Capon_PAD = CaponBeamformer.estimate_doa(R, ula_st_vectors)
print("Capon_PAD estimates", estimates)
estimates,MUSIC_ORTAD = MUSIC.estimate_doa(R, ula_st_vectors)
print("MUSIC_ORTAD estimates", estimates)

DOA_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC"])
DOA_polar_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC"])