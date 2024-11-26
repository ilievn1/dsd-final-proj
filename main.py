import numpy as np
from .utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
# Scenarios for Testing DOA Estimation Performance
# The following table summarizes different test cases designed to explore the behavior of DOA estimators
# under various challenging conditions. Each scenario introduces a unique challenge by modifying
# the signal characteristics or the environment.

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 1          | Close angular spacing | 2 sources at 20° and 21°, SNR=20dB, AWGN
# 2          | Low SNR/ Hign noise | 2 sources at -20° and 20°, SNR reduced to -40 dB, AWGN noise dominates signal,
# 3          | High SNR/ Low noise | 2 sources at -20° and 20°, SNR reduced to 80 dB, AWGN 
# 4          | Unequal sig powers | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN
# 5          | Calibration errors | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN, steering vectors perturbed
# 6          | Correlated sources | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN, weakly correlated
# 7          | Correlated sources | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN, highly correlated
# 8          | Correlated noise | 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN highly correlated
# 9          | Off-grid resolution | 2 sources at -23.7° and 23.3°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN, 1 deg angular scan resolution
# 10         | Reduced number of samples | 20 samples instead of 100, 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN
# 11         | Multipath coherency | 1 source at -20°, SNR=20dB, reflection at 20°, AWGN
# 12         | Sparse array | 2-element ULA, less spatial information, models struggle with resolution
# 13         | Complex noise distributions | Noise with mixed Laplacian, Rayleigh, Gaussian, Exponential, etc., across channels

# TODO: Add after establishing stochastic and deterministic signal models
# 14         | Signals with harmonics | Primary signal with harmonics added, interferes with accurate estimation

# Each scenario explores different aspects of DOA estimation, including challenges with:
# - Low or high SNR
# - Overlapping sources or noise
# - Array geometry issues
# - Complex noise distributions
# - Multipath or harmonics

M = 4
d = 0.5 # in wavelengths
N = 100  # sample size

inc_ang_deg = [20, 21]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

# Generate source signals
soi = np.random.randn(K, N)   # Signal(s) of Interest

# Augment generated signals with the given SNR
snr = [20,20]
snr = np.asarray(snr) # (K,)
power = 10**(snr / 10) 
power = np.sqrt(power) 
power = np.diag(power) # (K x K)

soi = power @ soi # (K x K) @ (K x N)

A = ula_steering_matrix(M,d,thetas_rad) # (M x K)

steered_soi_matrix = A @ soi # (M x K) @ (K x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_signals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
noise = 0.5 * noise # split between Re and Im components.

# Create received signal
tx_signal = steered_soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = (tx_signal @ tx_signal.conj().T)/tx_signal.shape[1]

# Generate steering vectors
ula_st_vectors = ula_scan_steering_matrix(M,0.5,angular_resolution=1)

# DOA estimation
estimates,Bartlett_PAD = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
print("Bartlett_PAD estimates", estimates)
estimates,Capon_PAD = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
print("Capon_PAD estimates", estimates)
estimates,MUSIC_ORTAD = MUSIC.estimate_doa(R, ula_st_vectors, K)
print("MUSIC_ORTAD estimates", estimates)

DOA_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC"])
DOA_polar_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD], inc_ang_deg, labels=["Bartlett","Capon", "MUSIC"])


# ===============================================================================================
# Complex case
# Params:
# SNR = 1
# sig_numb = 3

M = 4
d = 0.5 # in wavelengths
N = 100  # sample size

inc_ang_deg = [-40, 20, 30]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

# Generate source signals
soi = np.random.randn(K, N)   # Signal(s) of Interest

elems = np.arange(M).reshape(-1,1)
a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x K)

steered_soi_matrix = a @ soi # (M x K) @ (K x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_signals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.normal(0,np.sqrt(1),(M,N)) + 1j*np.random.normal(0,np.sqrt(1),(M,N))

# Create received signal
tx_signal = steered_soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = (tx_signal @ tx_signal.conj().T)/tx_signal.shape[1]

# Generate steering vectors
ula_st_vectors = ula_scan_steering_matrix(M,0.5,angular_resolution=1)

# DOA estimation
Bartlett_PAD = BartlettBeamformer.calculate_spectrum(R, ula_st_vectors)
Capon_PAD = CaponBeamformer.calculate_spectrum(R, ula_st_vectors)
DOA_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])
DOA_polar_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])

# ===============================================================================================
# Coherent case
M = 4
d = 0.5 # in wavelengths
N = 100  # sample size

inc_ang_deg = [50, 80]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal   
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

# Generate multiple copies of same signal
soi = np.random.randn(1, N)   # Signal(s) of Interest
soi = np.repeat(soi,K,axis=0)

elems = np.arange(M).reshape(-1,1)
a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x K)

steered_soi_matrix = a @ soi # (M x K) @ (K x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_signals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.normal(0,np.sqrt(1),(M,N)) + 1j*np.random.normal(0,np.sqrt(1),(M,N))

# Create received signal
tx_signal = steered_soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = (tx_signal @ tx_signal.conj().T)/tx_signal.shape[1]

# Generate steering vectors        
ula_st_vectors = ula_scan_steering_matrix(M,0.5,angular_resolution=1)

# DOA estimation
Bartlett_PAD = BartlettBeamformer.calculate_spectrum(R, ula_st_vectors)
Capon_PAD = CaponBeamformer.calculate_spectrum(R, ula_st_vectors)
DOA_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])
DOA_polar_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])