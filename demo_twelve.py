
import numpy as np
from .utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 12         | Complex noise distributions | Noise with mixed Laplacian, Rayleigh, Gaussian, Exponential, etc., across channels

M = 6
d = 0.5 # in wavelengths
N = 100  # sample size

inc_ang_deg = [-20, 20]
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x K) Incident angles of test signal
K = thetas_deg.shape[1] # K MUST BE < M - 1 FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

# Generate source signals
soi = np.random.randn(K, N)   # Signal(s) of Interest

# Augment generated signals with the given SNR
snr = [0,0]
snr = np.asarray(snr) # (K,)
power = 10**(snr / 10) 
power = np.sqrt(power) 
power = np.diag(power) # (K x K)

soi = power @ soi # (K x K) @ (K x N)

A = ula_steering_matrix(M,d,thetas_rad) # (M x K)

steered_soi_matrix = A @ soi # (M x K) @ (K x N) = (M x N)

# Generate noise for each channel with different distributions
noise = np.zeros((M, N), dtype=complex)
distributions = ['gaussian', 'laplacian', 'rician', 'rayleigh', 'exponential', 'custom']

for m in range(M):
    noise_power = 1
    # sqrt (np / 2) splits between Re and Im components.
    if distributions[m] == 'gaussian':
        noise[m, :] = np.sqrt(noise_power) * (np.random.randn(N) + 1j * np.random.randn(N))
    elif distributions[m] == 'laplacian':
        real = np.random.laplace(0, np.sqrt(noise_power / 2), N)
        imag = np.random.laplace(0, np.sqrt(noise_power / 2), N)
        noise[m, :] = real + 1j * imag
    elif distributions[m] == 'rician':
        v = np.sqrt(noise_power / 2)
        noise[m, :] = np.random.normal(v, v, N) + 1j * np.random.normal(v, v, N)
    elif distributions[m] == 'rayleigh':
        scale = np.sqrt(noise_power / 2)
        magnitude = np.random.rayleigh(scale, N)
        phase = np.random.uniform(0, 2 * np.pi, N)
        noise[m, :] = magnitude * np.exp(1j * phase)
    elif distributions[m] == 'exponential':
        scale = np.sqrt(noise_power / 2)
        real = np.random.exponential(scale, N)
        imag = np.random.exponential(scale, N)
        noise[m, :] = real + 1j * imag
    elif distributions[m] == 'custom':
        real = np.random.normal(0, np.sqrt(noise_power / 2), N)
        imag = np.random.uniform(-np.sqrt(noise_power), np.sqrt(noise_power), N)
        noise[m, :] = real + 1j * imag


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


