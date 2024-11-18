import numpy as np
from .utils import *
import BartlettBeamformer
import CaponBeamformer
# ===============================================================================================
# Simple case
# Params: 
# SNR = 1
# sig_numb = 1

M = 16
d = 0.5 # in wavelengths
N = 10000  # sample size

inc_ang_deg = [20]

# Generate source signal
soi = np.random.randn(1, N)   # Signal(s) of Interest
   
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x P) Incident angles of test signal   
P = thetas_deg.shape[1] # P MUST BE < M FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

elems = np.arange(M).reshape(-1,1)
a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x P)

soi_matrix = a @ soi # (M x P) @ (P x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_singals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.normal(0,np.sqrt(1),(M,N)) + 1j*np.random.normal(0,np.sqrt(1),(M,N))

# Create received signal
tx_signal = soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = np.cov(tx_signal)

# Generate steering vectors        
ula_st_vectors = ula_steering_vectors(M,0.5,num_st_vecs=1000)

# DOA estimation
Bartlett_PAD = BartlettBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)
Capon_PAD = CaponBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)

DOA_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])


# ===============================================================================================
# Complex case
# Params: 
# SNR = 1
# sig_numb = 3

M = 16
d = 0.5 # in wavelengths
N = 10000  # sample size

inc_ang_deg = [-40, 20, 30]

thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x P) Incident angles of test signal   
P = thetas_deg.shape[1] # P MUST BE < M FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

elems = np.arange(M).reshape(-1,1)
a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x P)

# Generate source signal
soi = np.random.randn(P, N)   # Signal(s) of Interest
   
soi_matrix = a @ soi # (M x P) @ (P x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_singals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.normal(0,np.sqrt(1),(M,N)) + 1j*np.random.normal(0,np.sqrt(1),(M,N))

# Create received signal
tx_signal = soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = np.cov(tx_signal)

# Generate steering vectors        
ula_st_vectors = ula_steering_vectors(M,0.5,num_st_vecs=1000)

# DOA estimation
Bartlett_PAD = BartlettBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)
Capon_PAD = CaponBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)
DOA_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])


# ===============================================================================================
# Coherent case
M = 16
d = 0.5 # in wavelengths
N = 10000  # sample size

inc_ang_deg = [-40, 20]

# Generate multiple copies of same signal
soi = np.random.randn(1, N)   # Signal(s) of Interest
soi = np.repeat(soi,2,axis=0)
   
thetas_deg=np.array(inc_ang_deg).reshape(1,-1)   # (1 x P) Incident angles of test signal   
P = thetas_deg.shape[1] # P MUST BE < M FOR CORRECT DETECTION
thetas_rad = np.deg2rad(thetas_deg)

elems = np.arange(M).reshape(-1,1)
a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x P)

soi_matrix = a @ soi # (M x P) @ (P x N) = (M x N)

# Generate multichannel uncorrelated noise
"""
Disclaimer!
Noise after phase shifting samples:
For AWGN, noise can be added to received signal after samples are phase-shifted and act as sensor noise
then "noise" var dimensions need be (num_sensors x num_samples).

Noise prior to phase shifting samples:
For AWGN, noise can be added to received signal prior to phase-shifting and act as environmental noise
then "noise" var dimensions need be (num_singals x num_samples).
We can do this because AWGN with a phase shift applied is still AWGN.
"""
noise = np.random.normal(0,np.sqrt(1),(M,N)) + 1j*np.random.normal(0,np.sqrt(1),(M,N))

# Create received signal
tx_signal = soi_matrix + noise

# R matrix calculation
# outside lib methds to allow different ways of calculating and augmenting
R = np.cov(tx_signal)

# Generate steering vectors        
ula_st_vectors = ula_steering_vectors(M,0.5,num_st_vecs=1000)

# DOA estimation
Bartlett_PAD = BartlettBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)
Capon_PAD = CaponBeamformer.calculate_pwr_ang_dens(R, ula_st_vectors)
DOA_plot([Bartlett_PAD,Capon_PAD], inc_ang_deg, labels=["Bartlett","Capon"])
