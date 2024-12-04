from demos import *
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
# 10         | Reduced number of samples | 10 samples instead of 100, 2 sources at -20° and 20°, SNR_1 is 15dB, SNR_2 is -15dB, AWGN
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

demo_one(M = 4,d = 0.5,N = 100, figName = "demo_one")
