import numpy as np
from utils import *
import BartlettBeamformer
import CaponBeamformer
import MUSIC
import RMUSIC
import ESPRIT

# Scenario # | Description | Key Modifications
# ----------------------------------------------
# 11         | Multipath coherency | 1 source at -20°, reflection at 20°,  SNR_1 is 0dB, SNR_2 is -5dB, AWGN

def demo_eleven(M = 8,d = 0.5,N = 100, figName = None):
    # Incident angles
    src_ang_deg = [-20]  # Main signal
    reflection_ang_deg = [20]  # Reflection
    src_ang_rad = np.deg2rad(np.array(src_ang_deg).reshape(1, -1)) # (1 x K) DOA of main signal
    reflection_rad = np.deg2rad(np.array(reflection_ang_deg).reshape(1, -1))
    K = src_ang_rad.shape[1]  # Number of signals

    # Signal parameters
    soi = np.random.randn(1, N)  # Main signal
    power = np.sqrt(10**(0 / 10))  # SNR for main signal in dB
    soi = power * soi

    # Reflection modeling: delayed, scaled, and phase-shifted version of main signal
    reflection_gain = np.sqrt(10**(-3/10))  # Reflection SNR: -5 dB
    reflection_delay = -1  # Delay in samples
    rand_ph_shift = scipy.stats.truncnorm((-np.pi/2), (np.pi/2), loc=0, scale=1).rvs(size=1)[0]
    reflection_phase = np.exp(1j * rand_ph_shift)  # Phase shift

    reflection_signal = (
        reflection_gain
        * np.roll(soi[0], reflection_delay)
        * reflection_phase
    ).reshape(1, -1)  # Ensure shape consistency

    # Steering matrix for the main sig
    A_main = ula_steering_matrix(M, d, src_ang_rad)  # (M x K)

    # Steering matrix for the reflection
    A_reflection = ula_steering_matrix(M, d, reflection_rad)

    # Combined steered signals (main + reflection)
    steered_soi_matrix = A_main @ soi + A_reflection @ reflection_signal  # (M x N)

    # Add multichannel AWGN noise
    noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    noise = 0.5 * noise  # Split power between real and imaginary parts
    tx_signal = steered_soi_matrix + noise  # Received signal

    # Covariance matrix w/o ss and fba
    R = cov(tx_signal)  # Main covariance matrix
    """    
    # Covariance matrix w/o ss
    R = cov(tx_signal,fba=True)
    """

    """
    # Covariance matrix 
    # R_main = cov(tx_signal,fba=True)
    # subarray_length = M - 1  # Length of subarrays for spatial smoothing
    # R = spatial_smoothing(R_main, subarray_length=subarray_length)
    """
    # Generate steering vectors for main array
    ula_st_vectors = ula_scan_steering_matrix(M, d, angular_resolution=1)
    # Generate steering vectors for subarray
    # ula_st_vectors = ula_scan_steering_matrix(subarray_length, d, angular_resolution=1)

    # DOA estimation
    _ ,Bartlett_PAD = BartlettBeamformer.estimate_doa(R, ula_st_vectors, K)
    _ ,Capon_PAD = CaponBeamformer.estimate_doa(R, ula_st_vectors, K)
    _ ,MUSIC_ORTAD = MUSIC.estimate_doa(R, ula_st_vectors, K)
    rm_estimates,_ = RMUSIC.estimate_doa(R, ula_st_vectors, K)
    esp_estimates,_ = ESPRIT.estimate_doa(R, ula_st_vectors, K)
    print("ROOT MUSIC estimates", rm_estimates)
    print("ESPRIT estimates", esp_estimates)
    DOA_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], src_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =figName)
    # DOA_polar_plot([Bartlett_PAD,Capon_PAD, MUSIC_ORTAD],[np.asarray(rm_estimates), np.asarray(esp_estimates)], src_ang_deg, labels=["Bartlett","Capon", "MUSIC", "ROOT", "ESPRIT"], save_fig=True, fig_name =f'{figName}_polar')
demo_eleven(figName="demo_eleven_fb")