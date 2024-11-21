import numpy as np
import matplotlib.pyplot as plt

def ula_steering_matrix(num_elements:int, element_spacing:float, thetas_rad: list[float]):
    """
    This function prepares steering vectors for ULA antenna systems
        
    Parameters:
        :param num_elements (int): number of ULA antenna elements.
        :param element_spacing (float): interelement spacing as fraction of a wavelength
        :param thetas_rad ([float]): angles (in radians) for which set of steering column vectors is generated
        
    Return values:
        :return A : steering matrix
        :rtype A: 2D numpy array with size: M x K, where M is the number of array elements, and where K is the number of signals
        
    """
    
    M,d = num_elements, element_spacing
    elems = np.arange(M).reshape(-1,1)
    A = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x K)

    return A

def ula_scan_steering_matrix(num_elements:int, element_spacing:float, angular_resolution:float = None, num_st_vecs:int = None):
    """
    This function prepares steering vectors for ULA antenna systems
        
    Parameters:
        :param num_elements (int): number of ULA antenna elements.
        :param element_spacing (float): interelement spacing as fraction of a wavelength
        :param angular_resolution (float): interval between two angles when sweeping
        :param num_st_vecs (int): number of points b/w [-180 180]
        
    Return values:
        :return steering_vectors : Estimated signal dimension
        :rtype A: 2D numpy array with size: M x P, where M is the number of array elements, and where P is the number of scanning angles
        
    """
    if angular_resolution and num_st_vecs:
       raise TypeError("Either angular_resolution or num_st_vecs can be supplied one at a time")
    
    M,d,r,p=num_elements, element_spacing, angular_resolution, num_st_vecs
    scan_thetas_deg = np.arange(-180, 180 + r, r) if r != None else np.linspace(-180, 180, p) 
    scan_thetas_rad = np.deg2rad(scan_thetas_deg).reshape(1,-1)
    arr_alignment = np.arange(M).reshape(-1,1)
    return np.exp(-1j *2* np.pi * d * arr_alignment @ np.sin(scan_thetas_rad))


def DOA_plot(spectra_data:list[np.ndarray], inc_angs:list[float], labels:list[str]=[], fullView=False, log_scale=True, normalize=True):
    """
    This plotting function takes in a number of spectrums from various methods (e.g. CAPON and MUSIC) and linearly plots DOA estimates and actual DOAs
        
    Parameters:
        :param spectra_data ([np.ndarray]): an array of one or more spectrums used for DOA estimations.
        :param inc_angs ([float]): originating angles (in deg) of sources used for validation of estimates
        :param labels ([str]): labels for legend in plot to map spectrum to algorithm which generated it
        :param fullView (bool): True: plot angles b/w [-180,180] degrees, else [-90,90] degrees
        :param log_scale (bool): True: plot spectra y values in dB, else watts
        :param normalize (bool): True: subtract max spectrum y value from all spectrum y values
        
    """    
    # Input check
    if not isinstance(spectra_data,list):
      raise TypeError("spectra_data must be Python list")

    if any((ang < -180) or (ang > 180) for ang in inc_angs):
      raise TypeError("Incident angles range is b/w [-180,180] degrees")

    if len(spectra_data) > 1 and len(spectra_data) != len(labels):
      raise TypeError("Number of spectra_data and labels must be equal")

    # Preprocess and format
    spectra_data = np.concatenate(spectra_data,axis=0)

    if(log_scale == True):
      spectra_data = 10*np.log10(spectra_data)

    if(normalize == True):
      spectra_data = (spectra_data.T - np.max(spectra_data, axis = 1)).T
    
    # angular_resolution based on count of spectra_data spectrum measurements
    r = 360.0/(len(spectra_data[0,:]) - 1 )

    scan_thetas_deg = np.arange(-180, 180 + r, r)

    if fullView == False:
      left_idx = np.argmin(np.abs(scan_thetas_deg - (-90)))
      right_idx = np.argmin(np.abs(scan_thetas_deg - 90))
      scan_thetas_deg = scan_thetas_deg[left_idx:right_idx+1]
      spectra_data = spectra_data[:,left_idx:right_idx+1]

    #Plot DOA results
    fig = plt.figure()
    axes  = fig.add_subplot(111)
    for i in range(spectra_data.shape[0]):
        axes.plot(scan_thetas_deg, spectra_data[i,:].squeeze(),label=labels[i])

    axes.set_title('Direction of Arrival estimation ',fontsize = 16)
    axes.set_xlabel('Incident angle [deg]')
    axes.set_ylabel('Amplitude [watt]')
    if(log_scale == True):
        axes.set_ylabel('Amplitude [dB]')
    
    # Mark source(s) actual DOAs
    for ang in inc_angs:
        axes.axvline(linewidth = 2,color = 'green',x = ang)

    plt.legend()
    plt.grid()
    plt.show()
   
def DOA_polar_plot(spectra_data:list[np.ndarray], inc_angs:list[float], labels:list[str]=[], fullView=False, log_scale=True, normalize=True):
    """
    This plotting function takes in a number of spectrums from various methods (e.g. CAPON and MUSIC) and radially plots DOA estimates and actual DOAs
        
    Parameters:
        :param spectra_data ([np.ndarray]): an array of one or more spectrums used for DOA estimations.
        :param inc_angs ([float]): originating angles (in deg) of sources used for validation of estimates
        :param labels ([str]): labels for legend in plot to map spectrum to algorithm which generated it
        :param fullView (bool): True: plot angles b/w [-180,180] degrees, else [-90,90] degrees
        :param log_scale (bool): True: plot spectra y values in dB, else watts
        :param normalize (bool): True: subtract max spectrum y value from all spectrum y values
    """    
    # Input check
    if not isinstance(spectra_data,list):
      raise TypeError("spectra_data must be Python list")

    if any((ang < -180) or (ang > 180) for ang in inc_angs):
      raise TypeError("Incident angles range is b/w [-180,180] degrees")

    if len(spectra_data) > 1 and len(spectra_data) != len(labels):
      raise TypeError("Number of spectra_data and labels must be equal")

    # Preprocess and format
    spectra_data = np.concatenate(spectra_data,axis=0) # (S x P) 

    if(log_scale == True):
      spectra_data = 10*np.log10(spectra_data)

    if(normalize == True):
      spectra_data = (spectra_data.T - np.max(spectra_data, axis = 1)).T
    
    # angular_resolution based on count of spectra_data spectrum measurements
    r = 360.0/(len(spectra_data[0,:]) - 1 )

    scan_thetas_deg = np.arange(-180, 180 + r, r)

    if fullView == False:
      left_idx = np.argmin(np.abs(scan_thetas_deg - (-90)))
      right_idx = np.argmin(np.abs(scan_thetas_deg - 90))
      scan_thetas_deg = scan_thetas_deg[left_idx:right_idx+1]
      spectra_data = spectra_data[:,left_idx:right_idx+1]

    #Plot DOA results
    fig, axes = plt.subplots(subplot_kw={'projection': 'polar'})
    
    for i in range(spectra_data.shape[0]):
        axes.plot(np.deg2rad(scan_thetas_deg), spectra_data[i,:].squeeze(),label=labels[i]) # MAKE SURE TO USE RADIAN FOR POLAR

    axes.set_title('Direction of Arrival estimation ',fontsize = 16)
    axes.set_xlabel('Incident angle [deg]')
    axes.set_ylabel('Amplitude [watt]')
    if(log_scale == True):
        axes.set_ylabel('Amplitude [dB]')
            
    # Mark source(s) actual DOAs
    for ang in inc_angs:
        axes.axvline(linewidth = 2,color = 'green',x = np.deg2rad(ang))

    axes.set_theta_zero_location('N') # make 0 degrees point up
    axes.set_theta_direction(-1) # increase clockwise
    axes.set_rlabel_position(55)  # Move grid labels away from other labels
    
    if fullView == False:
        axes.set_thetamin(-90) # only show top half
        axes.set_thetamax(90)
    plt.legend()
    plt.grid()
    plt.show()


# TODO: steering vec gen dependent on ULA geometry, generalize
""" 
TODO: Make possible to add different types of Noise (to be defined e.g. Rician, Laplace, Gauss, Rayleigh)
      Make possible to add different types of Noise to different signals and of different intensities.
"""
def generate_signal_array(num_elements=3, num_snapshots=10000, inc_ang_deg:list[int]=[20], d=0.5, coherent=False, SNR_dB=20):
    """
    Generates a multichannel test signal with desired SNR.
    
    Parameters:
        :param num_elements (int): Number of array elements.
        :param num_snapshots (int): Number of time samples.
        :param inc_ang_deg (list of float): Incident angles in degrees for each source.
        :param d (float): Element spacing in wavelengths (default 0.5).
        :param freq_sampling (float): Sampling frequency in Hz.
        :param SNR_dB (float): Desired signal-to-noise ratio in dB.

    Returns:
        :return tx_signal (np.ndarray): Received signal array with noise.
    """

    # Define basic signal parameters
    num_sources = len(inc_ang_deg)
    thetas_deg=np.array(inc_ang_deg).reshape(1,-1) # (1 x K)
    thetas_rad = np.deg2rad(thetas_deg) # (1 x K)
    
    # Generate std norm signals for each source
    soi_matrix = np.random.randn(num_sources,num_snapshots) + 1j * np.random.randn(num_sources,num_snapshots) # (K x N)
    
    if coherent:
        soi_matrix = np.random.randn(1,num_snapshots) + 1j * np.random.randn(1,num_snapshots) # (1 x N)
        soi_matrix = np.repeat(soi_matrix, num_sources, axis = 0) # (K x N)
    
    elems = np.arange(num_elements).reshape(-1,1)

    # Create the steering vec for each source and form steering matrix
    a = np.exp(-1j * 2 * np.pi * d * elems @ np.sin(thetas_rad)) # (M x K)

    # Calculate signal power and scale noise for the desired SNR
    # Here noise power is adjusted by keeping sig pwr const and varying SNR
    signal_power = 1 # Variance of randn
    noise_power = signal_power / (10**(SNR_dB / 10))

    # Generate noise with calculated power
    noise = np.random.normal(0, np.sqrt(noise_power/2), (num_elements, num_snapshots)) + 1j * np.random.normal(0, np.sqrt(noise_power/2), (num_elements, num_snapshots))

    # Received signal with noise
    tx_signal = a @ soi_matrix + noise # (M x K) @ (K x N) + (M x N) = (M x N)
    
    return (soi_matrix, tx_signal)

# Example parameters
num_elements = 8       # Number of array elements
num_snapshots = 1024   # Number of time samples
angles = [20, 30, -40] # Incident angles in degrees
SNR_dB = 15            # Desired SNR in dB

# Generate received signal
received_signal = generate_signal_array(num_elements, num_snapshots, angles, SNR_dB=SNR_dB)

print(received_signal.shape)  # Should be (num_elements, num_snapshots)