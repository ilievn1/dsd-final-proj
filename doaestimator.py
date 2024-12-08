from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class DOAEstimator(ABC):
    """
    Abstract base class for DOA estimation algorithms.

    Attributes:
        array (Array): The array configuration.
        num_sources (int): Number of signal sources.
        signal (Signal): The signal instance containing frequency, angles, and noise.
    """

    def __init__(self, array, num_sources: int, signal):
        """
        Initializes DOA estimator with array configuration and signal parameters.

        Args:
            array (Array): The array configuration.
            num_sources (int): Number of sources in the signal.
            signal (Signal): Signal instance containing sampling information and DOA angles.
        """
        self.array = array
        self.num_sources = num_sources
        self.signal = signal


    @classmethod
    @abstractmethod
    def calculate_spectrum(cls, R, steering_vectors, **kwargs):
        """
        Parameters:
            :param R: spatial covariance matrix
            :param steering_vectors : Generated using the array alignment and the incident angles
            :param K: expected signal count

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :type steering vectors: 2D numpy array with size: M x P, where P is the number of scanning angles
            :type K: int

        Return values:
            :return spectrum: spectrum (e.g. angular distribution of pwr or noise subspace orthogonality)
            :rtype spectrum: 1D ndarray (P,)
        """
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")
        
        pass

    @classmethod
    def estimate_doa(cls, R, steering_vectors, K, full_spectrum_peaks = False):
        """
        Parameters:
            :param R: spatial covariance matrix
            :param steering_vectors : Generated using the array alignment and the incident angles
            :param K: expected signal count
            :param full_spectrum_peaks: look for peaks in [-180,180] deg range, else look for peaks in [-90,90] deg range
            
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :type steering vectors: 2D numpy array with size: M x P, where P is the number of scanning angles
            :type K: int
            :type full_spectrum_peaks: bool

        Return values:
            :return DOA: DOA estimates (in deg)
            :return spectrum: spectrum (e.g. angular distribution of pwr or noise subspace orthogonality)
            :rtype DOA: 1D ndarray (K,)
            :rtype spectrum: 1D ndarray (P,)
        """
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")

        s = steering_vectors

        spectrum = cls.calculate_spectrum(R, s, K=K) # [-pi,pi] spectrum
        r = 360.0/(len(spectrum) - 1 )
        scan_thetas_deg = np.arange(-180, 180 + r, r)

        # Find all peaks in [-pi,pi]
        max_idxs, _ = scipy.signal.find_peaks(spectrum)
        max_idxs = sorted(max_idxs, key=lambda idx: spectrum[idx],reverse=True)[:K]

        if full_spectrum_peaks == False:
          left_idx = np.argmin(np.abs(scan_thetas_deg - (-90)))
          right_idx = np.argmin(np.abs(scan_thetas_deg - 90))
          scan_thetas_deg = scan_thetas_deg[left_idx:right_idx+1]
          half_range_sp = spectrum[left_idx:right_idx+1]
          max_idxs, _ = scipy.signal.find_peaks(half_range_sp)
          max_idxs = sorted(max_idxs, key=lambda idx: half_range_sp[idx],reverse=True)[:K]

        estimated_DOAs = np.sort(scan_thetas_deg[max_idxs]) # Sort left to right
        
        return estimated_DOAs, spectrum
