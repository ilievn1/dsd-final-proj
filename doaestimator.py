from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
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
        print(f'Inside @classmethod calculate_spectrum with cls ${cls}')
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
            :rtype DOA: [float]
            :rtype spectrum: 1D ndarray (P,)
        """
        print(f'Inside @classmethod estimate_doa with cls ${cls}')
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")

        s = steering_vectors

        spectrum = cls.calculate_spectrum(R, s, K=K)
        r = 360.0/(len(spectrum) - 1 )
        scan_thetas_deg = np.arange(-180, 180 + r, r)

        # Find K peaks
        max_idxs = np.argpartition(spectrum, -K)[-K:]
        
        if full_spectrum_peaks == False:
          left_idx = np.argmin(np.abs(scan_thetas_deg - (-90)))
          right_idx = np.argmin(np.abs(scan_thetas_deg - 90))
          scan_thetas_deg = scan_thetas_deg[left_idx:right_idx+1]
          max_idxs = np.argpartition(spectrum[left_idx:right_idx+1], -K)[-K:]
                  
        estimated_DOAs = sorted(scan_thetas_deg[max_idxs])
        return estimated_DOAs, spectrum
