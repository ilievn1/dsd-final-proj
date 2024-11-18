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

    @abstractmethod
    def calculate_weights(self, steering_vectors,**kwargs):
        """
        Parameters:
            :param R: spatial covariance matrix
            :param steering_vectors : Generated using the array alignment and the incident angles

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :type steering vectors: 2D numpy array with size: M x P, where P is the number of incident angles

        Return values:
            :return weight_matrix: Angular distribution of the power
            :rtype weight_matrix: 2D numpy array with size: M x P, where P is the number of incident angles

        """
        pass

    @abstractmethod
    def calculate_pwr_ang_dens(self, R, steering_vectors):
        """
        Parameters:
            :param R: spatial covariance matrix
            :param steering_vectors : Generated using the array alignment and the incident angles

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :type steering vectors: 2D numpy array with size: M x P, where P is the number of incident angles

        Return values:
            :return PAD: Angular distribution of the power
            :rtype PAD: 2D numpy array with size: 1 x P, where P is the number of incident angles

        """
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")

        pass

    def estimate_doa(self, R, steering_vectors):
        """
        Parameters:
            :param R: spatial covariance matrix
            :param steering_vectors : Generated using the array alignment and the incident angles

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :type steering vectors: 2D numpy array with size: M x P, where P is the number of incident angles

        Return values:
            :return DOA: Angle of arrival (in deg)
            :rtype DOA: float

        """
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")


        PAD = self.calculate_pwr_ang_dens(R, steering_vectors)
        r = 360.0/(len(PAD) - 1 )
        scan_thetas_deg = np.arange(-180, 180 + r, r)
        return scan_thetas_deg[np.argmax(PAD)]


    def calculate_mse(self, s, s_hat, plot=False):
        """
        Parameters:
            :param s: source signal w/o shifting nor noise
            :param s_hat: estimate of s after beamformer weights applied
            :param plot: visualize MSE b/w original and reconstructed signal

            :type s: 2D numpy array with size of 1 x N, where N is the number of signal samples
            :type s_hat: 2D numpy array with size of 1 x N, where N is the number of signal samples
            :type plot: bool

        Return values:
            :return MSE: difference b/w actual and reconstructed signal samples squared and averaged
            :rtype MSE: float
        """
        # --> Input check
        if s.shape[1] != s_hat.shape[1]:
            raise TypeError("original and reconstructed signal vector different length")

        if plot:
            num_samples = 200 if len(s[0,:]) > 200 else len(s[0,:])
            plt.plot(np.asarray(s[0,:num_samples]).squeeze().real[:num_samples])
            plt.show()
        return np.mean((s - s_hat)**2)