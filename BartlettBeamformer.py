from .doaestimator import DOAEstimator
import numpy as np

class BartlettBeamformer(DOAEstimator):
    @classmethod
    def calculate_weight_matrix(cls, steering_vectors, **kwargs):
        """
        Parameters:
            :param steering_vectors : Generated using the array alignment and the incident angles
            :type steering_vectors: 2D numpy array with size: M x P, where P is the number of scanning angles
        Return values:
            :return weight_matrix: A weights column vector for each scanning angle
            :rtype weight_matrix: 2D numpy array with size: M x P, where P is the number of scanning angles

        """        
        return steering_vectors

    @classmethod    
    def calculate_spectrum(cls, R, steering_vectors, **kwargs):
        s = steering_vectors
        PAD = s.conj().T @ R @ s # (P x P)
        PAD = np.diag(PAD) # (P,)

        return PAD
    
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
            :return MSE (float): difference b/w actual and reconstructed signal samples squared and averaged
        """
        # --> Input check
        if s.shape[1] != s_hat.shape[1]:
            raise TypeError("original and reconstructed signal vector different length")

        if plot:
            num_samples = 200 if len(s[0,:]) > 200 else len(s[0,:])
            plt.plot(np.asarray(s[0,:num_samples]).squeeze().real[:num_samples])
            plt.show()
        return np.mean((s - s_hat)**2)