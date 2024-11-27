from .doaestimator import DOAEstimator
import numpy as np
from .utils import *

class RMUSIC(DOAEstimator):
    @classmethod
    def calculate_spectrum(cls, R, steering_vectors, **kwargs):
        raise TypeError("Non-spectrum method")

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
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")

        _, U = np.linalg.eigh(R)

        # Eigenvalues are sorted in ascending order.
        U_n = U[:,:-K] # (M x (M - K))

        R_n = U_n @ U_n.conj().T # (M x (M - K)) @ ((M - K) x M)] = (M x M)

        pol_coef = sum_across_diagonals(R_n)
        roots = np.roots(pol_coef)
        print("roots of pol_coef", roots)
        print("est w/o filtering roots inside UN", np.rad2deg(np.arcsin(1/(2*np.pi*0.5)*np.angle(roots))))

        roots_inside_UC = roots[np.abs(roots) < 1]
        # Sort roots in asc order according to UC proximity
        roots_inside_UC = roots_inside_UC[np.argsort(np.abs(np.abs(roots_inside_UC)-1))]
        print("roots_inside_UC", roots_inside_UC)
        roots_inside_UC = roots_inside_UC[:K]
        print("roots_inside_UC after filter K most probable", roots_inside_UC)
        
        # TODO: Here 0.5 = d, is magic number, when DOA ABC is fully implemented, d should be accesible thru parent ABC
        estimated_DOAs = np.arcsin(1/(2*np.pi*0.5)*np.angle(roots_inside_UC))
        estimated_DOAs = np.rad2deg(estimated_DOAs).tolist()

        return estimated_DOAs, None