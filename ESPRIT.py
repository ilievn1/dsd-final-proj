from .doaestimator import DOAEstimator
import numpy as np
from scipy import linalg

class ESPRIT(DOAEstimator):
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

        # Eigenvalues/vectors are sorted in ascending order.
        _, U = np.linalg.eigh(R)

        # Signal subspace to be split
        U_s = U[:,-K:] # (M x K)

        # TODO: Here d_sub = 1 is magic number, d_sub denoting sub-array offset in number of elements
        # should be available when when strategy pattern is correctly implemented
        d_sub = 1
        U_s1 = U_s[:-d_sub,:] # ((M - d_sub) x K)
        U_s2 = U_s[d_sub:,:] # ((M - d_sub) x K)

        U_s1_H = U_s1.conj().T # (K x (M - d_sub))
        U_s2_H = U_s2.conj().T # (K x (M - d_sub))

        # TLS algorithm [pp.1175-1176; Sec. 9.3.4.1] H. L. Van Trees, Optimum array processing Vol.4
        C = np.vstack((U_s1_H, U_s2_H)) @ np.hstack((U_s1, U_s2))
        _, V = np.linalg.eigh(C)

        # Eigenvalues/vectors are sorted in descending order.
        V = np.fliplr(V) # (2K x 2K)
        V_12 = V[:K,K:] # (K x K)
        V_22 = V[K:,K:] # (K x K)
        V_22_inv = scipy.linalg.solve(V_22,np.eye(K))
        Psi_tls = -1*V_12 @ V_22_inv # (K x K)
        Phi = np.linalg.eigvals(Psi_tls)

        # TODO: Here 0.5 = d is magic number, when DOA ABC is fully implemented, d should be accesible thru parent ABC

        estimated_DOAs = np.arcsin(1/(d_sub)*1/(2*np.pi*0.5)*np.angle(Phi))
        estimated_DOAs = np.rad2deg(estimated_DOAs)
        estimated_DOAs = np.sort(estimated_DOAs)

        return estimated_DOAs, None