from .doaestimator import DOAEstimator
import numpy as np

class CaponBeamformer(DOAEstimator):
    def calculate_weight_matrix(steering_vectors,**kwargs):
        R = kwargs.get("R", None)
        assert R is not None
        
        # --> Input check
        if R.shape[0] != R.shape[1]:
            raise TypeError("Covariance matrix is not square")

        if R.shape[0] != steering_vectors.shape[0]:
            raise TypeError("Covariance matrix dimension does not match with the antenna array dimension")

        s = steering_vectors
        R_inv = np.linalg.pinv(R) # (M x M) pseudo-inverse tends to work better than a true inverse
        
        # MVDR/Capon Equation
        w_matrix = (R_inv @ s)/np.diag(s.conj().T @ R_inv @ s) # (M x M) * (M x P), denominator is (P x M) * (M x M) * (M x P), resulting in a (M x P) weights vector
        return w_matrix
    
    def calculate_pwr_ang_dens(R, steering_vectors):
        s = steering_vectors
        R_inv = np.linalg.pinv(R) # (M x M) pseudo-inverse tends to work better than a true inverse

        PAD = 1 / (s.conj().T @ R_inv @ s) # (P x P)
        PAD = np.diag(PAD) # (P,) 
        PAD = PAD.reshape(1,-1) # (1 x P) 
        
        return PAD