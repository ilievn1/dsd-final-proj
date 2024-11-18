from .doaestimator import DOAEstimator
import numpy as np

class BartlettBeamformer(DOAEstimator):

    def calculate_weight_matrix(steering_vectors,**kwargs):
        return steering_vectors
        
    def calculate_pwr_ang_dens(R, steering_vectors):
        s = steering_vectors
        PAD = s.conj().T @ R @ s # (P x P)
        PAD = np.diag(PAD) # (P,)
        PAD = PAD.reshape(1,-1) # (1 x P)

        return PAD
