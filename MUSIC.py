from .doaestimator import DOAEstimator
import numpy as np

class MUSIC(DOAEstimator):

    def calculate_spectrum(self, R, steering_vectors, **kwargs):
        K = kwargs.get("K", None)
        assert K is not None
        
        s = steering_vectors

        _, U = np.linalg.eigh(R)
        # Eigenvalues are sorted in ascending order.
        U_n = U[:,:-K] # (M x (M - K))

        # Angular distribution of the nullspace/noise subspace orthogonality
        ORTAD = 1 / (s.conj().T @ U_n @ U_n.conj().T @ s) # [(P x M) @ (M x (M - K)) @ ((M - K) x M) @ (M x P)] = (P x P)
        ORTAD = np.diag(ORTAD) # (P,) 

        return ORTAD
