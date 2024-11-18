class Signal:
    """
    Represents a signal with specific properties, independent of DOA estimation.
    
    Attributes:
        sampling_frequency (float): Sampling frequency in Hz.
        doa_angles (list): List of direction of arrival angles in degrees.
        noise (Noise): Instance of a noise class to apply noise to the signal.
    """

    def __init__(self, sampling_frequency: float, T,  doa_angles: list):
        """
        Initializes the signal with its sampling frequency and direction of arrival angles.

        Args:
            sampling_frequency (float): Sampling frequency.
            doa_angles (list): Direction of arrival angles.
            noise (Noise, optional): Noise instance to add to the signal.
        """
        self.sampling_frequency = sampling_frequency
        self.sampling_period = 1 / sampling_frequency
        self.sampling_time = T
        self.sampling_count = T * self.sampling_period
        self.doa_angles = doa_angles
