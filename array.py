class Array:
    """
    Base class for array configurations in DOA estimation.

    Attributes:
        num_elements (int): Number of elements in the array.
        element_spacing (float): Spacing between array elements.
    """

    def __init__(self, num_elements: int, element_spacing: float):
        """
        Initializes the array with specified elements and spacing.

        Args:
            num_elements (int): The number of elements in the array.
            element_spacing (float): Spacing between array elements.
        """
        self.num_elements = num_elements
        self.element_spacing = element_spacing
    
    def array_geometry(self):
        """
        Computes and returns the geometry of the array.
        
        Returns:
            numpy.ndarray: The position of each array element.
        """
        raise NotImplementedError("This method should be implemented in subclasses")

