import numpy as np
from scipy.ndimage import shift


def shift_3d_array_subpixel(array_3d, shift_values):
    """
    Shifts a 3D numpy array along the X and Y axes with subpixel accuracy.

    Parameters:
    array_3d (numpy.ndarray): The 3D array to shift.
    shift_values (list): A list of two floats representing the shift values for the X and Y axes.

    Returns:
    numpy.ndarray: The shifted 3D array.
    """
    if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
        raise ValueError("Input must be a 3D numpy array.")
    if not isinstance(shift_values, list) or len(shift_values) != 2:
        raise ValueError("Shift values must be a list of two floats.")

    shift_vector = [0, shift_values[0], shift_values[1]]  # No shift along Z axis
    shifted_array = shift(array_3d, shift_vector)

    return shifted_array
