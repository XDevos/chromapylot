from scipy.ndimage import shift
import numpy as np
from tifffile import imread, imsave

from .main import get_img_name, get_file_path, load_shifts_from_json
from .module import TiffModule


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


def skip_z_planes(array_3d):
    if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
        raise ValueError("Input must be a 3D numpy array.")

    return array_3d[::2, :, :]


class SkipModule(TiffModule):
    def __init__(self, z_binning) -> None:
        """
        Parameters:
        z_binning (int): The number of z-planes to skip.
        """
        super().__init__("skipped")
        self.z_binning = z_binning

    def run(self, array_3d):
        if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
            raise ValueError("Input must be a 3D numpy array.")

        return array_3d[:: self.z_binning, :, :]


class ShiftModule(TiffModule):
    def __init__(self, shift_dict) -> None:
        """
        Parameters:
        shift_dict (dict): A dictionary with the shift values for each label.
        """
        super().__init__("shifted")
        self.shift_dict = shift_dict

    def run(self, array_3d, label_name):
        if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
            raise ValueError("Input must be a 3D numpy array.")

        return shift_3d_array_subpixel(array_3d, self.shift_dict[label_name])
