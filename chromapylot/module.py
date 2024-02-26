from tifffile import imread, imsave
from typing import Any
from scipy.ndimage import shift
from .main import get_file_path
from .chromapylot import DataType
import numpy as np
import json
from .extract_module import extract_properties
from astropy.table import Table


class Module:
    def __init__(
        self,
        input_type: DataType,
        output_type: DataType,
        reference_type: DataType = None,
        supplementary_type: DataType = None,
    ):
        self.input_type = input_type
        self.output_type = output_type
        self.reference_type = reference_type
        self.reference_data = None
        self.supplementary_type = supplementary_type
        self.switched = False

    def run(self, data: Any):
        raise NotImplementedError

    def load_data(self, input_path):
        raise NotImplementedError

    def load_reference_data(self):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        """
        Load supplementary data for the module.

        Args:
            input_path (str): The path or the directory to the input data.
            cycle (str): The cycle to process.

        Returns:
            Any: The supplementary data.
        """
        raise NotImplementedError

    def save_data(self, output_path, data):
        raise NotImplementedError

    def switch_input_supplementary(self):
        """
        Switch the input type with the supplementary type.
        """
        self.switched = True
        self.input_type, self.supplementary_type = (
            self.supplementary_type,
            self.input_type,
        )


class TiffModule(Module):
    def __init__(self):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_3D)

    def run(self, array_3d):
        raise NotImplementedError

    def load_data(self, input_path):
        image_path = get_file_path(input_path, "", "tif")
        image = imread(image_path)
        return image

    def save_data(self, output_path, data):
        imsave(get_file_path(output_path, "_" + self.output_type, "tif"), data)


class ProjectModule(Module):
    def __init__(self):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_2D)


class SkipModule(TiffModule):
    def __init__(self, z_binning):
        """
        Parameters:
        z_binning (int): The number of z-planes to skip.
        """
        super().__init__()
        self.z_binning = z_binning

    def run(self, array_3d):
        if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
            raise ValueError("Input must be a 3D numpy array.")

        return array_3d[:: self.z_binning, :, :]


class ShiftModule(Module):
    def __init__(self, input_type, output_type):
        """
        Parameters:
        shift_dict (dict): A dictionary with the shift values for each label.
        """
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            reference_type=[DataType.SHIFT_DICT, None],
            supplementary_type=[DataType.SHIFT_TUPLE, None],
        )
        self.reference_data = None

    def run(self, array_2d_or_3d, shift_tuple):
        if len(array_2d_or_3d.shape) == 3:
            shift_tuple = [0, shift_tuple[0], shift_tuple[1]]  # No shift along Z axis
        elif len(array_2d_or_3d.shape) != 2:
            raise ValueError("Input must be a 2D or 3D numpy array.")

        return shift(array_2d_or_3d, shift_tuple)

    def load_supplementary_data(self, input_path, cycle):
        if self.reference_data is None:
            if input_path is None:
                return (0, 0)
            else:
                self.reference_data = json.load(open(input_path, "r"))
                return self.reference_data[cycle]
        else:
            return self.reference_data[cycle]


class Shift3DModule(ShiftModule):
    def __init__(self):
        super().__init__(
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D_SHIFTED,
        )


class Shift2DModule(ShiftModule):
    def __init__(self):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.IMAGE_2D_SHIFTED,
        )


class RegisterGlobalModule(Module):
    def __init__(self):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
        )


class RegisterLocalModule(Module):
    def __init__(self):
        super().__init__(
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
        )


class Segment3DModule(Module):
    def __init__(self):
        super().__init__(
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.IMAGE_3D_SEGMENTED,
        )


class Segment2DModule(Module):
    def __init__(self):
        super().__init__(
            input_type=[DataType.IMAGE_2D_SHIFTED, DataType.IMAGE_2D],
            output_type=DataType.IMAGE_2D_SEGMENTED,
        )


class ExtractModule(Module):
    def __init__(self, input_type, output_type, supplementary_type):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            supplementary_type=supplementary_type,
        )

    def run(self, image, mask):
        properties = extract_properties(image, mask)
        return Table(properties)

    def load_supplementary_data(self, input_path):
        shifted_img_path = get_file_path(input_path, "_shifted", "tif")
        self.supplementary_data["_shifted"] = imread(shifted_img_path)

    def load_data(self, input_path):
        mask_path = get_file_path(input_path, "_3Dmasks", "npy")
        masks = np.load(mask_path)
        return masks

    def save_data(self, output_path, data):
        data.write(
            get_file_path(output_path, "_props", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class Extract3DModule(ExtractModule):
    def __init__(self):
        super().__init__(
            input_type=[
                DataType.IMAGE_3D_SEGMENTED,
            ],
            output_type=DataType.TABLE_3D,
            supplementary_type=[
                DataType.IMAGE_3D_SHIFTED,
                DataType.IMAGE_3D,
            ],
        )


class Extract2DModule(ExtractModule):
    def __init__(self):
        super().__init__(
            input_type=[
                DataType.IMAGE_2D_SEGMENTED,
            ],
            output_type=DataType.TABLE_2D,
            supplementary_type=[
                DataType.IMAGE_2D_SHIFTED,
                DataType.IMAGE_2D,
            ],
        )


class FilterTableModule(Module):
    def __init__(self):
        super().__init__(
            input_type=DataType.TABLE,
            output_type=DataType.TABLE,
        )

    def run(self, table):
        """Filter the table based on the properties given in the props dictionary."""
        for key, value in self.props.items():
            table = table[table[key] > value]
        return table

    def load_data(self, input_path):
        props_path = get_file_path(input_path, "", "ecsv")
        properties_table = Table.read(props_path, format="ascii.ecsv")
        return properties_table

    def save_data(self, output_path, data):
        data.write(
            get_file_path(output_path, "_filtered", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class FilterMaskModule(FilterTableModule):
    def __init__(self):
        super().__init__()


class FilterLocalizationModule(FilterTableModule):
    def __init__(self):
        super().__init__()


class SelectMaskModule(Module):
    def __init__(self, input_type, output_type, supplementary_type):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            supplementary_type=supplementary_type,
        )

    def run(self, mask, table):
        filtered_mask = np.zeros_like(mask)
        for label in table["label"]:
            filtered_mask[mask == label] = label
        return filtered_mask

    def load_supplementary_data(self, input_path):
        mask_path = get_file_path(input_path, "_3Dmasks", "npy")
        self.supplementary_data["_3Dmasks"] = np.load(mask_path)

    def load_data(self, input_path):
        props_path = get_file_path(input_path, "_filtered", "ecsv")
        properties_table = Table.read(props_path, format="ascii.ecsv")
        return properties_table

    def save_data(self, output_path, data):
        np.save(get_file_path(output_path, "_mask_filtered", "npy"), data)


class SelectMask3DModule(SelectMaskModule):
    def __init__(self):
        super().__init__(
            input_type=DataType.IMAGE_3D_SEGMENTED,
            output_type=DataType.IMAGE_3D_SEGMENTED_SELECTED,
            supplementary_type=DataType.TABLE_3D,
        )


class SelectMask2DModule(SelectMaskModule):
    def __init__(self):
        super().__init__(
            input_type=DataType.IMAGE_2D_SEGMENTED,
            output_type=DataType.IMAGE_2D_SEGMENTED_SELECTED,
            supplementary_type=DataType.TABLE_2D,
        )


class RegisterLocalizationModule(Module):
    def __init__(self):
        super().__init__(
            input_type=DataType.TABLE_3D,
            output_type=DataType.TABLE_3D_REGISTERED,
            reference_type=[DataType.REGISTRATION_TABLE, None],
        )


class BuildTraceModule(Module):
    def __init__(self):
        super().__init__(
            input_type=DataType.TABLE,
            output_type=DataType.TRACE_TABLE,
            reference_type=DataType.IMAGE_SEGMENTED,
        )


class BuildMatrixModule(Module):
    def __init__(self):
        super().__init__(
            input_type=DataType.TRACE_TABLE,
            output_type=DataType.MATRIX,
        )
