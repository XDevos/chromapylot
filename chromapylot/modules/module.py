import json
from typing import Any, Dict, List, Union
import os
from chromapylot.core.data_manager import save_npy
import numpy as np
from astropy.table import Table
from scipy.ndimage import shift
from scipy.ndimage import shift as shift_image
from chromapylot.core.core_types import DataType, first_type_accept_second
from chromapylot.core.data_manager import get_file_path, DataManager
from chromapylot.parameters.acquisition_params import AcquisitionParams
from chromapylot.parameters.matrix_params import MatrixParams
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.parameters.segmentation_params import SegmentationParams
from chromapylot.core.core_logging import print_module


class Module:
    def __init__(
        self,
        data_manager: DataManager,
        input_type: Union[DataType, List[DataType]],
        output_type: DataType,
        reference_type: Union[DataType, List[DataType], None] = None,
        supplementary_type: Union[DataType, List[DataType], None] = None,
    ):
        self.data_m = data_manager
        self.input_type = input_type
        self.output_type = output_type
        self.reference_type = reference_type
        self.reference_data = None
        self.supplementary_type = supplementary_type
        self.switched = False

    def run(self, data: Any, supplementary_data: Any = None):
        raise NotImplementedError

    def load_data(self, input_path):
        raise NotImplementedError

    def load_reference_data(self, paths: List[str]):
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

    def save_data(self, data, input_path, input_data, supplementary_data):
        raise NotImplementedError

    def print_module_name(self):
        print()
        print_module(self.__class__.__name__)

    def switch_input_supplementary(self):
        """
        Switch the input type with the supplementary type.
        """
        print(f"Switching input and supplementary types for {self.__class__.__name__}.")
        self.switched = True
        self.input_type, self.supplementary_type = (
            self.supplementary_type,
            self.input_type,
        )

    def is_compatible(self, data_type: DataType):
        """
        Check if the module is compatible with the given data type.

        Args:
            data_type (DataType): The data type to check.

        Returns:
            bool: True if the module is compatible, False otherwise.
        """
        if isinstance(self.input_type, list):
            for input in self.input_type:
                if first_type_accept_second(input, data_type):
                    print(f"Replace input type {self.input_type} with {input}.")
                    self.input_type = input
                    return True
        elif first_type_accept_second(self.input_type, data_type):
            return True
        # Input type(s) aren't compatible, now check if the supplementary type(s) is compatible.
        if self.supplementary_type is None:
            return False
        if isinstance(self.supplementary_type, list):
            for sup_type in self.supplementary_type:
                if first_type_accept_second(sup_type, data_type):
                    print(
                        f"Replace supplementary type {self.supplementary_type} with {sup_type}."
                    )
                    self.supplementary_type = sup_type
                    self.switch_input_supplementary()
                    return True
            return False
        if first_type_accept_second(self.supplementary_type, data_type):
            self.switch_input_supplementary()
            return True
        # No compatible type found.
        return False


class SkipModule(Module):
    def __init__(
        self, data_manager: DataManager, acquisition_params: AcquisitionParams
    ):
        """
        Parameters:
        z_binning (int): The number of z-planes to skip.
        """
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D,
        )
        self.z_binning = acquisition_params.zBinning

    def run(self, array_3d):
        print(f"Skipping every {self.z_binning} z-planes.")
        return array_3d[:: self.z_binning, :, :].astype(np.float64)

    def load_data(self, input_path):
        print("Loading 3D image.")
        return self.data_m.load_image_3d(input_path)

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("NO Saving 3D image.")


class ShiftModule(Module):
    def __init__(self, data_manager: DataManager, input_type, output_type):
        """
        Parameters:
        shift_dict (dict): A dictionary with the shift values for each label.
        """
        super().__init__(
            data_manager=data_manager,
            input_type=input_type,
            output_type=output_type,
            reference_type=DataType.SHIFT_DICT,
            supplementary_type=DataType.SHIFT_TUPLE,
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
            print("ref data is none, loading from: ", input_path)
            if input_path is None:
                return (0, 0)
            else:
                self.reference_data = json.load(open(input_path, "r"))
                return self.reference_data[cycle]
        else:
            if cycle not in self.reference_data:
                print("Cycle not found in ref data, returning (0,0)")
                return (0, 0)
            print("Cycle found in ref data, returning: ", self.reference_data[cycle])
            return self.reference_data[cycle]

    def load_reference_data(self, paths: List[str]):
        path = paths[0] if len(paths) == 1 else None
        if path is None:
            print("No shift dictionary provided.")
            self.reference_data = None
        else:
            print("> Loading shift dictionary from: ", path)
            shift_dict = json.load(open(path, "r"))
            self.reference_data = list(shift_dict.values())[0]


class Shift3DModule(ShiftModule):
    def __init__(
        self, data_manager: DataManager, registration_params: RegistrationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D_SHIFTED,
        )

    def run(self, array_3d, shift_tuple):
        shift_3d = (0, shift_tuple[0], shift_tuple[1])
        if shift_3d == (0, 0, 0):
            print("[Shift] No shift for this cycle used for fiducial reference.")
            return array_3d
        else:
            print(f"[Shift] 3D image with {shift_3d}.")
            return shift_image(array_3d, shift_3d)

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("NO saving 3D image.")


class Shift2DModule(ShiftModule):
    def __init__(
        self, data_manager: DataManager, registration_params: RegistrationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_2D,
            output_type=DataType.IMAGE_2D_SHIFTED,
        )
        self.dirname = "register_global"

    def run(self, array_2d, shift_tuple):
        print(f"[Shift] 2D image with {shift_tuple}.")
        return shift_image(array_2d, shift_tuple)

    def load_data(self, input_path):
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[self.data_m.in_dir_len :]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("[Save] 2D npy")
        base = os.path.basename(input_path).split(".")[0]
        base = base[:-3] if base[-3:] == "_2d" else base
        npy_filename = base + "_2d_registered.npy"
        npy_path = os.path.join(
            self.data_m.output_folder, self.dirname, "data", npy_filename
        )
        save_npy(data, npy_path, self.data_m.out_dir_len)


class FilterTableModule(Module):
    def __init__(self, data_manager: DataManager):
        super().__init__(
            data_manager=data_manager,
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

    def save_data(self, data, input_path, input_data, supplementary_data):
        data.write(
            get_file_path(self.data_m.output_folder, "_filtered", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class FilterMaskModule(FilterTableModule):
    def __init__(
        self, data_manager: DataManager, segmentation_params: SegmentationParams
    ):
        super().__init__(
            data_manager=data_manager,
        )

    def run(self, table):
        print("Filtering mask.")
        return Table()

    def load_data(self, input_path):
        print("Loading properties.")
        return Table()

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving filtered mask table.")


class FilterLocalizationModule(FilterTableModule):
    def __init__(self, data_manager: DataManager, matrix_params: MatrixParams):
        super().__init__(
            data_manager=data_manager,
        )

    def run(self, table):
        print("Filtering ocalization.")
        return Table()

    def load_data(self, input_path):
        print("Loading properties.")
        return Table()

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving filtered ocalization table.")


class SelectMaskModule(Module):
    def __init__(
        self, data_manager: DataManager, input_type, output_type, supplementary_type
    ):
        super().__init__(
            data_manager=data_manager,
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

    def save_data(self, data, input_path, input_data, supplementary_data):
        np.save(get_file_path(self.data_m.output_folder, "_mask_filtered", "npy"), data)


class SelectMask3DModule(SelectMaskModule):
    def __init__(
        self, data_manager: DataManager, segmentation_params: SegmentationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.SEGMENTED_3D,
            output_type=DataType.SEGMENTED_3D_SELECTED,
            supplementary_type=DataType.TABLE_3D,
        )

    def run(self, mask, table):
        print("Selecting 3D mask.")
        return np.ones_like(mask)

    def load_data(self, input_path):
        print("Loading 3D mask.")
        return np.ones((10, 10, 10))

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving 3D mask.")

    def load_supplementary_data(self, input_path):
        print("Loading 3D mask table.")
        return Table()


class SelectMask2DModule(SelectMaskModule):
    def __init__(
        self, data_manager: DataManager, segmentation_params: SegmentationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.SEGMENTED_2D,
            output_type=DataType.SEGMENTED_2D_SELECTED,
            supplementary_type=DataType.TABLE_2D,
        )

    def run(self, mask, table):
        print("Selecting 2D mask.")
        return np.ones_like(mask)

    def load_data(self, input_path):
        print("Loading 2D mask.")
        return np.ones((10, 10, 10))

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving 2D mask.")

    def load_supplementary_data(self, input_path):
        print("Loading 2D mask table.")
        return Table()


class RegisterLocalizationModule(Module):
    def __init__(self, data_manager: DataManager, matrix_params: MatrixParams):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.TABLE_3D,
            output_type=DataType.TABLE_3D_REGISTERED,
            reference_type=DataType.REGISTRATION_TABLE,
        )

    def run(self, table):
        print("Registering localization.")
        return Table()

    def load_data(self, input_path):
        print("Loading properties.")
        return Table()

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving registered localization table.")

    def load_reference_data(self, paths: List[str]):
        print("Loading registration table.")
        return Table()


class BuildMatrixModule(Module):
    def __init__(self, data_manager: DataManager, matrix_params: MatrixParams):
        super().__init__(
            data_manager=data_manager,
            input_type=[DataType.TRACES_LIST, DataType.TRACES],
            output_type=DataType.MATRIX,
        )

    def run(self, trace_table):
        print("Building matrix.")
        return np.ones((10, 10))

    def load_data(self, input_path):
        print("Loading trace table.")
        return Table()

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("Saving matrix.")
