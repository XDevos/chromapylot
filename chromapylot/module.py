from tifffile import imread, imsave
from typing import Any, List, Union, Dict
from scipy.ndimage import shift
from data_manager import get_file_path
from core_types import DataType, first_type_accept_second
import numpy as np
import json
import os
from extract_module import extract_properties
from astropy.table import Table
from parameters import (
    ProjectionParams,
    AcquisitionParams,
    RegistrationParams,
    SegmentationParams,
    MatrixParams,
)


class Module:
    def __init__(
        self,
        input_type: Union[DataType, List[DataType]],
        output_type: DataType,
        reference_type: Union[DataType, List[DataType], None] = None,
        supplementary_type: Union[DataType, List[DataType], None] = None,
    ):
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

    def save_data(self, data, output_dir, input_path):
        raise NotImplementedError

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


class TiffModule(Module):
    def __init__(self):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_3D)

    def run(self, array_3d):
        raise NotImplementedError

    def load_data(self, input_path):
        image_path = get_file_path(input_path, "", "tif")
        image = imread(image_path)
        return image

    def save_data(self, data, output_dir, input_path):
        raise NotImplementedError


class ProjectModule(Module):
    def __init__(self, params: ProjectionParams):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_2D)
        self.params = params

    def run(self, array_3d):
        print("Projecting 3D image to 2D.")
        return np.max(array_3d, axis=0)

    def load_data(self, input_path):
        print("Loading 3D image.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 2D image.")


class SkipModule(Module):
    def __init__(self, params: AcquisitionParams):
        """
        Parameters:
        z_binning (int): The number of z-planes to skip.
        """
        super().__init__(
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D,
        )
        self.z_binning = params.zBinning

    def run(self, array_3d):
        print(f"Skipping every {self.z_binning} z-planes.")
        return array_3d[:: self.z_binning, :, :]

    def load_data(self, input_path):
        print("Loading 3D image.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 3D image.")


class ShiftModule(Module):
    def __init__(self, input_type, output_type):
        """
        Parameters:
        shift_dict (dict): A dictionary with the shift values for each label.
        """
        super().__init__(
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
        print(f"input_path: {input_path}")
        if self.reference_data is None:
            if input_path is None:
                return (0, 0)
            else:
                self.reference_data = json.load(open(input_path, "r"))
                return self.reference_data[cycle]
        else:
            return self.reference_data[cycle]

    def load_reference_data(self, paths: List[str]):
        pass


class Shift3DModule(ShiftModule):
    def __init__(self, params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D_SHIFTED,
        )

    def run(self, array_2d_or_3d, shift_tuple):
        print(f"Shifting 3D image with {shift_tuple}.")
        return array_2d_or_3d

    def load_data(self, input_path):
        print("Loading 3D image.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 3D image.")


class Shift2DModule(ShiftModule):
    def __init__(self, params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.IMAGE_2D_SHIFTED,
        )

    def run(self, array_2d_or_3d, shift_tuple):
        print("Shifting 2D image.")
        return array_2d_or_3d

    def load_data(self, input_path):
        print("Loading 2D image.")
        return np.ones((10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 2D image.")


class RegisterGlobalModule(Module):
    def __init__(self, params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
        )

    def run(self, image):
        return [0, 0]

    def load_data(self, input_path):
        print("Loading 2D image.")
        return np.ones((10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving shift tuple.")

    def load_reference_data(self, paths: List[str]):
        return {
            "DAPI": [0, 0],
            "RT1": [0, 0],
            "RT2": [0, 0],
            "RT3": [0, 0],
            "mask0": [0, 0],
            "mask1": [0, 0],
        }

    def load_supplementary_data(self, input_path, cycle):
        return [0, 0]


class RegisterLocalModule(Module):
    def __init__(self, params: RegistrationParams):
        super().__init__(
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
        )

    def run(self, shifted_image):
        return Table()

    def load_data(self, input_path):
        print("Loading 3D image.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving registration table.")

    def load_reference_data(self, paths: List[str]):
        print("Loading 3D image.")
        self.reference_data = np.ones((10, 10, 10))


class Segment3DModule(Module):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.IMAGE_3D_SEGMENTED,
        )

    def run(self, image):
        print("Segmenting 3D image.")
        return np.ones_like(image)

    def load_data(self, input_path):
        print("Loading 3D image.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 3D image.")


class Segment2DModule(Module):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=[DataType.IMAGE_2D_SHIFTED, DataType.IMAGE_2D],
            output_type=DataType.IMAGE_2D_SEGMENTED,
        )

    def run(self, image):
        print("Segmenting 2D image.")
        return np.ones_like(image)

    def load_data(self, input_path):
        print("Loading 2D image.")
        return np.ones((10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 2D image.")


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

    def save_data(self, data, output_dir, input_path):
        data.write(
            get_file_path(output_dir, "_props", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class Extract3DModule(ExtractModule):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=DataType.IMAGE_3D_SEGMENTED,
            output_type=DataType.TABLE_3D,
            supplementary_type=[
                DataType.IMAGE_3D_SHIFTED,
                DataType.IMAGE_3D,
            ],
        )

    def run(self, image, mask):
        print("Extracting properties from 3D image.")
        return Table()

    def load_data(self, input_path):
        print("Loading 3D mask.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving properties.")


class Extract2DModule(ExtractModule):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D_SEGMENTED,
            output_type=DataType.TABLE_2D,
            supplementary_type=[
                DataType.IMAGE_2D_SHIFTED,
                DataType.IMAGE_2D,
            ],
        )

    def run(self, image, mask):
        print("Extracting properties from 2D image.")
        return Table()

    def load_data(self, input_path):
        print("Loading 2D mask.")
        return np.ones((10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving properties.")


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

    def save_data(self, data, output_dir, input_path):
        data.write(
            get_file_path(output_dir, "_filtered", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class FilterMaskModule(FilterTableModule):
    def __init__(self, params: SegmentationParams):
        super().__init__()

    def run(self, table):
        print("Filtering mask.")
        return Table()

    def load_data(self, input_path):
        print("Loading properties.")
        return Table()

    def save_data(self, data, output_dir, input_path):
        print("Saving filtered mask table.")


class FilterLocalizationModule(FilterTableModule):
    def __init__(self, params: MatrixParams):
        super().__init__()

    def run(self, table):
        print("Filtering ocalization.")
        return Table()

    def load_data(self, input_path):
        print("Loading properties.")
        return Table()

    def save_data(self, data, output_dir, input_path):
        print("Saving filtered ocalization table.")


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

    def save_data(self, data, output_dir, input_path):
        np.save(get_file_path(output_dir, "_mask_filtered", "npy"), data)


class SelectMask3DModule(SelectMaskModule):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=DataType.IMAGE_3D_SEGMENTED,
            output_type=DataType.IMAGE_3D_SEGMENTED_SELECTED,
            supplementary_type=DataType.TABLE_3D,
        )

    def run(self, mask, table):
        print("Selecting 3D mask.")
        return np.ones_like(mask)

    def load_data(self, input_path):
        print("Loading 3D mask.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 3D mask.")

    def load_supplementary_data(self, input_path):
        print("Loading 3D mask table.")
        return Table()


class SelectMask2DModule(SelectMaskModule):
    def __init__(self, params: SegmentationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D_SEGMENTED,
            output_type=DataType.IMAGE_2D_SEGMENTED_SELECTED,
            supplementary_type=DataType.TABLE_2D,
        )

    def run(self, mask, table):
        print("Selecting 2D mask.")
        return np.ones_like(mask)

    def load_data(self, input_path):
        print("Loading 2D mask.")
        return np.ones((10, 10, 10))

    def save_data(self, data, output_dir, input_path):
        print("Saving 2D mask.")

    def load_supplementary_data(self, input_path):
        print("Loading 2D mask table.")
        return Table()


class RegisterLocalizationModule(Module):
    def __init__(self, params: MatrixParams):
        super().__init__(
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

    def save_data(self, data, output_dir, input_path):
        print("Saving registered localization table.")

    def load_reference_data(self, paths: List[str]):
        print("Loading registration table.")
        return Table()


class BuildTraceModule(Module):
    def __init__(self, params: MatrixParams):
        super().__init__(
            input_type=DataType.TABLE,
            output_type=DataType.TRACE_TABLE_LIST,
            reference_type=DataType.SEGMENTED,
        )
        self.reference_data: Dict[str, np.ndarray] = {}
        self.tracing_method = params.tracing_method
        self.masks2process: Dict[str, str] = params.masks2process
        if "masking" not in self.tracing_method:
            self.reference_type = None

    def init_trace_table(self):
        trace_table = Table(
            names=(
                "Spot_ID",
                "Trace_ID",
                "x",
                "y",
                "z",
                "Chrom",
                "Chrom_Start",
                "Chrom_End",
                "ROI #",
                "Mask_id",
                "Barcode #",
                "label",
            ),
            dtype=(
                "S2",
                "S2",
                "f4",
                "f4",
                "f4",
                "S2",
                "int",
                "int",
                "int",
                "int",
                "int",
                "S2",
            ),
        )
        trace_table.meta["xyz_unit"] = "micron"
        trace_table.meta["genome_assembly"] = "mm10"
        return trace_table

    def build_mask_trace_table(self, localizations, mask):
        trace_table = self.init_trace_table()
        for loc in localizations:
            mask_id = mask[loc["y"], loc["x"]]
            if mask_id != 0:
                trace_table.add_row(
                    [
                        loc["Spot_ID"],
                        loc["Trace_ID"],
                        loc["x"],
                        loc["y"],
                        loc["z"],
                        loc["Chrom"],
                        loc["Chrom_Start"],
                        loc["Chrom_End"],
                        loc["ROI #"],
                        mask_id,
                        loc["Barcode #"],
                        loc["label"],
                    ]
                )
        return trace_table

    def run(self, localizations):
        output = []
        if "clustering" in self.tracing_method:
            print("Building cluster trace table.")
            raise NotImplementedError
        if "masking" in self.tracing_method:
            for key, value in self.masks2process.items():
                print(f"Building mask {value} trace table.")
                trace_table = self.init_trace_table()
                trace_table = self.build_mask_trace_table(
                    localizations, self.reference_data[value]
                )
                output.append(trace_table)
        return output

    def load_data(self, input_path):
        print("Loading properties.")
        return Table.read(input_path, format="ascii.ecsv")

    def save_data(self, data, output_dir, input_path):
        print("Saving trace table.")
        print(f"data: {list(data)}")
        print(f"self.tracing_method: {list(self.tracing_method)}")
        for trace_table, method in zip(list(data), list(self.tracing_method)):
            print(f"Saving {method} trace table.")
            base = os.path.basename(input_path)
            out_name = "Trace_" + "_".join(base.split("_")[1:]) + "_" + method + ".ecsv"
            trace_table.write(
                os.path.join(output_dir, out_name), format="ascii.ecsv", overwrite=True
            )

    def load_reference_data(self, paths: List[str]):
        if "masking" in self.tracing_method:
            for key, val in self.masks2process.items():
                for path in paths:
                    if val in path:
                        print(f"Loading {path} for mask {val}.")
                        self.reference_data[val] = np.load(path)
        else:
            print("No reference data needed for tracing method {self.tracing_method}.")


class BuildMatrixModule(Module):
    def __init__(self, params: MatrixParams):
        super().__init__(
            input_type=[DataType.TRACE_TABLE_LIST, DataType.TRACE_TABLE],
            output_type=DataType.MATRIX,
        )

    def run(self, trace_table):
        print("Building matrix.")
        return np.ones((10, 10))

    def load_data(self, input_path):
        print("Loading trace table.")
        return Table()

    def save_data(self, data, output_dir, input_path):
        print("Saving matrix.")
