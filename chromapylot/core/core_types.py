from enum import Enum
from typing import Literal

DataType = Literal[
    "_3d",
    "_2d",
    "shift_tuple",
    "shift_dict",
    "shift_table",
    "_skipped",
    "_shifted",
    "_segmented",
    "_selected",
    "_table",
    "_filtered",
    "_registered",
]


class DataType(Enum):
    IMAGE_3D = "IMAGE_3D"
    IMAGE_2D = "IMAGE_2D"
    SHIFT_TUPLE = "SHIFT_TUPLE"
    SHIFT_DICT = "SHIFT_DICT"
    REGISTRATION_TABLE = "REGISTRATION_TABLE"
    IMAGE_3D_SHIFTED = "IMAGE_3D_SHIFTED"
    IMAGE_3D_SEGMENTED = "IMAGE_3D_SEGMENTED"
    IMAGE_2D_SHIFTED = "IMAGE_2D_SHIFTED"
    IMAGE_2D_SEGMENTED = "IMAGES_2D_SEGMENTED"
    IMAGE_2D_SEGMENTED_SELECTED = "IMAGE_2D_SEGMENTED_SELECTED"
    IMAGE_3D_SEGMENTED_SELECTED = "IMAGE_3D_SEGMENTED_SELECTED"
    SEGMENTED = "SEGMENTED"
    TABLE_3D = "TABLE_3D"
    TABLE_2D = "TABLE_2D"
    TABLE = "TABLE"
    TABLE_FILTERED = "TABLE_FILTERED"
    TABLE_3D_REGISTERED = "TABLE_3D_REGISTERED"
    TRACES = "TRACES"
    TRACES_LIST = "TRACES_LIST"
    TRACES_LIST_3D = "TRACES_LIST_3D"
    TRACES_3D = "TRACES_3D"
    TRACES_2D = "TRACES_2D"
    MATRIX_3D = "MATRIX_3D"
    MATRIX_2D = "MATRIX_2D"
    MATRIX = "MATRIX"


class AnalysisType(Enum):
    FIDUCIAL = "fiducial"
    BARCODE = "barcode"
    DAPI = "DAPI"
    RNA = "RNA"
    PRIMER = "primer"
    TRACE = "trace"


class CommandName(Enum):
    PROJECT = "project"
    SKIP = "skip"
    SHIFT_3D = "shift_3d"
    SHIFT_2D = "shift_2d"
    REGISTER_GLOBAL = "register_global"
    REGISTER_LOCAL = "register_local"
    SEGMENT_3D = "segment_3d"
    SEGMENT_2D = "segment_2d"
    EXTRACT_3D = "extract_3d"
    EXTRACT_2D = "extract_2d"
    FILTER_MASK = "filter_mask"
    SELECT_MASK_3D = "select_mask_3d"
    SELECT_MASK_2D = "select_mask_2d"
    FILTER_TABLE = "filter_table"
    FILTER_LOCALIZATION = "filter_localization"
    REGISTER_LOCALIZATION = "register_localization"
    BUILD_TRACE_3D = "build_trace_3d"
    BUILD_TRACE_2D = "build_trace_2d"
    BUILD_MATRIX = "build_matrix"


ExampleType = Literal[
    "fiducial_3d",
    "fiducial_2d",
    "shift_tuple",
    "shift_dict",
    "shift_table",
    "fiducial_3d_shifted",
    "barcode_3d",
    "barcode_3d_shifted",
    "barcode_3d_segmented",
    "barcode_3d_table",
    "barcode_3d_table_filtered",
    "barcode_3d_table_registered",
    "barcode_3d_table_filtered_registered",
    "barcode_3d_table_registered_filtered",
    "mask_3d",
    "mask_3d_shifted",
    "mask_3d_segmented",
    "mask_3d_table",
    "mask_3d_table_filtered",
]


def get_data_type(filename, extension):
    if extension in ["png", "log", "md"]:
        return None
    elif extension == "tif":
        if "_shifted" in filename:
            return DataType.IMAGE_3D_SHIFTED
        else:
            return DataType.IMAGE_3D
    elif extension == "npy":
        if "_2d" in filename:
            if "_Matrix" in filename:
                return DataType.MATRIX_2D
            elif "_shifted" in filename:
                return DataType.IMAGE_2D_SHIFTED
            else:
                return DataType.IMAGE_2D
        elif "_Matrix" in filename:
            return DataType.MATRIX_3D
        elif "_3Dmasks" in filename:
            return DataType.IMAGE_3D_SEGMENTED
        elif "_Masks" in filename:
            return DataType.IMAGE_2D_SEGMENTED
    elif extension == "json":
        if filename in ["shifts", "register_global"]:
            return DataType.SHIFT_DICT
    elif extension in ["ecsv", "dat"]:
        if "_block3D" in filename:
            return DataType.REGISTRATION_TABLE
        elif "Trace_3D" in filename:
            return DataType.TRACES_3D
        elif "Trace" in filename:
            return DataType.TRACES_2D
        elif "_3D_barcode" in filename or "_3D_mask" in filename:
            return DataType.TABLE_3D
        elif "_barcode" in filename or "_mask" in filename:
            return DataType.TABLE_2D
    else:
        raise ValueError(f"Unknown type of data: {filename}.{extension}")


def first_type_accept_second(first_type: DataType, second_type: DataType):
    """Check if the first type of data can accept the second type of data.
    Example:
    >>> first_type_accept_second(DataType.IMAGE_3D, DataType.IMAGE_3D_SHIFTED)
    True
    >>> first_type_accept_second(DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D)
    False
    >>> first_type_accept_second(DataType.IMAGE_3D, DataType.IMAGE_2D)
    False
    """
    return first_type.value in second_type.value
