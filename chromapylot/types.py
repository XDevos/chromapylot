from typing import Literal
from enum import Enum


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
    IMAGE_2D_SEGMENTED = "IMAGE_2D_SEGMENTED"
    TABLE_3D = "TABLE_3D"
    TABLE_2D = "TABLE_2D"
    TABLE = "TABLE"
    TABLE_FILTERED = "TABLE_FILTERED"
    TABLE_3D_REGISTERED = "TABLE_3D_REGISTERED"
    TRACE_TABLE_3D = "TRACE_TABLE_3D"
    TRACE_TABLE_2D = "TRACE_TABLE_2D"
    MATRIX_3D = "MATRIX_3D"
    MATRIX_2D = "MATRIX_2D"


class AnalysisType(Enum):
    FIDUCIAL = "fiducial"
    BARCODE = "barcode"
    DAPI = "dapi"
    RNA = "rna"
    PRIMER = "primer"
    SATELLITE = "satellite"
    TRACE = "trace"


ModuleName = Literal[
    "project",
    "register_global",
    "skip",
    "shift",
    "shift_fiducial",
    "register_local",
    "filter_table",
    "filter_mask",
    "segment",
    "extract",
    "filter_localization",
    "register_localization",
    "build_trace",
    "build_matrix",
]

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
        if "shifts" in filename:
            return DataType.SHIFT_DICT
    elif extension in ["ecsv", "dat"]:
        if "_block3D" in filename:
            return DataType.REGISTRATION_TABLE
        elif "Trace_3D" in filename:
            return DataType.TRACE_TABLE_3D
        elif "Trace" in filename:
            return DataType.TRACE_TABLE_2D
        elif "_3D_barcode" in filename or "_3D_mask" in filename:
            return DataType.TABLE_3D
        elif "_barcode" in filename or "_mask" in filename:
            return DataType.TABLE_2D
    else:
        raise ValueError(f"Unknown type of data: {filename}.{extension}")