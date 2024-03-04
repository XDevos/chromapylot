import os
from typing import Dict, List
import numpy as np
from astropy.table import Table
from modules.module import Module
from parameters import MatrixParams, AcquisitionParams
from core_types import DataType
import uuid


def init_localization_table():
    return Table(
        names=(
            "Buid",
            "ROI #",
            "CellID #",
            "Barcode #",
            "id",
            "zcentroid",
            "xcentroid",
            "ycentroid",
            "sharpness",
            "roundness1",
            "roundness2",
            "npix",
            "sky",
            "peak",
            "flux",
            "mag",
        ),
        dtype=(
            "S2",
            "int",
            "int",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )


def init_trace_table():
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


def add_localization_to_trace_table(
    localization, trace_table, trace_buid, x, y, z, mask_id
):
    trace_table.add_row()
    trace_table[-1]["Spot_ID"] = localization["Buid"]
    trace_table[-1]["Trace_ID"] = trace_buid
    trace_table[-1]["x"] = x
    trace_table[-1]["y"] = y
    trace_table[-1]["z"] = z
    trace_table[-1]["Chrom"] = "NotImplemented"
    trace_table[-1]["Chrom_Start"] = "NotImplemented"
    trace_table[-1]["Chrom_End"] = "NotImplemented"
    trace_table[-1]["ROI #"] = localization["ROI #"]
    trace_table[-1]["Mask_id"] = mask_id
    trace_table[-1]["Barcode #"] = localization["Barcode #"]
    trace_table[-1]["label"] = ""
    return trace_table


class BuildTrace3DModule(Module):
    def __init__(
        self, acquisition_params: AcquisitionParams, matrix_params: MatrixParams
    ):
        super().__init__(
            input_type=DataType.TABLE_3D,
            output_type=DataType.TRACE_TABLE_3D_LIST,
            reference_type=[
                DataType.IMAGE_3D_SEGMENTED_SELECTED,
                DataType.IMAGE_3D_SEGMENTED,
            ],
        )
        self.reference_data: Dict[str, np.ndarray] = {}
        self.z_offset = matrix_params.z_offset
        self.tracing_method = matrix_params.tracing_method
        self.masks2process: Dict[str, str] = matrix_params.masks2process
        if "masking" not in self.tracing_method:
            self.reference_type = None
        self.pixel_size_xy = acquisition_params.pixelSizeXY
        self.pixel_size_z = acquisition_params.pixelSizeZ

    def convert_coords(self, x, y, z):
        """
        Convert image coordinates to nanometers.

        Parameters
        ----------
        x : float
        y : float
        z : float

        Returns
        -------
        coords : tuple
            vector with coordinates in nanometers.

        """

        return (x * self.pixel_size_xy, y * self.pixel_size_xy, z * self.pixel_size_z)

    def build_mask_trace_table(self, localizations, mask):
        trace_table = init_trace_table()
        nbr_of_masks = np.max(mask)
        mask_buid_dict = {}
        barcode_nbr_by_mask_id = np.zeros(
            nbr_of_masks + 2
        )  # +2 to include background ([0]) and unassigned ([-1])

        image_dim = mask.shape
        if len(image_dim) == 3:
            axis_size = {
                "x": image_dim[1],
                "y": image_dim[2],
                "z": image_dim[0],
            }
        else:
            raise ValueError("Segmented image dimension must be 3.")

        # loops over barcode Table rows
        print(f"> Aligning localizations to {nbr_of_masks} masks...")
        if "registered" not in localizations.meta["comments"]:
            print("WARNING: Localizations are not 3D registered!")
        for barcode_row in localizations:
            # gets xyz coordinates
            x_sub_pix = barcode_row["xcentroid"]
            y_sub_pix = barcode_row["ycentroid"]
            z_sub_pix = barcode_row["zcentroid"]

            # binarizes coordinate
            x_int = int(np.nan_to_num(x_sub_pix, nan=-1))
            y_int = int(np.nan_to_num(y_sub_pix, nan=-1))
            z_int = int(np.nan_to_num(z_sub_pix, nan=-1)) + int(self.z_offset)

            if (
                x_int < axis_size["x"]
                and y_int < axis_size["y"]
                and z_int < axis_size["z"]
                and x_int > 0
                and y_int > 0
                and z_int > 0
            ):
                mask_id = mask[z_int, x_int, y_int]
                if mask_id != 0:
                    x_nm, y_nm, z_nm = self.convert_coords(
                        x_sub_pix, y_sub_pix, z_sub_pix
                    )
                    if mask_id not in mask_buid_dict:
                        # Assigns a unique identifier to each mask
                        mask_buid_dict[mask_id] = str(uuid.uuid4())
                    # Adds localization to trace table
                    trace_table = add_localization_to_trace_table(
                        barcode_row,
                        trace_table,
                        mask_buid_dict[mask_id],
                        x_nm,
                        y_nm,
                        z_nm,
                        mask_id,
                    )
            else:
                # Barcode has numpy.NaN coordinates or outside the image, it is unassigned
                mask_id = nbr_of_masks + 2

            barcode_nbr_by_mask_id[mask_id] += 1

        # Report part of the trace table built
        n_cells_assigned = np.count_nonzero(barcode_nbr_by_mask_id[1:-1])
        n_cells_unassigned = nbr_of_masks - n_cells_assigned
        n_barcodes_in_mask = barcode_nbr_by_mask_id[1:-1]
        n_barcodes_in_background = barcode_nbr_by_mask_id[0]
        n_barcodes_unassigned = barcode_nbr_by_mask_id[-1]

        print(
            f"$ Number of cells assigned: {n_cells_assigned} \
                | unassigned: {n_cells_unassigned}"
        )
        print(
            f"$ Number of barcodes in masks: {n_barcodes_in_mask} \
                | in background: {n_barcodes_in_background} \
                | outisde the image: {n_barcodes_unassigned}"
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
        method_names = []
        if "clustering" in self.tracing_method:
            method_names.append("KDtree")
        if "masking" in self.tracing_method:
            for key in self.masks2process.keys():
                method_names.append(f"mask-{key}")
        for trace_table, method in zip(list(data), method_names):
            print(f"Saving {method} trace table.")
            base = os.path.basename(input_path)
            roi_nbr = trace_table["ROI #"][0]
            barcode_2d_or_3d = "_".join(base.split("_")[1:])
            out_name = f"Trace_{barcode_2d_or_3d}_{method}_ROI-{roi_nbr}.ecsv"
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
