#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import matplotlib

matplotlib.use("Agg")

from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.patches import Polygon
from modules.module import Module
from scipy.spatial import KDTree
from stardist import random_label_cmap

from chromapylot.core.core_types import AnalysisType, DataType
from chromapylot.core.data_manager import load_json, save_ecsv, DataManager
from chromapylot.parameters.params_manager import PipelineParams
from chromapylot.parameters.acquisition_params import AcquisitionParams
from chromapylot.parameters.matrix_params import MatrixParams
from chromapylot.core.run_args import RunArgs


def remove_duplicated_barcodes_to_simulate_pyhim_wrong_output(n_loc, group_list):
    # deep copy of group_list
    group_list_copy = group_list.copy()
    ow_rep = np.ones((n_loc, 2), dtype=int) * (-1)
    # iterates over traces to remove duplicated barcodes
    for trace_id, trace in enumerate(group_list):
        for idx, locus_id in enumerate(trace):
            prec_trace_id = ow_rep[locus_id, 0]
            prec_idx = ow_rep[locus_id, 1]
            if prec_trace_id != -1 and prec_idx != -1:
                group_list_copy[prec_trace_id][prec_idx] = -1
            ow_rep[locus_id, 0] = trace_id
            ow_rep[locus_id, 1] = idx
    # also remove the trace_id <= 1
    group_list_copy[0] = [-1]
    group_list_copy[1] = [-1]
    return group_list_copy


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
    trace_table.meta["comments"] = ["xyz_unit=micron", "genome_assembly=mm10"]
    return trace_table


def add_localization_to_trace_table(
    localization, trace_table, trace_buid, x, y, z, mask_id
):
    entry = [
        localization["Buid"],  # spot uid
        trace_buid,  # trace uid
        x,  # x, microns
        y,  # y, microns
        z,  # z, microns
        "xxxxx",  # chromosome
        0,  # start sequence
        999999999,  # end sequence
        localization["ROI #"],  # ROI number
        mask_id,  # Mask number
        localization["Barcode #"],  # Barcode name
        "x" * 20,  # label
    ]

    trace_table.add_row(entry)
    return trace_table


class BuildTrace3DModule(Module):
    def __init__(
        self,
        data_manager: DataManager,
        acquisition_params: AcquisitionParams,
        matrix_params: MatrixParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.TABLE_3D,
            output_type=DataType.TRACES_LIST_3D,
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
        self.pixel_size_z = acquisition_params.pixelSizeZ * acquisition_params.zBinning
        self.kdtree_distance_threshold_mum = matrix_params.KDtree_distance_threshold_mum
        self.dirname = "tracing"

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
        return [x * self.pixel_size_xy, y * self.pixel_size_xy, z * self.pixel_size_z]

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
        if "registered" not in localizations.meta["comments"]:
            print("WARNING: Localizations are not 3D registered!")

        print(f"> Aligning localizations to {nbr_of_masks} masks...")
        # loops over barcode Table rows
        for barcode_row in localizations:
            # gets xyz coordinates
            x_sub_pix = barcode_row["ycentroid"]
            y_sub_pix = barcode_row["xcentroid"]
            z_sub_pix = barcode_row["zcentroid"]

            # binarizes coordinate
            x_int = int(np.nan_to_num(x_sub_pix, nan=-1))
            y_int = int(np.nan_to_num(y_sub_pix, nan=-1))
            z_int = int(np.nan_to_num(z_sub_pix, nan=-1)) + int(self.z_offset)

            # if a barcode has coordinates with NaNs, it is assigned to background
            if x_int == -1 or y_int == -1 or z_int == -1:
                mask_id = 0
            elif (
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
                        y_sub_pix, x_sub_pix, z_sub_pix
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
                # Barcode outside the image, it is unassigned
                mask_id = -1
            barcode_nbr_by_mask_id[mask_id] += 1

        # Report part of the trace table built
        n_cells_assigned = np.count_nonzero(barcode_nbr_by_mask_id[1:-1])
        n_cells_unassigned = nbr_of_masks - n_cells_assigned
        n_barcodes_in_mask = sum(barcode_nbr_by_mask_id[1:-1])
        n_barcodes_in_background = barcode_nbr_by_mask_id[0]
        n_barcodes_unassigned = barcode_nbr_by_mask_id[-1]

        print(
            f"$ [Number of cells]\n\t| assigned: {n_cells_assigned}\n\t| unassigned: {n_cells_unassigned}"
        )
        print(
            f"$ [Number of barcodes]\n\t| in masks: {n_barcodes_in_mask}\n\t| in background: {n_barcodes_in_background}\n\t| outisde the image: {n_barcodes_unassigned}"
        )

        return trace_table

    def build_traces_by_kdtree(self, localizations):
        # initializes trace table
        trace_table = init_trace_table()
        n_localizations = len(localizations)
        print(f"> Building traces by KDTree for {n_localizations} localizations.")
        # build traces by spatial clustering
        coordinates = np.concatenate(
            [
                self.pixel_size_xy
                * localizations["ycentroid"].data.reshape(n_localizations, 1),
                self.pixel_size_xy
                * localizations["xcentroid"].data.reshape(n_localizations, 1),
                self.pixel_size_z
                * localizations["zcentroid"].data.reshape(n_localizations, 1),
            ],
            axis=1,
        )
        # gets tree of coordinates
        print(f"> Creating KDTree for 3 dimensions.")
        x_tree = KDTree(coordinates)

        ## set distance thresold
        r = self.kdtree_distance_threshold_mum
        # Groups points when they're less than r away
        points = [x_tree.query_ball_point(element, r, p=2.0) for element in coordinates]

        # Get unique groups
        groups = list({tuple(x) for x in points})
        group_list = [list(x) for x in groups]
        n_cells_assigned = len(group_list)

        group_list_copy = remove_duplicated_barcodes_to_simulate_pyhim_wrong_output(
            len(localizations), group_list
        )

        # iterates over traces
        for trace_id, trace in enumerate(group_list_copy):
            trace_buid = str(uuid.uuid4())
            # iterates over localizations in trace
            for locus_id in trace:
                if locus_id != -1:
                    x_nm, y_nm, z_nm = self.convert_coords(
                        localizations[locus_id]["xcentroid"],
                        localizations[locus_id]["ycentroid"],
                        localizations[locus_id]["zcentroid"],
                    )
                    # Adds localization to trace table
                    trace_table = add_localization_to_trace_table(
                        localizations[locus_id],
                        trace_table,
                        trace_buid,
                        x_nm,
                        y_nm,
                        z_nm,
                        trace_id,
                    )

        print(f"$ [Number of cells]\n\t| assigned: {n_cells_assigned}")
        return trace_table

    def run(self, localizations):
        output = []
        if "clustering" in self.tracing_method:
            print("Building cluster trace table.")
            trace_table = self.build_traces_by_kdtree(localizations)
            output.append(trace_table)
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

    def _save_one_trace_table(self, trace_table, input_path, method):
        base = os.path.basename(input_path).split(".")[0]
        roi_nbr = trace_table["ROI #"][0]
        barcode_2d_or_3d = "_".join(base.split("_")[1:])
        out_name = f"Trace_{barcode_2d_or_3d}_{method}_ROI-{roi_nbr}.ecsv"
        output_dir = self.data_m.output_folder
        table_path = os.path.join(output_dir, self.dirname, "data", out_name)
        save_ecsv(trace_table, table_path)
        png_path = os.path.join(
            output_dir, self.dirname, f"{out_name}_traces_XYZ_ROI{roi_nbr}.png"
        )
        mask = self.reference_data[method.split("-")[1]] if method != "KDtree" else None
        self._plot_traces(trace_table, png_path, mask)

    def _plot_traces(self, data, png_path, masks):
        pixel_size = [self.pixel_size_xy, self.pixel_size_xy, self.pixel_size_z]

        # initializes figure
        fig = plt.figure(constrained_layout=False)
        im_size = 20
        fig.set_size_inches((im_size * 2, im_size))
        gs = fig.add_gridspec(2, 2)
        ax = [
            fig.add_subplot(gs[:, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 1]),
        ]

        # defines variables
        x = data["x"]
        y = data["y"]
        z = data["z"]

        # colors = [i for i in range(len(np.unique(data["Barcode #"])))]
        unique_barcodes = np.unique(data["Barcode #"])
        output_array = range(unique_barcodes.shape[0])
        color_dict = {
            str(barcode): output
            for barcode, output in zip(unique_barcodes, output_array)
        }
        colors = [color_dict[str(x)] for x in data["Barcode #"]]
        titles = ["Z-projection", "X-projection", "Y-projection"]

        # plots masks if available
        if masks is not None and len(masks.shape) == 3:
            masks = np.max(masks, axis=0)
        if masks is not None and len(masks.shape) == 2:
            ax[0].imshow(masks, cmap=random_label_cmap(), alpha=0.3)

        # makes plot
        ax[0].scatter(
            x / pixel_size[0], y / pixel_size[1], s=5, c=colors, alpha=0.9, cmap="hsv"
        )
        ax[0].set_title(titles[0])

        ax[1].scatter(x, z, s=5, c=colors, alpha=0.9, cmap="hsv")
        ax[1].set_title(titles[1])

        ax[2].scatter(y, z, s=5, c=colors, alpha=0.9, cmap="hsv")
        ax[2].set_title(titles[2])

        fig.tight_layout()

        # calculates mean trace positions and sizes by looping over traces
        data_traces = data.group_by("Trace_ID")
        colors_traces = [i for i in range(len(np.unique(data_traces["Trace_ID"])))]
        cmap_traces = plt.cm.get_cmap("hsv", np.max(colors_traces))

        for trace, color in zip(data_traces.groups, colors_traces):
            # Plots polygons for each trace
            poly_coord = np.array(
                [
                    (trace["x"].data) / pixel_size[0],
                    (trace["y"].data) / pixel_size[1],
                ]
            ).T
            polygon = Polygon(
                poly_coord,
                closed=False,
                fill=False,
                edgecolor=cmap_traces(color),
                linewidth=1,
                alpha=1,
            )
            ax[0].add_patch(polygon)

        # saves output figure
        fig.savefig(png_path)
        plt.close(fig)
        print(f"Saved {png_path}.")

    def save_data(self, data, input_path, input_data):
        method_names = []
        if "clustering" in self.tracing_method:
            method_names.append("KDtree")
        if "masking" in self.tracing_method:
            for key in self.masks2process.keys():
                method_names.append(f"mask-{key}")
        for trace_table, method in zip(list(data), method_names):
            print(f"Saving {method} trace table.")
            self._save_one_trace_table(trace_table, input_path, method)

    def load_reference_data(self, paths: List[str]):
        if "masking" in self.tracing_method:
            for key, val in self.masks2process.items():
                for path in paths:
                    if val in path:
                        print(f"Loading {path} for mask {val}.")
                        self.reference_data[val] = np.load(path)
        else:
            print("No reference data needed for tracing method {self.tracing_method}.")


def main(command_line_args=None):
    begin_time = datetime.now()
    print(f"Start time: {begin_time}")

    # INITIALIZATION
    run_args = RunArgs(command_line_args)
    raw_params = load_json(os.path.join(run_args.input, "parameters.json"))
    pipe_params = PipelineParams(raw_params, AnalysisType.TRACE)
    mod = BuildTrace3DModule(pipe_params.acquisition, pipe_params.matrix)
    ref_files = run_args.ref_file.split(",") if run_args.ref_file else []

    # MODULE EXECUTION
    mod.load_reference_data(ref_files)
    input_data = mod.load_data(run_args.in_file)
    output_data = mod.run(input_data)
    mod.save_data(output_data, run_args.output, run_args.in_file)

    # TERMINATION
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    main()
