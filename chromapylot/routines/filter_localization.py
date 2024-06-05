#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chromapylot.routines.routine import Routine
from chromapylot.core.data_manager import DataManager
from chromapylot.core.core_types import DataType
from chromapylot.parameters.segmentation_params import SegmentationParams
from chromapylot.parameters.matrix_params import MatrixParams
from tqdm import trange
import glob
from astropy.table import Table as table
from chromapylot.core.data_manager import save_ecsv


class FilterLocalization(Routine):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
        matrix_params: MatrixParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.TABLE,
            output_type=DataType.TABLE,
            reference_type=None,
            supplementary_type=None,
        )
        self.flux_min_2d = matrix_params.flux_min
        self.flux_min_3d = matrix_params.flux_min_3D

    def load_data(self, input_path):
        return self.data_manager.load_table(input_path)

    def run(self, barcode_map):
        rows_to_remove = []
        n_barcodes = len(barcode_map)
        flux_min = self._get_flux_min(barcode_map)
        print(f"$ Minimum flux: {flux_min}")
        for i in trange(n_barcodes):
            if not (barcode_map["flux"][i] > flux_min):
                rows_to_remove.append(i)
        barcode_map.remove_rows(rows_to_remove)
        print(
            f"$ Removed {len(rows_to_remove)} barcode localizations from table out of {n_barcodes}."
        )
        return barcode_map

    def _get_flux_min(self, barcode_map):
        if barcode_map["zcentroid"][0] is not None:
            return self.flux_min_3d
        return self.flux_min_2d

    def save_data(self, data, input_path, input_data, supplementary_data):
        self._save_old_table(input_path, input_data)
        save_ecsv(data, input_path, comments="filtered")

    def _save_old_table(self, input_path, input_data):
        existing_versions = glob.glob(input_path.split(".")[0] + "_version_*.dat")
        if len(existing_versions) < 1:
            new_version = 0
        else:
            version_numbers = [
                int(x.split("_version_")[1].split(".")[0]) for x in existing_versions
            ]
            new_version = max(version_numbers) + 1 if version_numbers else 0
        new_file = input_path.split(".dat")[0] + "_version_" + str(new_version) + ".dat"
        save_ecsv(input_data, new_file)
