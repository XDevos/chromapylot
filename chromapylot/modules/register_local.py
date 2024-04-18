#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List

from tqdm import tqdm, trange
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.util.shape import view_as_blocks
from scipy.ndimage import shift as shift_image
from skimage import exposure
from skimage.metrics import normalized_root_mse
from astropy.table import Table, vstack
from dask.distributed import Lock

from chromapylot.modules.module import Module
from chromapylot.core.data_manager import DataManager, get_roi_number_from_image_path
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.core.core_types import DataType
from chromapylot.modules.register_global import image_adjust


class RegisterLocal(Module):
    def __init__(
        self, data_manager: DataManager, registration_params: RegistrationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
            supplementary_type=None,
        )
        self.dirname = "register_local"
        self.block_size_xy = registration_params.blockSizeXY
        self.upsample_factor = registration_params.upsample_factor
        self.ref_fiducial = registration_params.referenceFiducial

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path and good_path[-3:] == "tif":
            ref_3d = self.data_m.load_image_3d(good_path)
            # TODO: it's a (may be long) tempo fix to apply SkipModule at the ref 3d fiducial
            self.reference_data = ref_3d[::2, :, :]

        else:
            raise NotImplementedError("Reference data must be a 3D tif file")

    def run(self, data):
        # - break in blocks
        num_planes = data.shape[0]
        block_size = (num_planes, self.block_size_xy, self.block_size_xy)

        print("$ Breaking images into blocks")
        ref_blocks = view_as_blocks(
            self.reference_data, block_shape=block_size
        ).squeeze()
        img_blocks = view_as_blocks(data, block_shape=block_size).squeeze()

        # - loop thru blocks and calculates block shift in xyz:
        shift_matrices = [np.zeros(ref_blocks.shape[:2]) for _ in range(3)]

        for i in trange(ref_blocks.shape[0]):
            for j in range(ref_blocks.shape[1]):
                # - cross correlate in 3D to find 3D shift
                shifts_xyz, _, _ = phase_cross_correlation(
                    ref_blocks[i, j],
                    img_blocks[i, j],
                    upsample_factor=self.upsample_factor,
                )
                for matrix, _shift in zip(shift_matrices, shifts_xyz):
                    matrix[i, j] = _shift

        nrmse_matrices = self.calculate_nrmse_matrices(
            shift_matrices, ref_blocks, img_blocks
        )
        registration_table = self.create_registration_table(
            shift_matrices, nrmse_matrices
        )
        return registration_table

    def save_data(self, data, input_path):
        if data is None:
            return
        self._save_registration_table(data, input_path)
        # raw_img = np.load(input_path)
        # shifted_3d_img = shift_image(raw_img, data)
        # self._save_3d_alignments(block_ref, block_shifted)  # fig3 outputs
        # self._save_bkg_substracted()  # fig1 plot_4_images(...)
        # self._save_mse_blocks()  # fig5 plot_3d_shift_matrices(...)
        # self._save_shift_matrices(data)  # fig2 plot_3d_shift_matrices(...)

    def _save_registration_table(self, data, input_path):
        data = self.__add_cycle_roi_and_filename(data, input_path)
        out_path = os.path.join(
            self.data_m.output_folder, self.dirname, "data", "shifts_block3D.dat"
        )
        # cycle = DataManager.get_cycle_from_path(input_path)
        # roi = get_roi_number_from_image_path(input_path)

        try:
            with Lock(out_path):
                self.__save_registration_table(data, out_path)
        except RuntimeError:
            self.__save_registration_table(data, out_path)

    def __add_cycle_roi_and_filename(self, data, input_path):
        cycle = self.data_m.get_cycle_from_path(input_path)
        filename = os.path.basename(input_path)
        data["aligned file"] = [filename] * len(data)
        data["ROI #"] = get_roi_number_from_image_path(input_path)
        data["label"] = [cycle] * len(data)
        return data

    def __save_registration_table(self, data, out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        else:
            existing_table = Table.read(out_path, format="ascii")
            data = vstack([existing_table, data])
        data.write(out_path, format="ascii", overwrite=True)

    def create_registration_table(self, shift_matrices, nrmse_matrices):
        alignment_results_table = create_output_table()
        num_blocks = shift_matrices[0].shape[0]
        for i in range(num_blocks):
            for j in range(num_blocks):
                table_entry = [
                    self.data_m.get_3d_ref_filename(),
                    "",
                    int(self.block_size_xy),
                    0,
                    "",
                    i,
                    j,
                    shift_matrices[0][i, j],
                    shift_matrices[1][i, j],
                    shift_matrices[2][i, j],
                    nrmse_matrices[0][i, j],
                    nrmse_matrices[1][i, j],
                    nrmse_matrices[2][i, j],
                ]
                alignment_results_table.add_row(table_entry)
        return alignment_results_table

    def calculate_nrmse_matrices(self, shift_matrices, block_ref, block_target):
        nrmse_matrices = []
        for axis1 in range(3):
            number_blocks = block_ref.shape[0]
            block_sizes = list(block_ref.shape[2:])
            block_sizes.pop(axis1)
            img_sizes = [x * number_blocks for x in block_sizes]

            # gets ranges for slicing
            slice_coordinates = [
                [
                    range(x * block_size, (x + 1) * block_size)
                    for x in range(number_blocks)
                ]
                for block_size in block_sizes
            ]

            nrmse_as_blocks = np.zeros((number_blocks, number_blocks))

            # blank image for blue channel to show borders between blocks
            blue = np.zeros(block_sizes)
            blue[0, :], blue[:, 0], blue[:, -1], blue[-1, :] = [0.5] * 4

            # reassembles image
            # takes one plane block
            for i, i_slice in enumerate(tqdm(slice_coordinates[0])):
                for j, j_slice in enumerate(slice_coordinates[1]):
                    imgs = [block_ref[i, j]]
                    if shift_matrices is not None:
                        shift_3d = np.array(
                            [x[i, j] for x in shift_matrices]
                        )  # gets 3D shift from block decomposition
                        imgs.append(
                            shift_image(block_target[i, j], shift_3d)
                        )  # realigns and appends to image list
                    else:
                        imgs.append(
                            block_target[i, j]
                        )  # appends original target with no re-alignment

                    imgs = [np.sum(x, axis=axis1) for x in imgs]  # projects along axis1
                    imgs = [
                        exposure.rescale_intensity(x, out_range=(0, 1)) for x in imgs
                    ]  # rescales intensity values
                    imgs = [
                        image_adjust(x, lower_threshold=0.5, higher_threshold=0.9999)[0]
                        for x in imgs
                    ]  # adjusts pixel intensities

                    nrmse_as_blocks[i, j] = normalized_root_mse(
                        imgs[0], imgs[1], normalization="euclidean"
                    )
            nrmse_matrices.append(nrmse_as_blocks)

        return nrmse_matrices


def create_output_table():
    return Table(
        names=(
            "reference file",
            "aligned file",
            "blockXY",
            "ROI #",
            "label",
            "block_i",
            "block_j",
            "shift_z",
            "shift_x",
            "shift_y",
            "quality_xy",
            "quality_zy",
            "quality_zx",
        ),
        dtype=(
            "S2",
            "S2",
            "int",
            "int",
            "S2",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )
